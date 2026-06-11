import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))

from src.g1sim import G1Sim

# ==============================================================================
# PPO Actor-Critic Network
# ==============================================================================
class ActorCritic(nn.Module):
    def __init__(self, num_obs, num_actions):
        super().__init__()
        
        # Policy (Actor)
        self.actor = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions)
        )
        
        # Value Function (Critic)
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Action standard deviation
        self.std = nn.Parameter(torch.zeros(num_actions))
        
        self.register_buffer("obs_mean", torch.zeros(num_obs))
        self.register_buffer("obs_var", torch.ones(num_obs))
        self.register_buffer("obs_count", torch.tensor(1e-4))
        self.clip_obs = 5.0
        
    def update_obs_norm(self, obs):
        batch_mean = obs.mean(dim=0)
        batch_var = obs.var(dim=0, unbiased=False)
        batch_count = obs.shape[0]
        
        delta = batch_mean - self.obs_mean
        tot_count = self.obs_count + batch_count
        
        new_mean = self.obs_mean + delta * batch_count / tot_count
        m_a = self.obs_var * self.obs_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.obs_count * batch_count / tot_count
        new_var = M2 / tot_count
        
        self.obs_mean.copy_(new_mean)
        self.obs_var.copy_(new_var)
        self.obs_count.copy_(tot_count)

    def normalize_obs(self, obs):
        return torch.clamp((obs - self.obs_mean) / torch.sqrt(self.obs_var + 1e-8), -self.clip_obs, self.clip_obs)

    def forward(self, obs):
        obs_norm = self.normalize_obs(obs)
        mean = self.actor(obs_norm)
        std = self.std.exp()
        return mean, std
    
    def evaluate(self, obs, action):
        obs_norm = self.normalize_obs(obs)
        mean = self.actor(obs_norm)
        std = self.std.exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs_norm).squeeze(-1)
        return log_prob, value, entropy

    def act(self, obs, update_norm=True):
        if update_norm:
            self.update_obs_norm(obs)
        obs_norm = self.normalize_obs(obs)
        mean = self.actor(obs_norm)
        std = self.std.exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs_norm).squeeze(-1)
        return action, log_prob, value

# ==============================================================================
# Rollout Buffer
# ==============================================================================
class RolloutBuffer:
    def __init__(self, num_envs, num_steps, num_obs, num_actions, device):
        self.obs = torch.zeros((num_steps, num_envs, num_obs), device=device)
        self.actions = torch.zeros((num_steps, num_envs, num_actions), device=device)
        self.log_probs = torch.zeros((num_steps, num_envs), device=device)
        self.rewards = torch.zeros((num_steps, num_envs), device=device)
        self.dones = torch.zeros((num_steps, num_envs), device=device)
        self.values = torch.zeros((num_steps, num_envs), device=device)
        self.step = 0

    def add(self, obs, action, log_prob, reward, done, value):
        self.obs[self.step] = obs
        self.actions[self.step] = action
        self.log_probs[self.step] = log_prob
        self.rewards[self.step] = reward
        self.dones[self.step] = done
        self.values[self.step] = value
        self.step += 1

    def compute_returns(self, next_value, next_done, gamma=0.99, gae_lambda=0.95):
        returns = torch.zeros_like(self.rewards)
        advs = torch.zeros_like(self.rewards)
        last_gaelam = 0
        
        for t in reversed(range(self.step)):
            if t == self.step - 1:
                next_non_terminal = 1.0 - next_done.float()
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1].float()
                next_values = self.values[t + 1]
                
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            advs[t] = last_gaelam = delta + gamma * gae_lambda * next_non_terminal * last_gaelam
            
        returns = advs + self.values
        return returns, advs

# ==============================================================================
# Reward & Termination Logic
# ==============================================================================
def compute_reward_and_done(env, obs, std_walking, std_running, actions, prev_actions):
    # obs layout:
    # [0:3] commands, [3:6] proj_gravity, [6:9] lin_vel, [9:12] ang_vel
    # [12:41] joint_pos_err, [41:70] joint_vel, [70:99] last_action
    
    commands = obs[:, 0:3]
    proj_gravity = obs[:, 3:6]
    lin_vel = obs[:, 6:9]
    ang_vel = obs[:, 9:12]
    joint_vel = obs[:, 41:70]
    
    # 1. Velocity Tracking (mjlab formulation)
    xy_error = torch.sum(torch.square(commands[:, :2] - lin_vel[:, :2]), dim=1)
    z_error = torch.square(lin_vel[:, 2])
    lin_vel_error = xy_error + z_error
    rew_lin_vel = torch.exp(-lin_vel_error / 0.25)
    
    yaw_error = torch.square(commands[:, 2] - ang_vel[:, 2])
    roll_pitch_error = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    ang_vel_error = yaw_error + roll_pitch_error
    rew_ang_vel = torch.exp(-ang_vel_error / 0.5)
    
    # 2. Posture/Orientation
    orient_error = torch.sum(torch.square(proj_gravity[:, :2]), dim=1)
    rew_orient = torch.exp(-orient_error / 0.2)
    
    # Variable posture reward
    total_speed = torch.norm(commands[:, :2], dim=1) + torch.abs(commands[:, 2])
    standing_mask = (total_speed < 0.05).float().unsqueeze(1)
    running_mask = (total_speed >= 1.5).float().unsqueeze(1)
    walking_mask = 1.0 - standing_mask - running_mask
    std = 0.05 * standing_mask + std_walking * walking_mask + std_running * running_mask
    pose_error = torch.mean(torch.square(obs[:, 12:41]) / (std**2), dim=1)
    rew_pose = torch.exp(-pose_error)
    
    # 3. Penalties
    # Penalize the difference in network actions to suppress high frequency vibration
    rew_action_rate = torch.sum(torch.square(actions - prev_actions), dim=1)
    
    # Body angular velocity penalty
    rew_body_ang_vel = torch.sum(torch.square(ang_vel[:, :2]), dim=1)
    
    current_joint_pos = env.qpos[7:36, :]
    joint_range = env.joint_pos_upper - env.joint_pos_lower
    soft_lower = env.joint_pos_lower + 0.05 * joint_range
    soft_upper = env.joint_pos_upper - 0.05 * joint_range
    out_of_bounds = torch.clamp(current_joint_pos - soft_upper, min=0.0) + torch.clamp(soft_lower - current_joint_pos, min=0.0)
    rew_dof_limits = torch.sum(out_of_bounds, dim=0)
    
    # 4. Foot Rewards (using CUDA kinematics)
    left_z = env.foot_pos[2, :]
    right_z = env.foot_pos[5, :]
    left_vel_xy_sq = torch.square(env.foot_vel[0, :]) + torch.square(env.foot_vel[1, :])
    right_vel_xy_sq = torch.square(env.foot_vel[3, :]) + torch.square(env.foot_vel[4, :])
    left_vel_xy_norm = torch.sqrt(left_vel_xy_sq + 1e-6)
    right_vel_xy_norm = torch.sqrt(right_vel_xy_sq + 1e-6)
    
    cmd_active = (total_speed > 0.05).float()
    rew_foot_clearance = (torch.abs(left_z - 0.1) * left_vel_xy_norm + torch.abs(right_z - 0.1) * right_vel_xy_norm) * cmd_active
    
    left_contact = (torch.sum(env.contact_forces[0:4, :], dim=0) > 1.0).float()
    right_contact = (torch.sum(env.contact_forces[4:8, :], dim=0) > 1.0).float()
    rew_foot_slip = (left_vel_xy_sq * left_contact + right_vel_xy_sq * right_contact) * cmd_active
    
    # 5. Combine (full mjlab suite)
    reward = (
        2.0 * rew_lin_vel + 
        2.0 * rew_ang_vel + 
        1.0 * rew_orient +
        1.0 * rew_pose -
        1.0 * rew_dof_limits -
        2.0 * rew_foot_clearance -
        0.5 * rew_foot_slip -
        0.05 * rew_action_rate -
        0.05 * rew_body_ang_vel
    )
    
    # 4. Termination (Tilt > 70 degrees)
    limit_angle = 70.0 * 3.14159 / 180.0
    tilt = torch.acos(torch.clamp(-proj_gravity[:, 2], -1.0, 1.0)).abs()
    done = (tilt > limit_angle).to(torch.uint8)
    
    return reward, done

def resample_commands(env, env_ids):
    if len(env_ids) == 0:
        return
    # x: [-1.0, 1.0], y: [-1.0, 1.0], yaw: [-0.5, 0.5] (matched to mjlab initially)
    env.commands[0, env_ids] = torch.rand(len(env_ids), device=env.device) * 2.0 - 1.0
    env.commands[1, env_ids] = torch.rand(len(env_ids), device=env.device) * 2.0 - 1.0
    env.commands[2, env_ids] = torch.rand(len(env_ids), device=env.device) * 1.0 - 0.5

# ==============================================================================
# Main Training Loop
# ==============================================================================
def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # PPO Hyperparams
    num_envs = 4096
    num_steps = 24       # steps per rollout
    nsub = 20            # 1e-3 dt * 20 = 20ms policy dt
    epochs = 5
    batch_size = 4096 * 24
    mini_batch_size = batch_size // 4
    clip_param = 0.2
    max_grad_norm = 1.0
    lr = 1e-3            # Matched mjlab
    desired_kl = 0.01    # Matched mjlab
    iterations = 30000   # Matched mjlab
    
    env = G1Sim(nenv=num_envs, device=device)
    
    # Provide simple forward walking commands initially
    resample_commands(env, torch.arange(num_envs, device=device))
    
    ac = ActorCritic(num_obs=99, num_actions=29).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=lr)
    
    buffer = RolloutBuffer(num_envs, num_steps, 99, 29, device)
    
    env.reset_all(noise=0.1)
    obs = env.get_obs()
    
    # Variable posture standard deviations
    std_walking = torch.tensor([
        0.3, 0.15, 0.15, 0.35, 0.25, 0.1,  # left leg
        0.3, 0.15, 0.15, 0.35, 0.25, 0.1,  # right leg
        0.2, 0.08, 0.1,                    # waist
        0.15, 0.15, 0.1, 0.15, 0.3, 0.3, 0.3, # left arm
        0.15, 0.15, 0.1, 0.15, 0.3, 0.3, 0.3  # right arm
    ], device=device).unsqueeze(0)
    
    std_running = torch.tensor([
        0.5, 0.2, 0.2, 0.6, 0.35, 0.15,
        0.5, 0.2, 0.2, 0.6, 0.35, 0.15,
        0.3, 0.08, 0.2,
        0.5, 0.2, 0.15, 0.35, 0.3, 0.3, 0.3,
        0.5, 0.2, 0.15, 0.35, 0.3, 0.3, 0.3
    ], device=device).unsqueeze(0)

    # Track command resample interval, episode length, and push interval
    command_timeout = torch.zeros(num_envs, dtype=torch.int32, device=device)
    env_episode_length = torch.zeros(num_envs, dtype=torch.int32, device=device)
    push_interval = torch.randint(50, 150, (num_envs,), device=device, dtype=torch.int32)
    max_episode_length = 1000  # 20 seconds at 20ms policy dt
    
    for it in range(iterations):
        t0 = time.time()
        
        # Rollout Phase
        buffer.step = 0
        total_reward = 0.0
        
        # Initialize prev_actions for action rate penalty
        prev_actions = torch.zeros((num_envs, 29), device=device)
        
        for step in range(num_steps):
            with torch.no_grad():
                actions, log_probs, values = ac.act(obs)
                
            # Scale network output to joint positions (PD targets)
            env.ctrl[:] = env.default_joint_pos + actions.T * env.action_scale
            
            # Apply push perturbations
            push_interval -= 1
            push_indices = (push_interval <= 0).nonzero(as_tuple=True)[0]
            if len(push_indices) > 0:
                env.qvel[0, push_indices] += torch.rand(len(push_indices), device=device) * 1.0 - 0.5
                env.qvel[1, push_indices] += torch.rand(len(push_indices), device=device) * 1.0 - 0.5
                env.qvel[2, push_indices] += torch.rand(len(push_indices), device=device) * 0.8 - 0.4
                env.qvel[3, push_indices] += torch.rand(len(push_indices), device=device) * 1.04 - 0.52
                env.qvel[4, push_indices] += torch.rand(len(push_indices), device=device) * 1.04 - 0.52
                env.qvel[5, push_indices] += torch.rand(len(push_indices), device=device) * 1.56 - 0.78
                push_interval[push_indices] = torch.randint(50, 150, (len(push_indices),), device=device, dtype=torch.int32)
            
            env.step(nsub=nsub)
            next_obs = env.get_obs()
            
            rewards, dones = compute_reward_and_done(env, next_obs, std_walking, std_running, actions, prev_actions)
            prev_actions = actions.clone()
            total_reward += rewards.mean().item()
            
            command_timeout -= 1
            env_episode_length += 1
            
            timeouts = (env_episode_length >= max_episode_length).to(torch.uint8)
            resets = dones | timeouts
            
            # Reset environments that died or timed out
            if resets.any():
                reset_indices = resets.nonzero(as_tuple=True)[0]
                env.reset_done(resets, noise=0.1)
                resample_commands(env, reset_indices)
                command_timeout[reset_indices] = torch.randint(150, 250, (len(reset_indices),), device=device, dtype=torch.int32)
                env_episode_length[reset_indices] = 0
                push_interval[reset_indices] = torch.randint(50, 150, (len(reset_indices),), device=device, dtype=torch.int32)
                prev_actions[reset_indices] = 0.0
                
            # Resample commands for envs that hit timeout but didn't die
            command_indices = (command_timeout <= 0).nonzero(as_tuple=True)[0]
            if len(command_indices) > 0:
                resample_commands(env, command_indices)
                command_timeout[command_indices] = torch.randint(150, 250, (len(command_indices),), device=device, dtype=torch.int32)
                
            # Recompute obs after resets/resamples
            if resets.any() or len(command_indices) > 0:
                next_obs = env.get_obs()
                
            buffer.add(obs, actions, log_probs, rewards, dones, values)
            obs = next_obs
            
        # PPO Update Phase
        with torch.no_grad():
            _, _, next_values = ac.act(obs, update_norm=False)
            returns, advs = buffer.compute_returns(next_values, dones)
            
            # Normalize advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            
        # Flatten buffers
        b_obs = buffer.obs.reshape(-1, 99)
        b_actions = buffer.actions.reshape(-1, 29)
        b_log_probs = buffer.log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advs = advs.reshape(-1)
        b_values = buffer.values.reshape(-1)
        
        # Epochs
        mean_kl = 0.0
        for epoch in range(epochs):
            indices = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]
                
                new_log_probs, values, entropy = ac.evaluate(b_obs[mb_idx], b_actions[mb_idx])
                
                ratio = torch.exp(new_log_probs - b_log_probs[mb_idx])
                surr1 = ratio * b_advs[mb_idx]
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * b_advs[mb_idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_pred_clipped = b_values[mb_idx] + (values - b_values[mb_idx]).clamp(-clip_param, clip_param)
                value_losses = (values - b_returns[mb_idx]).pow(2)
                value_losses_clipped = (value_pred_clipped - b_returns[mb_idx]).pow(2)
                critic_loss = torch.max(value_losses, value_losses_clipped).mean()
                
                entropy_loss = -0.01 * entropy.mean()
                
                loss = actor_loss + 1.0 * critic_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()
                
                with torch.no_grad():
                    kl = (b_log_probs[mb_idx] - new_log_probs).mean()
                    mean_kl += kl.item()
                    
        mean_kl /= (epochs * (batch_size // mini_batch_size))
        
        # Adaptive LR
        if mean_kl > desired_kl * 2.0:
            lr = max(1e-5, lr / 1.5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif mean_kl < desired_kl / 2.0 and lr < 1e-2:
            lr = min(1e-2, lr * 1.5)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                
        t1 = time.time()
        fps = (num_envs * num_steps) / (t1 - t0)
        
        print(f"Iter: {it:03d} | Reward: {total_reward/num_steps:.3f} | FPS: {fps:.0f} | KL: {mean_kl:.4f} | LR: {lr:.2e}")
        
        # Periodic saving
        if (it + 1) % 100 == 0:
            os.makedirs(os.path.join(HERE, "..", "build"), exist_ok=True)
            save_path = os.path.join(HERE, "..", "build", "policy.pt")
            torch.save(ac.state_dict(), save_path)
            print(f"Saved intermediate policy to {save_path}")

    # Final Save policy
    os.makedirs(os.path.join(HERE, "..", "build"), exist_ok=True)
    save_path = os.path.join(HERE, "..", "build", "policy.pt")
    torch.save(ac.state_dict(), save_path)
    print(f"Saved final policy to {save_path}")

if __name__ == "__main__":
    train()
