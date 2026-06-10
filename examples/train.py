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
            nn.Linear(num_obs, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions)
        )
        
        # Value Function (Critic)
        self.critic = nn.Sequential(
            nn.Linear(num_obs, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 1)
        )
        
        # Action standard deviation
        self.std = nn.Parameter(torch.zeros(num_actions))
        
    def forward(self, obs):
        mean = self.actor(obs)
        std = self.std.exp()
        return mean, std
    
    def evaluate(self, obs, action):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return log_prob, value, entropy

    def act(self, obs):
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
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
def compute_reward_and_done(env, obs):
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
    
    # Variable posture reward (speed-dependent std)
    total_speed = torch.norm(commands[:, :2], dim=1) + torch.abs(commands[:, 2])
    standing_mask = (total_speed < 0.5).float().unsqueeze(1)
    running_mask = (total_speed >= 1.5).float().unsqueeze(1)
    walking_mask = 1.0 - standing_mask - running_mask
    std = 0.05 * standing_mask + 0.2 * walking_mask + 0.3 * running_mask
    pose_error = torch.mean(torch.square(obs[:, 12:41]) / (std**2), dim=1)
    rew_pose = torch.exp(-pose_error)
    
    # 3. Penalties
    rew_action_rate = torch.sum(torch.square(env.ctrl - env.last_actions), dim=0)
    rew_joint_vel = torch.sum(torch.square(joint_vel), dim=1)
    
    current_joint_pos = env.qpos[7:36, :]
    out_of_bounds = ((current_joint_pos < env.joint_pos_lower) | (current_joint_pos > env.joint_pos_upper)).float()
    rew_dof_limits = torch.sum(out_of_bounds, dim=0)
    
    # 4. Combine (weights match mjlab/tasks/velocity/velocity_env_cfg.py)
    reward = (
        2.0 * rew_lin_vel + 
        2.0 * rew_ang_vel + 
        1.0 * rew_orient +
        1.0 * rew_pose -
        1.0 * rew_dof_limits -
        0.1 * rew_action_rate - 
        0.001 * rew_joint_vel
    )
    
    # 4. Termination (Base height < 0.45 or > 1.0)
    base_z = env.qpos[2, :]
    done = ((base_z < 0.45) | (base_z > 1.0)).to(torch.uint8)
    
    return reward, done

def resample_commands(env, env_ids):
    if len(env_ids) == 0:
        return
    # x: [-1.0, 1.0], y: [-0.5, 0.5], yaw: [-1.0, 1.0]
    env.commands[0, env_ids] = torch.rand(len(env_ids), device=env.device) * 2.0 - 1.0
    env.commands[1, env_ids] = torch.rand(len(env_ids), device=env.device) * 1.0 - 0.5
    env.commands[2, env_ids] = torch.rand(len(env_ids), device=env.device) * 2.0 - 1.0

# ==============================================================================
# Main Training Loop
# ==============================================================================
def train():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # PPO Hyperparams
    num_envs = 4096
    num_steps = 24       # steps per rollout
    nsub = 10            # 2e-3 dt * 10 = 20ms policy dt
    epochs = 5
    batch_size = 4096 * 24
    mini_batch_size = batch_size // 4
    clip_param = 0.2
    max_grad_norm = 1.0
    lr = 3e-4
    iterations = 1000
    
    env = G1Sim(nenv=num_envs, device=device)
    
    # Provide simple forward walking commands initially
    resample_commands(env, torch.arange(num_envs, device=device))
    
    ac = ActorCritic(num_obs=99, num_actions=29).to(device)
    optimizer = optim.Adam(ac.parameters(), lr=lr)
    
    buffer = RolloutBuffer(num_envs, num_steps, 99, 29, device)
    
    env.reset_all(noise=0.1)
    obs = env.get_obs()
    
    # Track command resample interval
    command_timeout = torch.zeros(num_envs, dtype=torch.int32, device=device)
    
    for it in range(iterations):
        t0 = time.time()
        
        # Rollout Phase
        buffer.step = 0
        total_reward = 0.0
        
        for step in range(num_steps):
            with torch.no_grad():
                actions, log_probs, values = ac.act(obs)
                
            # Scale network output to joint positions (PD targets)
            env.ctrl[:] = env.default_joint_pos + actions.T * 0.25
            
            env.step(nsub=nsub)
            next_obs = env.get_obs()
            
            rewards, dones = compute_reward_and_done(env, next_obs)
            total_reward += rewards.mean().item()
            
            command_timeout -= 1
            
            # Reset environments that died
            if dones.any():
                done_indices = dones.nonzero(as_tuple=True)[0]
                env.reset_done(dones, noise=0.1)
                resample_commands(env, done_indices)
                command_timeout[done_indices] = torch.randint(150, 250, (len(done_indices),), device=device, dtype=torch.int32)
                
            # Resample commands for envs that hit timeout but didn't die
            timeout_indices = (command_timeout <= 0).nonzero(as_tuple=True)[0]
            if len(timeout_indices) > 0:
                resample_commands(env, timeout_indices)
                command_timeout[timeout_indices] = torch.randint(150, 250, (len(timeout_indices),), device=device, dtype=torch.int32)
                
            # Recompute obs after resets/resamples
            if dones.any() or len(timeout_indices) > 0:
                next_obs = env.get_obs()
                
            buffer.add(obs, actions, log_probs, rewards, dones, values)
            obs = next_obs
            
        # PPO Update Phase
        with torch.no_grad():
            _, _, next_values = ac.act(obs)
            returns, advs = buffer.compute_returns(next_values, dones)
            
            # Normalize advantages
            advs = (advs - advs.mean()) / (advs.std() + 1e-8)
            
        # Flatten buffers
        b_obs = buffer.obs.reshape(-1, 99)
        b_actions = buffer.actions.reshape(-1, 29)
        b_log_probs = buffer.log_probs.reshape(-1)
        b_returns = returns.reshape(-1)
        b_advs = advs.reshape(-1)
        
        # Epochs
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
                
                critic_loss = 0.5 * (b_returns[mb_idx] - values).pow(2).mean()
                entropy_loss = -0.01 * entropy.mean()
                
                loss = actor_loss + critic_loss + entropy_loss
                
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(ac.parameters(), max_grad_norm)
                optimizer.step()
                
        t1 = time.time()
        fps = (num_envs * num_steps) / (t1 - t0)
        
        print(f"Iter: {it:03d} | Reward: {total_reward/num_steps:.3f} | FPS: {fps:.0f}")

    # Save policy
    os.makedirs(os.path.join(HERE, "..", "build"), exist_ok=True)
    save_path = os.path.join(HERE, "..", "build", "policy.pt")
    torch.save(ac.state_dict(), save_path)
    print(f"Saved policy to {save_path}")

if __name__ == "__main__":
    train()
