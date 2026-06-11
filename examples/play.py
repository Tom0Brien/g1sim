#!/usr/bin/env python3
import torch
import mujoco
import mujoco.viewer
import time
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, ".."))

from src.g1sim import G1Sim
from examples.train import ActorCritic

def play():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    policy_path = os.path.join(HERE, "..", "build", "policy.pt")
    
    if not os.path.exists(policy_path):
        print(f"Error: Could not find trained policy at {policy_path}")
        print("Please run `uv run python examples/train.py` first.")
        sys.exit(1)
        
    print(f"Loading policy from {policy_path}")
    ac = ActorCritic(num_obs=99, num_actions=29).to(device)
    ac.load_state_dict(torch.load(policy_path, map_location=device))
    ac.eval()
    
    # Initialize a single environment
    env = G1Sim(nenv=1, device=device)
    env.reset_all(noise=0.0)
    
    # Set command (forward walking)
    env.commands[0, 0] = 0.0
    env.commands[1, 0] = 0.0
    env.commands[2, 0] = 0.0
    
    # Setup MuJoCo viewer (for rendering only)
    scene_xml = os.path.join(HERE, "..", "model", "scene.xml")
    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    
    # Sync visual model initial state
    d.qpos[:] = env.qpos[:, 0].cpu().numpy()
    mujoco.mj_forward(m, d)
    
    nsub = 20
    control_dt = 1e-3 * nsub  # 20ms policy step
    
    print("Launching viewer.")
    print("Controls:")
    print("  Space: Pause/Unpause")
    print("  R: Reset Environment")
    print("  W/S: Increase/Decrease Forward Velocity (Vx)")
    print("  A/D: Increase/Decrease Lateral Velocity (Vy)")
    print("  Q/E: Increase/Decrease Yaw Velocity (Wz)")
    
    paused = [False]
    def key_callback(keycode):
        try:
            char = chr(keycode).upper()
        except ValueError:
            char = ''
            
        if keycode == ord(' '):
            paused[0] = not paused[0]
        elif char == 'R':
            env.reset_all(noise=0.0)
            env.commands[0, 0] = 0.0
            env.commands[1, 0] = 0.0
            env.commands[2, 0] = 0.0
        elif char == 'W':
            env.commands[0, 0] = torch.clamp(env.commands[0, 0] + 0.1, -1.0, 2.0)
        elif char == 'S':
            env.commands[0, 0] = torch.clamp(env.commands[0, 0] - 0.1, -1.0, 2.0)
        elif char == 'A':
            env.commands[1, 0] = torch.clamp(env.commands[1, 0] + 0.1, -1.0, 1.0)
        elif char == 'D':
            env.commands[1, 0] = torch.clamp(env.commands[1, 0] - 0.1, -1.0, 1.0)
        elif char == 'Q':
            env.commands[2, 0] = torch.clamp(env.commands[2, 0] + 0.2, -1.0, 1.0)
        elif char == 'E':
            env.commands[2, 0] = torch.clamp(env.commands[2, 0] - 0.2, -1.0, 1.0)
            
        if char in ['R', 'W', 'S', 'A', 'D', 'Q', 'E']:
            print(f"Command -> Vx: {env.commands[0, 0].item():.1f}, Vy: {env.commands[1, 0].item():.1f}, Wz: {env.commands[2, 0].item():.1f}")

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            
            if not paused[0]:
                with torch.no_grad():
                    obs = env.get_obs()
                    
                    # Deterministic action for evaluation
                    obs_norm = ac.normalize_obs(obs)
                    action = ac.actor(obs_norm)
                    
                    env.actions = action.T
                    env.ctrl[:] = env.default_joint_pos + env.actions * env.action_scale
                    env.step(nsub=nsub)
                    
                    # Auto-reset if fallen (orientation-based)
                    q = env.qpos[3:7, :]
                    qw, qx, qy, qz = q[0], q[1], q[2], q[3]
                    # Projected gravity z-component
                    grav_z = -(1.0 - 2.0*(qx*qx + qy*qy))
                    tilt = torch.acos(torch.clamp(-grav_z, -1.0, 1.0)).abs()
                    if torch.isnan(env.qpos).any() or torch.abs(env.qpos).max() > 1000.0:
                        print("Invalid state detected! Simulation blew up.")
                        env.reset_all(noise=0.0)
                        torch.cuda.current_stream().synchronize()
                        continue
                    if tilt.item() > 1.2217:  # 70 degrees
                        env.reset_all(noise=0.0)
                        torch.cuda.current_stream().synchronize()
                
                # Sync state back to MuJoCo for rendering
                d.qpos[:] = env.qpos[:, 0].cpu().numpy()
                mujoco.mj_forward(m, d)
                viewer.sync()
                
            # Timekeeping to match real-time (roughly)
            time_until_next_step = control_dt - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    play()
