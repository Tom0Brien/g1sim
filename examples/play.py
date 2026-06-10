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
    env.commands[0, 0] = 0.5
    env.commands[1, 0] = 0.0
    env.commands[2, 0] = 0.0
    
    # Setup MuJoCo viewer (for rendering only)
    scene_xml = os.path.join(HERE, "..", "model", "scene.xml")
    m = mujoco.MjModel.from_xml_path(scene_xml)
    d = mujoco.MjData(m)
    
    # Sync visual model initial state
    d.qpos[:] = env.qpos[:, 0].cpu().numpy()
    mujoco.mj_forward(m, d)
    
    nsub = 10
    control_dt = 2e-3 * nsub  # 20ms policy step
    
    print("Launching viewer. Press Space to pause, R to reset.")
    
    paused = [False]
    def key_callback(keycode):
        if keycode == ord(' '):
            paused[0] = not paused[0]
        elif keycode == ord('R') or keycode == ord('r'):
            env.reset_all(noise=0.0)

    with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            
            if not paused[0]:
                with torch.no_grad():
                    obs = env.get_obs()
                    
                    # Deterministic action for evaluation
                    action, _ = ac.forward(obs)
                    
                    env.ctrl[:] = env.default_joint_pos + action.T * 0.25
                    env.step(nsub=nsub)
                    
                    # Auto-reset if fallen
                    base_z = env.qpos[2, 0].item()
                    if base_z < 0.45 or base_z > 1.0:
                        env.reset_all(noise=0.0)
                
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
