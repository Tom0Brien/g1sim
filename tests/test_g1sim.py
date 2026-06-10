import torch
import pytest
import os
import ctypes
from src.g1sim import G1Sim

@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device("cuda:0")

def test_g1sim_init(device):
    env = G1Sim(nenv=10, device=device)
    assert env.qpos.shape == (36, 10)
    assert env.qvel.shape == (35, 10)
    assert env.anchor.shape == (16, 10)
    assert env.ctrl.shape == (29, 10)
    assert env.done.shape == (10,)

def test_g1sim_reset_all(device):
    env = G1Sim(nenv=5, device=device)
    # Fill with garbage
    env.qpos.fill_(999.0)
    env.qvel.fill_(999.0)
    
    # Reset
    env.reset_all(seed=42, noise=0.0, drop=0.0)
    torch.cuda.synchronize()
    
    # Check that it's no longer garbage, and velocities are 0
    assert torch.all(env.qvel == 0.0)
    assert not torch.any(env.qpos == 999.0)
    
    # Base height (index 2) should be roughly 0.79 (the stand pose)
    assert torch.allclose(env.qpos[2, :], torch.tensor(0.79, device=device), atol=0.05)

def test_g1sim_step(device):
    env = G1Sim(nenv=2, device=device)
    env.reset_all(seed=42, noise=0.0, drop=0.0)
    torch.cuda.synchronize()
    
    qpos_initial = env.qpos.clone()
    
    # Step physics
    env.step(nsub=10)
    torch.cuda.synchronize()
    
    # Positions should have changed due to gravity/PD
    assert not torch.allclose(env.qpos, qpos_initial)

def test_g1sim_selective_reset(device):
    env = G1Sim(nenv=2, device=device)
    env.reset_all(seed=42, noise=0.0, drop=0.0)
    torch.cuda.synchronize()
    
    # Step to change state
    env.step(nsub=50)
    torch.cuda.synchronize()
    
    qpos_stepped = env.qpos.clone()
    
    # Mark only environment 1 as done, leave 0 alone
    env.done[0] = 0
    env.done[1] = 1
    
    # Reset with a drop so the height is obviously different
    env.reset_done(env.done, seed=42, noise=0.0, drop=1.0)
    torch.cuda.synchronize()
    
    # Environment 0 should be exactly where it was before reset
    assert torch.allclose(env.qpos[:, 0], qpos_stepped[:, 0])
    
    # Environment 1 should be reset (base height roughly 0.79 + 1.0 drop = 1.79)
    assert not torch.allclose(env.qpos[:, 1], qpos_stepped[:, 1])
    assert torch.allclose(env.qpos[2, 1], torch.tensor(1.79, device=device), atol=0.05)

def test_g1sim_get_obs(device):
    env = G1Sim(nenv=5, device=device)
    env.reset_all(noise=0.0, drop=0.0)
    
    # Check output shape
    obs = env.get_obs()
    assert obs.shape == (5, 99)
    
    # On reset, gravity is straight down, robot should be perfectly upright
    # The projected gravity should be roughly [0, 0, -1]
    assert torch.allclose(obs[:, 3:6], torch.tensor([0.0, 0.0, -1.0], device=device).expand(5, 3), atol=1e-5)
    
    # On reset, the joint position error (indices 12:41) should be 0 
    assert torch.allclose(obs[:, 12:41], torch.zeros(5, 29, device=device), atol=1e-5)

def test_g1sim_foot_data(device):
    env = G1Sim(nenv=2, device=device)
    
    # Check shape
    assert env.foot_pos.shape == (6, 2)
    assert env.foot_vel.shape == (6, 2)
    assert env.contact_forces.shape == (8, 2)
    
    env.reset_all(seed=42, noise=0.0, drop=0.0)
    torch.cuda.synchronize()
    
    # Step physics to populate foot data from CUDA
    env.step(nsub=10)
    torch.cuda.synchronize()
    
    # Foot positions should be populated
    assert not torch.all(env.foot_pos == 0.0)
    
    # Because it is standing, the feet Z coordinates should be close to 0 (the ground)
    # left foot z = 2, right foot z = 5
    assert torch.all(env.foot_pos[2, :] < 0.15)
    assert torch.all(env.foot_pos[5, :] < 0.15)
    assert torch.all(env.foot_pos[2, :] > -0.05)
    assert torch.all(env.foot_pos[5, :] > -0.05)
    
    # Contact forces should not be all zero since the robot is standing on the ground
    assert torch.any(env.contact_forces > 0.0)

def test_g1sim_foot_kinematics_vs_mujoco(device):
    import numpy as np
    import mujoco
    import os
    
    env = G1Sim(nenv=1, device=device)
    # Initialize with some random noise so joints and velocities are non-zero
    env.reset_all(seed=42, noise=0.1, drop=0.0)
    torch.cuda.synchronize()
    
    # Record state BEFORE step
    qpos_before = env.qpos[:, 0].cpu().numpy().copy()
    qvel_before = env.qvel[:, 0].cpu().numpy().copy()
    
    # Step exactly 1 substep so the foot_pos exported by CUDA is evaluated exactly on qpos_before
    env.step(nsub=1)
    torch.cuda.synchronize()
    
    HERE = os.path.dirname(os.path.abspath(__file__))
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "..", "model", "g1_stripped.xml"))
    d = mujoco.MjData(m)
    
    # Load the BEFORE state into mujoco
    d.qpos[:] = qpos_before
    d.qvel[:] = qvel_before
    mujoco.mj_forward(m, d)
    
    # In MuJoCo, world is body 0. Left foot is body 7, Right foot is body 13.
    left_foot_pos_mj = d.xpos[7]
    right_foot_pos_mj = d.xpos[13]
    
    env_foot_pos = env.foot_pos[:, 0].cpu().numpy()
    
    assert np.allclose(env_foot_pos[0:3], left_foot_pos_mj, atol=1e-5)
    assert np.allclose(env_foot_pos[3:6], right_foot_pos_mj, atol=1e-5)
    
    res = np.zeros(6)
    
    # flg_local=0 gets world frame twist [ang, lin]
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 7, res, 0)
    left_foot_vel_mj = res[3:6]
    
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 13, res, 0)
    right_foot_vel_mj = res[3:6]
    
    env_foot_vel = env.foot_vel[:, 0].cpu().numpy()
    
    assert np.allclose(env_foot_vel[0:3], left_foot_vel_mj, atol=1e-5)
    assert np.allclose(env_foot_vel[3:6], right_foot_vel_mj, atol=1e-5)

