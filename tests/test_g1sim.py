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
