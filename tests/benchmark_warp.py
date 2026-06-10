import os
import sys
import time
import subprocess

def run_g1sim(nenv: int, steps: int):
    print(f"=========================================")
    print(f"Running g1sim GPU with {nenv} envs for {steps} steps...")
    exe = os.path.join(os.path.dirname(__file__), "..", "build", "g1bench")
    t0 = time.time()
    out = subprocess.run([exe, "--nenv", str(nenv), "--steps", str(steps)], capture_output=True, text=True)
    t1 = time.time()
    if out.returncode != 0:
        print(f"g1sim failed: {out.stderr}")
        return
    print(out.stdout)
    return out.stdout

def run_mujoco_warp(nenv: int, steps: int):
    print(f"=========================================")
    print(f"Running mujoco_warp with {nenv} envs for {steps} steps...")
    import warp as wp
    import mujoco
    import mujoco_warp as mw
    
    wp.init()
    
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "g1_stripped.xml")
    m = mujoco.MjModel.from_xml_path(model_path)
    
    # Pre-allocate the warp structures
    mw_m = mw.put_model(m)
    mw_d = mw.make_data(m, nworld=nenv)
    
    # Warmup
    mw.step(mw_m, mw_d)
    wp.synchronize()
    
    # Benchmark
    t0 = time.time()
    for _ in range(steps):
        mw.step(mw_m, mw_d)
    wp.synchronize()
    t1 = time.time()
    
    elapsed = t1 - t0
    env_steps = nenv * steps
    throughput = env_steps / elapsed
    print(f"mujoco_warp [CUDA] {nenv} envs x {steps} steps")
    print(f"  {elapsed:.3f} s  ->  {throughput:.2e} env-steps/s")

if __name__ == "__main__":
    nenv = 16384
    steps = 250
    run_g1sim(nenv, steps)
    run_mujoco_warp(nenv, steps)
