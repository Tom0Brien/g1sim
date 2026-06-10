# g1sim — Specialized GPU Simulator for Unitree G1

<p align="center">
  <img src="https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1/g1.png" alt="Unitree G1 Humanoid" width="400"/>
</p>

A from-scratch, single-purpose rigid-body simulator for the Unitree G1 humanoid (29-DoF), built specifically for massively parallel RL locomotion training. 

One CUDA thread steps one environment. All model constants are baked into the binary at compile time. Contacts are strictly the 8 official foot points against a ground plane. No broadphase, no convex solver, no generality — **that is the point**.

## Repository Structure

```text
model/                   MJCF models and generator scripts
  gen_model.py           Generates the C header and stripped oracle XML
  g1_raw.xml             Official menagerie g1.xml
src/                     Core physics engine
  g1_core.h              FK, contacts, ABA, PD, integrator (host/device)
  g1_kernel.cu           Thread-per-env step/reset kernels + benchmark
  spatial.h              Spatial algebra ([angular; linear], body-frame)
  g1_model.h             GENERATED: tree, inertias, joints, gains, contacts
tests/                   Validation suite vs MuJoCo
```

## Build & Run

Ensure you have a C++ compiler (`g++` or `clang++`), the CUDA Toolkit (`nvcc`), and [uv](https://docs.astral.sh/uv/) installed.

Initialize the Python environment and dependencies:
```bash
uv sync --extra dev
```

Build the simulator and run tests:
```bash
make              # host lib (double) + kernel via CPU shim (float)
uv run make test  # full validation against MuJoCo oracle
make bench-cpu    # CPU kernel benchmark
make gpu          # real CUDA build: nvcc --expt-relaxed-constexpr
```

To benchmark the CUDA implementation against `mujoco_warp`:
```bash
uv run python tests/benchmark_warp.py
```

## Performance & Benchmarks

This simulator has been heavily optimized for massively parallel GPU execution:
- **Packed Symmetric Inertias:** Uses a 21-float packed symmetric matrix and closed-form spatial inertia congruence transforms, slashing dominant FLOPs by ~2x.
- **Minimized Thread State:** Per-thread local memory (`G1Ws`) has been shrunk by ~27%, drastically reducing register spilling on GPUs and improving warp occupancy.
- **Loop Fusion:** PD control, contact modeling, and the Featherstone Articulated Body Algorithm (ABA) are fused into a single tight loop to maximize register locality and minimize memory trips.

### Hardware & Throughput

Benchmarked on a laptop with an **NVIDIA GeForce RTX 3080 Laptop GPU (16 GiB)** and an **x86_64** CPU, simulating a batch of 16,384 environments for 250 steps (dt=2e-3):

- **g1sim CPU shim (single core):** ~1.27 million env-steps/s
- **g1sim CUDA GPU:** ~12.8 million env-steps/s
- **mujoco_warp GPU (reference):** ~1.08 million env-steps/s

By stripping away the generalized constraint solver and broadphase layers, the specialized CUDA kernel runs nearly **12x faster** than the generalized `mujoco_warp` equivalent on the same model and hardware.

## Validation Status

The physics core is scalar-type-agnostic C++ compiling identically as CUDA device code and host code. The host build (double precision) is checked numerically against MuJoCo 3.9 as an oracle.

| Check | Result |
|---|---|
| Forward kinematics (50 random configs) | max err `1.8e-15` |
| Smooth forward dynamics vs ABA | max rel err `6.2e-12` |
| 400-step torque-driven trajectory vs `mj_step` | max divergence `1.2e-13` |
| Quaternion integration vs `mju_quatIntegrate` | `< 1e-12` |
| Float kernel path vs double reference (500 steps) | max err `2.6e-7` |

## Physics & Divergences from MuJoCo

- **Dynamics**: Featherstone ABA ($O(n)$) with a floating base solved exactly via a 6×6 Cholesky. Gravity is applied as explicit per-body external forces. Matches MuJoCo to `~1e-12`.
- **Actuation**: PD position servos with implicit velocity-feedback terms (exact per-joint backward-Euler damping). This is unconditionally stable at dt=2e-3, matching MuJoCo's `integrator="implicitfast"`.
- **Contacts**: Compliant spring-damper normals + anchor-based stick-slip Coulomb friction at the 8 official foot spheres vs the ground plane. This gives true stiction without rest creep, skipping MuJoCo's constraint solver entirely.
- **Not Modeled**: Joint `frictionloss`, contacts other than feet-vs-ground, and tendon constraints (the G1 has none).

## RL Integration Notes

Per-env persistent state is `qpos[36] + qvel[35] + anchor[16]` floats, stored in Struct-of-Arrays (SoA) format with the environment as the fastest axis. 

A training loop owns the `ctrl` buffer (PD targets) and calls `g1_step_kernel` with `nsub` = control decimation. It computes observations/rewards in its own framework (heights, base velocities, and contact forces are cheap to export). Resets are handled by `g1_reset_kernel` with per-env hash RNG. Wrap the buffers with DLPack/CuPy/Torch directly — no copies needed.
