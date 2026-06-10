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

Ensure you have a C++ compiler (`g++`, `clang++`, or `cl.exe`) and Python installed.

```bash
make            # host lib (double) + kernel via CPU shim (float)
make test       # full validation (needs: pip install mujoco)
make bench-cpu  # kernel benchmark through the CPU shim
make gpu        # real CUDA build: nvcc --expt-relaxed-constexpr, ARCH=sm_80
make model      # regenerate g1_model.h from model/g1_raw.xml
```

## Performance & Optimizations

This simulator has been heavily optimized for massively parallel GPU execution:
- **Packed Symmetric Inertias:** Uses a 21-float packed symmetric matrix and closed-form spatial inertia congruence transforms, slashing dominant FLOPs by ~2x.
- **Minimized Thread State:** Per-thread local memory (`G1Ws`) has been shrunk by ~27%, drastically reducing register spilling on GPUs and improving warp occupancy.
- **Loop Fusion:** PD control, contact modeling, and the Featherstone Articulated Body Algorithm (ABA) are fused into a single tight loop to maximize register locality and minimize memory trips.

*Reference throughput:* **~925,000 env-steps/s** on a single CPU core (via the CPU shim). Actual CUDA hardware will be orders of magnitude faster.

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
