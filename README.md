# g1sim — specialized GPU simulator for Unitree G1 RL locomotion

A from-scratch, single-purpose rigid-body simulator for the Unitree G1
(29-DoF, `mujoco_menagerie` model), built for massively parallel RL walking
training. One CUDA thread steps one environment; all model constants are
baked into the binary at compile time; contacts are exactly the 8 official
foot points against a ground plane. No broadphase, no convex solver, no
generality — that is the point.

## Layout

```
tools/gen_model.py      MJCF -> g1_model.h generator (+ stripped oracle XML)
tools/g1_raw.xml        official menagerie g1.xml (29 DoF)
tools/fake_cuda.h       CPU shim: compiles/runs the kernel TU without a GPU
src/g1_types.h          scalar precision switch, tunable config
src/g1_model.h          GENERATED: tree, inertias, joints, gains, contacts
src/spatial.h           spatial algebra ([angular; linear], body-frame)
src/g1_core.h           FK, contacts, ABA, PD, integrator (host/device)
src/g1_kernel.cu        thread-per-env step/reset kernels + benchmark
src/host_lib.cpp        C ABI for ctypes validation (double precision)
tests/g1_stripped.xml   GENERATED: the MuJoCo oracle model
tests/test_vs_mujoco.py oracle validation suite
tests/test_kernel_vs_host.py  float kernel vs double reference
```

## Build & test

```
make            # host lib (double) + kernel via CPU shim (float)
make test       # full validation (needs: pip install mujoco)
make bench-cpu  # kernel benchmark through the CPU shim
make gpu        # real CUDA build: nvcc --expt-relaxed-constexpr, ARCH=sm_XX
./build/g1bench --nenv 8192 --steps 1000        # GPU benchmark
make model      # regenerate g1_model.h from tools/g1_raw.xml
```

## Validation status (measured, reproducible via `make test`)

Strategy: the physics core is scalar-type-agnostic C++ compiling identically
as CUDA device code and host code. The host build (double precision) is
checked numerically against MuJoCo 3.9 as an oracle on the stripped model;
the kernel translation unit is then cross-checked against that reference.

| check | result |
|---|---|
| MuJoCo free-joint conventions (lin=world, ang=body) | pinned empirically, asserted |
| forward kinematics, 50 random configs | max err 1.8e-15 |
| smooth forward dynamics (random qpos/qvel/τ incl. base wrench), 200 states | max rel err 6.2e-12 |
| 400-step torque-driven trajectory vs `mj_step` | max divergence 1.2e-13 |
| quaternion integration vs `mju_quatIntegrate` | < 1e-12 |
| drop test: settles at 0.7892 m (nominal 0.79), upright, Σfₙ = weight (327.1 N) | pass |
| float kernel path vs double reference, 500 steps incl. impact | max 2.6e-7 |
| 4096 randomized envs, 0.5 s drop+settle, kernel path | all standing, no NaN |

**Not validated here: actual GPU execution.** The development container has
no GPU or CUDA toolkit. The exact kernel code path runs and passes through
the CPU shim, and the CUDA-specific surface is deliberately tiny (launch
syntax, `cudaMalloc/Memcpy`, constexpr model arrays needing
`--expt-relaxed-constexpr`) — but compile and run `make gpu` on real
hardware before trusting it. Reference throughput: 1.08e5 env-steps/s in
float on a single CPU core (≈9 µs/env-step).

## Physics & deliberate divergences from MuJoCo

- **Dynamics**: Featherstone ABA (O(n)), floating base solved exactly via a
  6×6 Cholesky; gravity applied as explicit per-body external forces;
  armature on every hinge; semi-implicit Euler matching MuJoCo's Euler with
  `eulerdamp` disabled. This part *matches MuJoCo to ~1e-12*.
- **Actuation**: PD position servos reproducing the model's `<position>`
  actuators (kp=500, compiled dampratio kv per joint), torque-clamped to
  `actuatorfrcrange`. The velocity-feedback terms (kv, joint damping, active
  limit damping) are integrated **implicitly** by folding `dt·b` into the
  articulated `D` — exact per-joint backward-Euler damping. This is required:
  the menagerie gains are unstable under explicit damping at dt=2e-3 (which
  is why the official model ships with `integrator="implicitfast"`). Caveat:
  the implicit damping increment is not subject to the torque clamp.
- **Contacts**: compliant spring-damper normals + anchor-based stick-slip
  Coulomb friction (true stiction, no rest creep) at the 8 official foot
  spheres vs the plane z=0. Intentionally *not* MuJoCo's constraint solver;
  gains in `g1_types.h` are sized for the foot's articulated effective mass
  (~2 mm static penetration, stable at dt=2e-3).
- **Joint limits**: soft one-sided spring-dampers (no hard constraints).
- **Not modeled**: joint `frictionloss` (zeroed in the oracle too), contacts
  other than feet-vs-ground (self-collision, terrain), tendon/equality
  constraints (the G1 has none).

## RL integration notes

Per-env persistent state is `qpos[36] + qvel[35] + anchor[16]` floats, SoA
with env as fastest axis. A training loop owns the `ctrl` buffer (PD targets,
typically `default_pose + action_scale * action`), calls `g1_step_kernel`
with `nsub` = control decimation, and computes observations/rewards in its
own kernel or framework (heights, base velocities, and contact forces are
cheap to re-derive or export — `g1_contacts` already returns per-point fₙ).
Resets are `g1_reset_kernel` with per-env hash RNG. Wrap the buffers with
DLPack/CuPy/Torch as desired; no copies needed.

## Optimization roadmap (correctness-first v0 leaves this on the table)

1. Pack the 6×6 articulated inertias symmetric (21 floats) and replace the
   generic congruence transform with the closed-form spatial-inertia
   transform — biggest single win, cuts the dominant FLOPs ~2×.
2. Shrink per-thread state below local-memory spill thresholds; stage hot
   arrays in shared memory; `__launch_bounds__` tuning.
3. Fuse PD+contacts+ABA loops; precompute per-hinge constants now reloaded
   per step; consider warp-per-env with the backward pass kept serial.
4. Curand-based domain randomization (mass/friction/gains) at reset.
