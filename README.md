# 🚀 g1sim: Ultra-Fast GPU Simulator for Unitree G1

<p align="center">
  <img src="https://raw.githubusercontent.com/google-deepmind/mujoco_menagerie/main/unitree_g1/g1.png" alt="Unitree G1 Humanoid" width="250"/>
</p>

<p align="center">
  <b>12.8 Million env-steps/second on a single laptop GPU.</b>
</p>

A from-scratch, bare-metal rigid-body physics engine built specifically for massively parallel RL locomotion training on the Unitree G1 (29-DoF). 

By stripping away broadphase collision detection, complex constraint solvers, and generalized data structures, `g1sim` maps exactly **one environment to one CUDA thread**. The result is a hyper-specialized kernel that runs **~12x faster** than generalized GPU simulators like `mujoco_warp`.

---

## ⚡ Quickstart

Ensure you have a C++ compiler (`g++` or `clang++`), the CUDA Toolkit (`nvcc`), and [uv](https://docs.astral.sh/uv/) installed.

```bash
# 1. Install Python dependencies
uv sync --extra dev

# 2. Build the engine & run MuJoCo numerical validation tests
uv run make test

# 3. Compile the CUDA kernel & run the benchmark vs mujoco_warp
make gpu
uv run python examples/benchmark_warp.py

# 4. Watch a drop-test rollout in the visualizer
uv run python examples/viewer.py
```

## 🏎️ Performance vs. MuJoCo Warp

Benchmarked simulating **16,384 environments** for 250 steps (dt=2ms):

| Simulator | Throughput | Hardware |
|---|---|---|
| **`g1sim` (Custom CUDA)** | **12.8M steps/s** 🚀 | RTX 3080 Laptop GPU |
| `mujoco_warp` (Reference) | 1.08M steps/s | RTX 3080 Laptop GPU |
| `g1sim` CPU Shim | 1.27M steps/s | x86_64 CPU Core |

**Under the Hood Optimizations:**
- **Packed Symmetric Inertias:** 21-float packed matrices slash spatial algebra FLOPs by ~2x.
- **Minimized Thread State:** Memory footprint shrunk by 27%, maximizing warp occupancy and killing register spills.
- **Loop Fusion:** PD control, contact modeling, and Featherstone ABA are fused into a single tight loop.

## 🔬 Physics & Validation

The physics engine is scalar-type-agnostic C++ that compiles identically as device or host code. The double-precision host build is continuously validated against MuJoCo 3.9 as an oracle:

| Validation Metric | MuJoCo Agreement Error |
|---|---|
| **Forward Kinematics** | `< 1.8e-15` |
| **Forward Dynamics (ABA)** | `< 6.2e-12` |
| **400-step Torque Rollout** | `< 1.2e-13` |
| **Float vs. Double Kernel**| `< 2.6e-7` |

### Design Philosophy
- **Dynamics:** Exact $O(n)$ Featherstone ABA with a floating base solved via 6x6 Cholesky.
- **Actuation:** Implicitly-damped PD servos unconditionally stable at `dt=2e-3` (matching MuJoCo's `implicitfast`).
- **Contacts:** Exactly 8 foot spheres against a ground plane using compliant spring-damper normals + anchor stick-slip Coulomb friction. True stiction, zero rest creep.

## 🧠 RL

The project includes a complete, high-performance reinforcement learning pipeline for training locomotion policies:
- **`examples/train.py`**: A fully functional PPO implementation that trains a robust walking policy directly on the GPU in minutes. It matches benchmark configurations for optimal posture and velocity tracking.
- **`examples/play.py`**: An interactive MuJoCo viewer script to test and evaluate the trained policy (`policy.pt`) in real-time with keyboard velocity commands.
---
*Note: This engine is purpose-built for speed over generality. It intentionally does not model internal frictionloss, upper-body mesh collisions, or generic constraints.*

## 🙏 Acknowledgments
This project builds upon the fantastic work from:
- **[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)**: For providing the high-quality Unitree G1 robot models and MJCF files used as the foundation of this simulation.
- **[mjlab](https://github.com/google-deepmind/mujoco_playground)**: For the excellent baseline locomotion configurations, reward formulations, and training parameters which heavily inspired the RL setup included in this project.
