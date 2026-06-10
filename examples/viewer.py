#!/usr/bin/env python3
"""Visual rollout of the g1sim custom physics rendered in MuJoCo's viewer.

Runs the g1sim host library (double-precision, with contacts + PD control)
for a drop-and-stand scenario, records the qpos trajectory, then plays it
back through MuJoCo's passive viewer at real-time speed.

Usage:
    python tests/test_viewer.py                  # default 10 s drop + stand
    python tests/test_viewer.py --duration 20    # 20 seconds of simulation
    python tests/test_viewer.py --perturb        # apply random joint perturbations
"""
import ctypes as C
import numpy as np
import mujoco
import mujoco.viewer
import os, sys, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")

# ── load host library ───────────────────────────────────────────────────
lib = C.CDLL(os.path.join(ROOT, "build", "libg1host.so"))

D = np.float64
P = lambda a: a.ctypes.data_as(C.POINTER(C.c_double))

nb = C.c_int(); nq = C.c_int(); nv = C.c_int(); nu = C.c_int(); nc = C.c_int()
lib.g1_sizes(C.byref(nb), C.byref(nq), C.byref(nv), C.byref(nu), C.byref(nc))
NB, NQ, NV, NU, NC = nb.value, nq.value, nv.value, nu.value, nc.value

# ── load MuJoCo model (for rendering only) ──────────────────────────────
# Priority: scene.xml (full lighting + floor) > g1_raw.xml > stripped model
scene_xml = os.path.join(ROOT, "model", "scene.xml")
raw_xml = os.path.join(ROOT, "model", "g1_raw.xml")
stripped_xml = os.path.join(ROOT, "model", "g1_stripped.xml")
assets_dir = os.path.join(ROOT, "model", "assets")
has_meshes = (os.path.isdir(assets_dir) and
              len([f for f in os.listdir(assets_dir) if f.endswith(".STL")]) >= 20)
use_visual_model = False

if has_meshes and os.path.exists(scene_xml):
    print("Using scene model (floor + lighting + mesh assets)")
    m = mujoco.MjModel.from_xml_path(scene_xml)
    use_visual_model = True
elif has_meshes and os.path.exists(raw_xml):
    print("Using full visual model (g1_raw.xml + mesh assets)")
    m = mujoco.MjModel.from_xml_path(raw_xml)
    use_visual_model = True
else:
    if not has_meshes:
        print("Mesh assets not found — run: python model/download_assets.py")
    print("Falling back to stripped model (inertia-box visualization)")
    m = mujoco.MjModel.from_xml_path(stripped_xml)
d = mujoco.MjData(m)

# ── parse args ──────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="g1sim visual rollout")
parser.add_argument("--duration", type=float, default=10.0,
                    help="simulation duration in seconds (default: 10)")
parser.add_argument("--perturb", action="store_true",
                    help="apply random joint perturbations at 2-second intervals")
parser.add_argument("--drop-height", type=float, default=0.5,
                    help="extra height above nominal for initial drop (m)")
parser.add_argument("--substeps", type=int, default=4,
                    help="physics substeps per control step (default: 4)")
args = parser.parse_args()

# ── simulate with g1sim ─────────────────────────────────────────────────
control_dt = 2e-3                          # outer control rate
nsub = args.substeps                       # physics substeps per control step
sub_dt = control_dt / nsub                 # physics timestep
n_steps = int(args.duration / control_dt)

qp = m.key_qpos[0].copy().astype(D)
qp[2] += args.drop_height
qv = np.zeros(NV, dtype=D)
ctrl = m.key_qpos[0][7:].copy().astype(D)   # PD target = keyframe pose
fn = np.zeros(NC, dtype=D)
anchor = np.full(2 * NC, 1e30, dtype=D)

rng = np.random.default_rng(42)

print(f"Simulating {args.duration:.1f} s ({n_steps} steps, "
      f"{nsub} substeps @ sub_dt={sub_dt*1e3:.2f} ms) ...")
t0 = time.perf_counter()

trajectory = []
for i in range(n_steps):
    # optional perturbations: bump random joints every 2 s after settling
    if args.perturb and i > 0 and i % int(2.0 / control_dt) == 0:
        # add random velocity impulses to a few joints
        joints_to_kick = rng.choice(NV - 6, size=4, replace=False) + 6
        qv[joints_to_kick] += rng.uniform(-1.0, 1.0, size=4)
        print(f"  [t={i*control_dt:.1f}s] perturbed joints {joints_to_kick - 6}")

    lib.g1_c_step(P(qp), P(qv), P(ctrl), nsub, C.c_double(sub_dt), P(fn), P(anchor))

    if not np.isfinite(qp).all():
        print(f"NaN detected at step {i} (t={i*control_dt:.3f} s), aborting.")
        break

    trajectory.append(qp.copy())

elapsed = time.perf_counter() - t0
print(f"Done: {len(trajectory)} steps in {elapsed:.2f} s "
      f"({len(trajectory)/elapsed:.0f} steps/s)")
print(f"Final height: {qp[2]:.4f} m, speed: {np.linalg.norm(qv):.2e}")

trajectory = np.array(trajectory)

# ── playback in MuJoCo viewer ───────────────────────────────────────────
print(f"\nLaunching viewer — playing back {len(trajectory)} frames at real-time...")
print("Close the viewer window to exit.")

frame_idx = [0]
wall_t0 = [None]
paused = [False]

def key_callback(keycode):
    """Space to pause/resume, R to restart."""
    if keycode == ord(' '):
        paused[0] = not paused[0]
        print("PAUSED" if paused[0] else "RESUMED")
    elif keycode == ord('R') or keycode == ord('r'):
        frame_idx[0] = 0
        wall_t0[0] = None
        paused[0] = False
        print("RESTARTED")

with mujoco.viewer.launch_passive(m, d, key_callback=key_callback) as viewer:
    if not use_visual_model:
        # enable inertia-box visualization so the stripped model is visible
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = True
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False

    while viewer.is_running():
        if wall_t0[0] is None:
            wall_t0[0] = time.perf_counter()

        if not paused[0]:
            # real-time sync: advance frame index to match wall clock
            wall_elapsed = time.perf_counter() - wall_t0[0]
            target_frame = int(wall_elapsed / control_dt)
            frame_idx[0] = min(target_frame, len(trajectory) - 1)

            # loop back
            if frame_idx[0] >= len(trajectory) - 1:
                frame_idx[0] = 0
                wall_t0[0] = time.perf_counter()

        # set qpos from trajectory and run FK for rendering
        d.qpos[:] = trajectory[frame_idx[0]]
        mujoco.mj_forward(m, d)
        viewer.sync()

        # don't spin too fast — sleep to hit roughly 60 fps
        time.sleep(max(0, 1/60 - 0.001))
