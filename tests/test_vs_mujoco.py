#!/usr/bin/env python3
"""Validation of the specialized G1 simulator against MuJoCo (oracle).

  0. Empirically pin MuJoCo's free-joint qvel convention (linear=world,
     angular=body) so the comparison itself is not built on assumptions.
  1. FK: world body poses must match for random qpos.
  2. Smooth forward dynamics: qacc must match mj_forward (no contacts,
     random qpos/qvel/tau incl. base wrench) to near machine precision.
  3. Trajectory: 400 semi-implicit Euler steps under torques must track.
  4. Contacts: drop from keyframe, robot must settle upright with total
     normal force = weight, no NaNs, bounded penetration.
"""
import ctypes as C
import numpy as np
import mujoco, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
if sys.platform == "win32":
    lib = C.CDLL(os.path.join(HERE, "..", "build", "libg1host.dll"))
else:
    lib = C.CDLL(os.path.join(HERE, "..", "build", "libg1host.so"))
D = np.float64
P = lambda a: a.ctypes.data_as(C.POINTER(C.c_double))

nb = C.c_int(); nq = C.c_int(); nv = C.c_int(); nu = C.c_int(); nc = C.c_int()
prec = lib.g1_sizes(C.byref(nb), C.byref(nq), C.byref(nv), C.byref(nu), C.byref(nc))
NB, NQ, NV, NU, NC = nb.value, nq.value, nv.value, nu.value, nc.value
assert prec == 8, "build host lib with G1_PRECISION=8 for validation"

m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "..", "model", "g1_stripped.xml"))
d = mujoco.MjData(m)
assert (m.nbody - 1, m.nq, m.nv, m.nu) == (NB, NQ, NV, NU)
rng = np.random.default_rng(0)

def random_state():
    qpos = np.zeros(NQ)
    qpos[:3] = rng.uniform(-1, 1, 3)
    q = rng.normal(size=4); qpos[3:7] = q / np.linalg.norm(q)
    lo, hi = m.jnt_range[1:, 0], m.jnt_range[1:, 1]
    qpos[7:] = rng.uniform(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo))
    qvel = rng.uniform(-2, 2, NV)
    return qpos, qvel

# ---- 0. pin the free-joint convention empirically --------------------------
d.qpos[:] = 0; d.qpos[2] = 1
# rotate base 90 deg about z; set qvel angular = x; check world angular velocity
d.qpos[3:7] = [np.cos(np.pi/4), 0, 0, np.sin(np.pi/4)]
d.qvel[:] = 0; d.qvel[3:6] = [1, 0, 0]
mujoco.mj_forward(m, d)
res = np.zeros(6)
mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 1, res, 0)  # world
ang_world = res[:3]
assert np.allclose(ang_world, [0, 1, 0], atol=1e-12), \
    f"free-joint angular qvel is not body-local? got {ang_world}"
d.qvel[:] = 0; d.qvel[0:3] = [1, 0, 0]
mujoco.mj_forward(m, d)
mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, 1, res, 0)
assert np.allclose(res[3:], [1, 0, 0], atol=1e-12), \
    f"free-joint linear qvel is not world-frame? got {res[3:]}"
print("[0] MuJoCo free-joint convention confirmed: linear=world, angular=body")

m.opt.disableflags |= (mujoco.mjtDisableBit.mjDSBL_ACTUATION | mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
# ---- 1. forward kinematics --------------------------------------------------
worst = 0
for _ in range(50):
    qpos, _ = random_state()
    d.qpos[:] = qpos; d.qvel[:] = 0
    mujoco.mj_forward(m, d)
    xpos = np.zeros(NB * 3); xmat = np.zeros(NB * 9)
    lib.g1_c_fk(P(qpos), P(xpos), P(xmat))
    e1 = np.abs(xpos.reshape(NB, 3) - d.xpos[1:]).max()
    e2 = np.abs(xmat.reshape(NB, 9) - d.xmat[1:]).max()
    worst = max(worst, e1, e2)
print(f"[1] FK max error over 50 random configs: {worst:.3e}")
assert worst < 1e-12

# ---- 2. smooth forward dynamics --------------------------------------------
worst = 0
for _ in range(200):
    qpos, qvel = random_state()
    tau = rng.uniform(-30, 30, NV)            # includes base wrench dofs
    d.qpos[:] = qpos; d.qvel[:] = qvel
    d.qfrc_applied[:] = tau; d.ctrl[:] = 0
    mujoco.mj_forward(m, d)
    qacc = np.zeros(NV)
    lib.g1_c_fd(P(qpos), P(qvel), P(tau), P(qacc))
    scale = np.maximum(np.abs(d.qacc), 1.0)
    worst = max(worst, (np.abs(qacc - d.qacc) / scale).max())
print(f"[2] forward dynamics max rel error over 200 random states: {worst:.3e}")
assert worst < 1e-9

# ---- 3. trajectory rollout (smooth, torque-driven) --------------------------
qpos, qvel = random_state()
qpos[7:] *= 0.3; qvel *= 0.2
d.qpos[:] = qpos.copy(); d.qvel[:] = qvel.copy(); d.ctrl[:] = 0
tau = rng.uniform(-5, 5, NV); tau[:6] = 0
d.qfrc_applied[:] = tau
qp, qv = qpos.copy(), qvel.copy()
m.opt.timestep = 1e-3
nsteps = 400
for _ in range(nsteps):
    mujoco.mj_step(m, d)
# my rollout: integrate with same dt using fd + same scheme as MuJoCo Euler
for _ in range(nsteps):
    qacc = np.zeros(NV)
    lib.g1_c_fd(P(qp), P(qv), P(tau), P(qacc))
    qv += 1e-3 * qacc
    qp[:3] += 1e-3 * qv[:3]
    quat = qp[3:7].copy()
    mujoco.mju_quatIntegrate(quat, qv[3:6], 1e-3)
    qp[3:7] = quat / np.linalg.norm(quat)
    qp[7:] += 1e-3 * qv[6:]
err = np.abs(qp - d.qpos).max()
print(f"[3] trajectory divergence after {nsteps} steps (0.4 s): {err:.3e}")
assert err < 1e-6

# my own quaternion integrator must agree with mju_quatIntegrate.
# g1_c_step is semi-implicit: quat is integrated with the UPDATED omega,
# so replicate that exactly: w_new = w + dt*qacc, then mju_quatIntegrate.
quat = np.array([0.3, -0.5, 0.7, 0.4]); quat /= np.linalg.norm(quat)
qp2 = np.zeros(NQ); qv2 = np.zeros(NV)
qp2[3:7] = quat; qp2[2] = 5.0                  # airborne: no contacts
qv2[3:6] = [1.3, -2.1, 0.7]
ctrl = np.zeros(NU)                            # ctrl = qpos joints => PD inert
anch = np.full(2*NC, 1e30)
lib.g1_c_step(P(qp2), P(qv2), P(ctrl), 1, C.c_double(0.01), P(np.zeros(NC)), P(anch))
# semi-implicit: quat must equal mju_quatIntegrate with the POST-step omega
ref = quat.copy(); mujoco.mju_quatIntegrate(ref, qv2[3:6].copy(), 0.01)
assert np.abs(qp2[3:7] - ref / np.linalg.norm(ref)).max() < 1e-12, "quat integration mismatch"
print("[3b] quaternion integration matches mju_quatIntegrate")

# ---- 4. contact settling ----------------------------------------------------
qp = m.key_qpos[0].copy().astype(D)
qp[2] += 0.02                                   # small drop
qv = np.zeros(NV)
ctrl = m.key_ctrl[0].copy().astype(D)
fn = np.zeros(NC)
anchor = np.full(2*NC, 1e30)
dt = 2e-3
for i in range(int(10.0 / dt)):
    lib.g1_c_step(P(qp), P(qv), P(ctrl), 1, C.c_double(dt), P(fn), P(anchor))
    assert np.isfinite(qp).all() and np.isfinite(qv).all(), f"NaN at step {i}"
total_mass = m.body_mass[1:].sum()
weight = total_mass * 9.81
print(f"[4] settled: height={qp[2]:.4f} m  speed={np.linalg.norm(qv):.2e}"
      f"  sum(fn)={fn.sum():.1f} N  weight={weight:.1f} N")
assert abs(qp[2] - m.key_qpos[0][2]) < 0.03, "did not settle near nominal height"
assert np.linalg.norm(qv) < 1e-2, "did not come to rest"
assert abs(fn.sum() - weight) / weight < 0.02, "contact forces != weight"
up = np.zeros(9); mujoco.mju_quat2Mat(up, qp[3:7])
assert up.reshape(3,3)[2,2] > 0.99, "robot fell over"
print("ALL TESTS PASSED")
