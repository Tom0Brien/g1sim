#!/usr/bin/env python3
"""Generate src/g1_model.h from the official mujoco_menagerie Unitree G1 MJCF.

Pipeline:
  1. Parse g1_raw.xml; extract foot contact spheres (class="foot").
  2. Strip assets/geoms/sites/sensors (visual & collision irrelevant: explicit
     <inertial> on every body). Zero frictionloss (not modeled in v0 core).
     Disable eulerdamp so the MuJoCo oracle integrates damping explicitly,
     matching the specialized sim. fusestatic merges jointless bodies.
  3. Compile with MuJoCo, save model/g1_stripped.xml (the oracle model).
  4. Emit constexpr arrays into src/g1_model.h.

Conventions baked into the header (must match g1_core.h):
  * Bodies indexed 0..NB-1 in MuJoCo order (0 = pelvis, free joint).
  * Every body has exactly one joint. Body 0: free. Others: hinge.
  * Spatial vectors are [angular; linear], body-frame (Featherstone).
  * Hinge motion subspace S = [axis; anchor x axis] (Pluecker line).
"""
import xml.etree.ElementTree as ET
import numpy as np
import mujoco
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(HERE, "g1_raw.xml")
STRIPPED = os.path.join(HERE, "g1_stripped.xml")
HEADER = os.path.join(ROOT, "src", "g1_model.h")

# ---------------------------------------------------------------- strip xml
tree = ET.parse(RAW)
root = tree.getroot()

# Extract foot contact spheres before stripping. friction & size come from
# the "foot" default class.
foot_default = None
for d in root.iter("default"):
    if d.get("class") == "foot":
        foot_default = d.find("geom")
assert foot_default is not None, "foot default class not found"
foot_radius = float(foot_default.get("size"))
foot_mu = float(foot_default.get("friction").split()[0])

contacts = []  # (body_name, pos[3])
for body in root.iter("body"):
    for geom in body.findall("geom"):
        if geom.get("class") == "foot":
            pos = np.fromstring(geom.get("pos"), sep=" ")
            contacts.append((body.get("name"), pos))
assert len(contacts) == 8, f"expected 8 foot spheres, got {len(contacts)}"

# Strip: assets, geoms, sites, cameras, lights, sensors. Keep actuators
# (to read compiled PD gains) and keyframe (ctrl dim stays valid).
for parent in root.iter():
    for tag in ("geom", "site", "camera", "light"):
        for el in list(parent.findall(tag)):
            parent.remove(el)
for tag in ("asset", "sensor", "visual"):
    for el in list(root.findall(tag)):
        root.remove(el)

# Zero frictionloss everywhere (v0 core does not model dry friction).
for j in root.iter("joint"):
    if j.get("frictionloss") is not None:
        j.set("frictionloss", "0")
for d in root.iter("default"):
    j = d.find("joint")
    if j is not None and j.get("frictionloss") is not None:
        j.set("frictionloss", "0")

comp = root.find("compiler")
comp.set("fusestatic", "true")
opt = root.find("option")
opt.set("integrator", "Euler")
ET.SubElement(opt, "flag", {"eulerdamp": "disable"})

os.makedirs(os.path.dirname(STRIPPED), exist_ok=True)
tree.write(STRIPPED)

# ---------------------------------------------------------------- compile
m = mujoco.MjModel.from_xml_path(STRIPPED)
assert m.nbody >= 2 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE

NB = m.nbody - 1            # physical bodies (world excluded)
NV = m.nv                   # 6 + hinges
NQ = m.nq
NU = m.nu
assert NV == 6 + (NB - 1), "every non-root body must have exactly one hinge"
for b in range(1, m.nbody):
    assert m.body_jntnum[b] == 1, f"body {b} has {m.body_jntnum[b]} joints"
    if b > 1:
        assert m.jnt_type[m.body_jntadr[b]] == mujoco.mjtJoint.mjJNT_HINGE

def quat_to_mat(q):
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, q)
    return r.reshape(3, 3)

names = [m.body(b).name for b in range(1, m.nbody)]
parent = [int(m.body_parentid[b]) - 1 for b in range(1, m.nbody)]  # -1 = world
tpos   = [m.body_pos[b].copy() for b in range(1, m.nbody)]
tquat  = [m.body_quat[b].copy() for b in range(1, m.nbody)]
mass   = [float(m.body_mass[b]) for b in range(1, m.nbody)]
com    = [m.body_ipos[b].copy() for b in range(1, m.nbody)]
# inertia about COM, rotated into the body frame
Icom = []
for b in range(1, m.nbody):
    R = quat_to_mat(m.body_iquat[b])
    Icom.append(R @ np.diag(m.body_inertia[b]) @ R.T)

axis, anchor, armature, damping, rng = [], [], [], [], []
for b in range(2, m.nbody):
    j = m.body_jntadr[b]
    d = m.jnt_dofadr[j]
    assert d == 6 + (b - 2), "dof ordering must follow body ordering"
    axis.append(m.jnt_axis[j].copy())
    anchor.append(m.jnt_pos[j].copy())
    armature.append(float(m.dof_armature[d]))
    damping.append(float(m.dof_damping[d]))
    rng.append(m.jnt_range[j].copy())
# free-joint armature must be zero for the simple base solve
assert np.all(m.dof_armature[:6] == 0) and np.all(m.dof_damping[:6] == 0)

# actuators: position servos, one per hinge, in joint order
kp, kv, frclim = [], [], []
assert NU == NB - 1
for a in range(NU):
    j = m.actuator_trnid[a, 0]
    assert m.jnt_dofadr[j] == 6 + a, "actuator order must match dof order"
    kp.append(float(m.actuator_gainprm[a, 0]))
    assert abs(m.actuator_biasprm[a, 1] + kp[-1]) < 1e-9
    kv.append(float(-m.actuator_biasprm[a, 2]))
    lim = m.jnt_actfrcrange[j]
    assert lim[1] > 0
    frclim.append(float(lim[1]))

body_index = {n: i for i, n in enumerate(names)}
cbody = [body_index[n] for n, _ in contacts]
cpos = [p for _, p in contacts]

key = m.key_qpos[0].copy()

# ---------------------------------------------------------------- emit
def arr(name, data, per_line=6):
    flat = np.asarray(data, dtype=np.float64).reshape(-1)
    lines = []
    for i in range(0, len(flat), per_line):
        lines.append(", ".join(f"R({v:.17g})" for v in flat[i:i+per_line]))
    body = ",\n  ".join(lines)
    return f"G1_MODEL_CONST G1Real {name}[{len(flat)}] = {{\n  {body}\n}};\n"

def iarr(name, data):
    flat = np.asarray(data, dtype=np.int64).reshape(-1)
    body = ", ".join(str(v) for v in flat)
    return f"G1_MODEL_CONST int {name}[{len(flat)}] = {{ {body} }};\n"

with open(HEADER, "w") as f:
    f.write("// AUTO-GENERATED by model/gen_model.py -- do not edit.\n")
    f.write(f"// Source model: {root.get('model')} (mujoco_menagerie)\n")
    f.write("#pragma once\n#include \"g1_types.h\"\n\n")
    f.write("#define R(x) G1Real(x)\n\n")
    f.write(f"#define G1_NB {NB}   // bodies (pelvis=0, free joint)\n")
    f.write(f"#define G1_NV {NV}   // dofs\n")
    f.write(f"#define G1_NQ {NQ}   // qpos size (quat base)\n")
    f.write(f"#define G1_NU {NU}   // actuated hinges\n")
    f.write(f"#define G1_NC {len(contacts)}   // foot contact points\n\n")
    f.write("// body names: " + ", ".join(f"{i}:{n}" for i, n in enumerate(names)) + "\n")
    f.write(iarr("g1_parent", parent))
    f.write(arr("g1_tree_pos", tpos, 3))
    f.write(arr("g1_tree_quat", tquat, 4))
    f.write(arr("g1_mass", mass))
    f.write(arr("g1_com", com, 3))
    f.write(arr("g1_inertia_com", Icom, 3))          # NB x 3 x 3
    f.write(arr("g1_jnt_axis", axis, 3))             # hinges only (body 1..)
    f.write(arr("g1_jnt_anchor", anchor, 3))
    f.write(arr("g1_armature", armature))
    f.write(arr("g1_damping", damping))
    f.write(arr("g1_jnt_range", rng, 2))
    f.write(arr("g1_act_kp", kp))
    f.write(arr("g1_act_kv", kv))
    f.write(arr("g1_act_frclim", frclim))
    f.write(iarr("g1_contact_body", cbody))
    f.write(arr("g1_contact_pos", cpos, 3))
    f.write(f"G1_MODEL_CONST G1Real g1_contact_radius = R({foot_radius:.17g});\n")
    f.write(f"G1_MODEL_CONST G1Real g1_contact_mu = R({foot_mu:.17g});\n")
    f.write(arr("g1_qpos_stand", key, 6))
    f.write("\n#undef R\n")

print(f"NB={NB} NV={NV} NQ={NQ} NU={NU} contacts={len(contacts)}")
print("bodies:", ", ".join(names))
print(f"wrote {HEADER} and {STRIPPED}")
