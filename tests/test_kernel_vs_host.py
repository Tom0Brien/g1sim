#!/usr/bin/env python3
"""Cross-validate the CUDA kernel code path (built via the CPU shim, float)
against the double-precision host library that was validated vs MuJoCo."""
import ctypes as C
import numpy as np
import subprocess
import os
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
lib = C.CDLL(os.path.join(ROOT, "build", "libg1host.so"))
P = lambda a: a.ctypes.data_as(C.POINTER(C.c_double))

def test_kernel_cross_validation():
    m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "..", "model", "g1_stripped.xml"))
    NV, NC = m.nv, 8

    exe_name = "g1bench_cpu"
    out = subprocess.run([os.path.join(ROOT, "build", exe_name),
                          "--dump", "--steps", "500"],
                         capture_output=True, text=True, check=True).stdout
    traj_f = np.array([[float(x) for x in l.split()] for l in out.strip().splitlines()])

    qp = m.key_qpos[0].copy(); qp[2] += 0.02
    qv = np.zeros(NV); ctrl = m.key_qpos[0][7:].copy()
    fn = np.zeros(NC); anchor = np.full(2 * NC, 1e30)
    traj_d = []
    
    for _ in range(500):
        lib.g1_c_step(P(qp), P(qv), P(ctrl), 1, C.c_double(2e-3), P(fn), P(anchor))
        traj_d.append(qp.copy())
        
    err = np.abs(traj_f - np.array(traj_d))
    assert err[-1].max() < 5e-3
