// host_lib.cpp -- C ABI wrapper for validation from Python (ctypes).
// Build with -DG1_PRECISION=8 for double-precision oracle comparison.
#include "g1_core.h"

extern "C" {

int g1_sizes(int* nb, int* nq, int* nv, int* nu, int* nc) {
  *nb = G1_NB; *nq = G1_NQ; *nv = G1_NV; *nu = G1_NU; *nc = G1_NC;
  return G1_PRECISION;
}

// Forward kinematics: world position (3) + rotation (9, row-major) per body.
void g1_c_fk(const double* qpos, double* xpos, double* xmat) {
  static G1Ws w;
  G1Real qp[G1_NQ], qv[G1_NV] = {0};
  for (int i = 0; i < G1_NQ; ++i) qp[i] = G1Real(qpos[i]);
  g1_fk_vel(qp, qv, w);
  for (int b = 0; b < G1_NB; ++b) {
    xpos[3*b+0] = w.oP[b].x; xpos[3*b+1] = w.oP[b].y; xpos[3*b+2] = w.oP[b].z;
    for (int k = 0; k < 9; ++k) xmat[9*b+k] = w.oR[b].m[k];
  }
}

// Smooth forward dynamics (no contacts, no PD): qacc from (qpos, qvel, tau).
void g1_c_fd(const double* qpos, const double* qvel, const double* tau,
             double* qacc) {
  static G1Ws w;
  G1Config cfg = g1_default_config();
  G1Real qp[G1_NQ], qv[G1_NV], tu[G1_NV], qa[G1_NV];
  for (int i = 0; i < G1_NQ; ++i) qp[i] = G1Real(qpos[i]);
  for (int i = 0; i < G1_NV; ++i) { qv[i] = G1Real(qvel[i]); tu[i] = G1Real(tau[i]); }
  g1_fk_vel(qp, qv, w);
  g1_aba_pure(cfg, qv, tu, w, qa);
  for (int i = 0; i < G1_NV; ++i) qacc[i] = qa[i];
}

// Full step (contacts + PD + limits), n substeps.
void g1_c_step(double* qpos, double* qvel, const double* ctrl, int nsub,
               double dt, double* fn_out, double* anchor) {
  static G1Ws w;
  G1Config cfg = g1_default_config();
  cfg.dt = G1Real(dt);
  G1Real qp[G1_NQ], qv[G1_NV], ct[G1_NU], fn[G1_NC], an[2*G1_NC];
  for (int i = 0; i < G1_NQ; ++i) qp[i] = G1Real(qpos[i]);
  for (int i = 0; i < G1_NV; ++i) qv[i] = G1Real(qvel[i]);
  for (int i = 0; i < G1_NU; ++i) ct[i] = G1Real(ctrl[i]);
  for (int i = 0; i < 2*G1_NC; ++i) an[i] = G1Real(anchor[i]);
  for (int s = 0; s < nsub; ++s) g1_step(cfg, qp, qv, ct, an, w, fn);
  for (int i = 0; i < 2*G1_NC; ++i) anchor[i] = an[i];
  for (int i = 0; i < G1_NQ; ++i) qpos[i] = qp[i];
  for (int i = 0; i < G1_NV; ++i) qvel[i] = qv[i];
  if (fn_out) for (int i = 0; i < G1_NC; ++i) fn_out[i] = fn[i];
}

}  // extern "C"
