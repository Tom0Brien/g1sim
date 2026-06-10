// g1_core.h -- specialized Unitree G1 dynamics (Featherstone ABA, fixed tree,
// 8 fixed foot-plane contacts). Host/device. State conventions match MuJoCo:
//   qpos = [base pos(3, world), base quat(4, wxyz), 29 hinge angles]
//   qvel = [base linvel(3, WORLD frame), base angvel(3, BODY frame), 29 qd]
//   qacc = d/dt qvel in the same frames.
// Internally the base twist is body-frame [omega; v] (Featherstone).
#pragma once
#include "spatial.h"
#include "g1_model.h"

#define G1_NJ (G1_NB - 1)   // hinge joints

struct G1Ws {                // per-environment scratch (thread-local on GPU)
  SE3 liMi[G1_NB];           // pose of body i in parent frame
  M3  oR[G1_NB];             // world orientation of body i
  V3  oP[G1_NB];             // world position of body i
  SV  v[G1_NB];              // body-frame twist
  SV  fext[G1_NB];           // external spatial force, body frame
  M6  IA[G1_NB];             // articulated inertia
  SV  pA[G1_NB];             // articulated bias force
  SV  S[G1_NJ];              // hinge motion subspace
  SV  U[G1_NJ];              // IA * S
  G1Real D[G1_NJ], u[G1_NJ]; // articulated quantities
  SV  a[G1_NB];              // body-frame spatial acceleration
};

G1_FN V3 ld3(const G1Real* p) { return v3(p[0], p[1], p[2]); }

// ---------------------------------------------------------------- pass 1
// FK + velocities. qpos/qvel in MuJoCo convention.
G1_FN void g1_fk_vel(const G1Real* qpos, const G1Real* qvel, G1Ws& w) {
  // base
  Quat bq = qnorm(Quat{qpos[3], qpos[4], qpos[5], qpos[6]});
  w.liMi[0].R = quat_to_mat(bq);
  w.liMi[0].p = ld3(qpos);
  w.oR[0] = w.liMi[0].R;
  w.oP[0] = w.liMi[0].p;
  V3 w_loc = ld3(qvel + 3);
  w.v[0] = sv(w_loc, mulT(w.oR[0], ld3(qvel)));   // [omega_b; R^T v_world]
  // hinges
  for (int i = 1; i < G1_NB; ++i) {
    int h = i - 1, par = g1_parent[i];
    V3 ax = ld3(g1_jnt_axis + 3*h);
    V3 an = ld3(g1_jnt_anchor + 3*h);
    G1Real q = qpos[7 + h], qd = qvel[6 + h];
    M3 RT = quat_to_mat(qnorm(Quat{g1_tree_quat[4*i], g1_tree_quat[4*i+1],
                                   g1_tree_quat[4*i+2], g1_tree_quat[4*i+3]}));
    M3 RJ = axis_angle_mat(ax, q);
    w.liMi[i].R = matmul(RT, RJ);
    w.liMi[i].p = mul(RT, an - mul(RJ, an)) + ld3(g1_tree_pos + 3*i);
    w.oR[i] = matmul(w.oR[par], w.liMi[i].R);
    w.oP[i] = mul(w.oR[par], w.liMi[i].p) + w.oP[par];
    w.S[h] = sv(ax, cross(an, ax));
    w.v[i] = motion_act_inv(w.liMi[i], w.v[par]) + qd * w.S[h];
  }
}

// ------------------------------------------------------------- contacts
// Compliant sphere-vs-plane(z=0) at the 8 fixed foot points. Friction is
// stick-slip: a tangential spring to a per-contact anchor point (2 persistent
// floats per contact in `anchor`), projected onto the Coulomb cone when
// slipping (anchor dragged so the spring matches the cone force). This gives
// true stiction -- no rest creep, unlike purely viscous Coulomb capping.
// anchor[2k] >= G1_ANCHOR_FREE marks "not in contact".
#define G1_ANCHOR_FREE G1Real(1e30)
G1_FN void g1_contacts(const G1Config& cfg, G1Ws& w, G1Real* anchor,
                       G1Real* fn_out) {
  for (int k = 0; k < G1_NC; ++k) {
    int b = g1_contact_body[k];
    V3 r = ld3(g1_contact_pos + 3*k);
    V3 p = mul(w.oR[b], r) + w.oP[b];                     // sphere center
    G1Real depth = g1_contact_radius - p.z;
    G1Real fn = 0;
    if (depth > 0) {
      V3 vp = mul(w.oR[b], w.v[b].l + cross(w.v[b].a, r)); // point vel, world
      fn = cfg.contact_kn * depth - cfg.contact_dn * vp.z;
      if (fn < 0) fn = 0;
      if (anchor[2*k] >= G1_ANCHOR_FREE) {                 // new touchdown
        anchor[2*k] = p.x; anchor[2*k+1] = p.y;
      }
      G1Real ftx = -cfg.contact_kt * (p.x - anchor[2*k])   - cfg.contact_dt * vp.x;
      G1Real fty = -cfg.contact_kt * (p.y - anchor[2*k+1]) - cfg.contact_dt * vp.y;
      G1Real ftn = sqrt(ftx*ftx + fty*fty), fmax = g1_contact_mu * fn;
      if (ftn > fmax) {                                    // slip: project +
        G1Real sc = fmax / (ftn + G1Real(1e-12));          // drag the anchor
        ftx *= sc; fty *= sc;
        anchor[2*k]   = p.x + (ftx + cfg.contact_dt * vp.x) / cfg.contact_kt;
        anchor[2*k+1] = p.y + (fty + cfg.contact_dt * vp.y) / cfg.contact_kt;
      }
      V3 fw = v3(ftx, fty, fn);
      V3 fb = mulT(w.oR[b], fw);                           // to body frame
      w.fext[b] = w.fext[b] + sv(cross(r, fb), fb);
    } else {
      anchor[2*k] = G1_ANCHOR_FREE; anchor[2*k+1] = G1_ANCHOR_FREE;
    }
    if (fn_out) fn_out[k] = fn;
  }
}

// ------------------------------------------------------------------- ABA
// tau: generalized force (MuJoCo layout; tau[0:6] base wrench dual to qvel).
// Joint damping is applied internally (tau_h -= damping * qd).
// Result: qacc in MuJoCo convention.
// bimp (nullable): per-hinge velocity-feedback coefficient b_h treated
// IMPLICITLY: solves (M + dt*B) qacc = f(q, v) - B v by folding dt*b_h into
// the articulated D (exact per-joint backward-Euler damping; unconditionally
// stable where the explicit update b*dt/m_eff > 2 diverges).
G1_FN void g1_aba(const G1Config& cfg, const G1Real* qvel, const G1Real* tau,
                  G1Ws& w, G1Real* qacc, const G1Real* bimp = nullptr) {
  // init IA, pA with gravity + external forces
  for (int i = 0; i < G1_NB; ++i) {
    V3 c = ld3(g1_com + 3*i);
    w.IA[i] = spatial_inertia(g1_mass[i], c, g1_inertia_com + 9*i);
    V3 gb = mulT(w.oR[i], v3(0, 0, cfg.gravity_z));        // gravity, body frm
    V3 fg = g1_mass[i] * gb;
    SV f = w.fext[i] + sv(cross(c, fg), fg);
    w.pA[i] = cross_force(w.v[i], m6_mul(w.IA[i], w.v[i])) - f;
  }
  // base applied wrench (force world @ base origin, torque body-local)
  {
    V3 fb = mulT(w.oR[0], ld3(tau));
    w.pA[0] = w.pA[0] - sv(ld3(tau + 3), fb);
  }
  // backward
  for (int i = G1_NB - 1; i >= 1; --i) {
    int h = i - 1, par = g1_parent[i];
    G1Real qd = qvel[6 + h];
    w.U[h] = m6_mul(w.IA[i], w.S[h]);
    w.D[h] = svdot(w.S[h], w.U[h]) + g1_armature[h]
           + (bimp ? cfg.dt * bimp[h] : G1Real(0));
    w.u[h] = tau[6 + h] - g1_damping[h] * qd - svdot(w.S[h], w.pA[i]);
    G1Real invd = G1Real(1) / w.D[h];
    SV c = cross_motion(w.v[i], qd * w.S[h]);              // velocity bias
    M6 Ia = w.IA[i];
    m6_sub_outer(Ia, w.U[h], invd);
    SV pa = w.pA[i] + m6_mul(Ia, c) + (w.u[h] * invd) * w.U[h];
    m6_psum_transform(w.IA[par], Ia, w.liMi[i]);
    w.pA[par] = w.pA[par] + force_act(w.liMi[i], pa);
  }
  // base solve: IA[0] a0 = -pA[0]
  w.a[0] = m6_solve(w.IA[0], sv(v3(0,0,0), v3(0,0,0)) - w.pA[0]);
  // forward
  for (int i = 1; i < G1_NB; ++i) {
    int h = i - 1, par = g1_parent[i];
    SV c = cross_motion(w.v[i], qvel[6 + h] * w.S[h]);
    SV ap = motion_act_inv(w.liMi[i], w.a[par]) + c;
    G1Real qdd = (w.u[h] - svdot(w.U[h], ap)) / w.D[h];
    w.a[i] = ap + qdd * w.S[h];
    qacc[6 + h] = qdd;
  }
  // base qacc -> MuJoCo convention:
  //   angular: d/dt omega_local = body-frame spatial angular acceleration
  //   linear:  d/dt v_world = R (a_lin_body + omega x v_body)
  V3 aw = w.a[0].a;
  V3 al = mul(w.oR[0], w.a[0].l + cross(w.v[0].a, w.v[0].l));
  qacc[0] = al.x; qacc[1] = al.y; qacc[2] = al.z;
  qacc[3] = aw.x; qacc[4] = aw.y; qacc[5] = aw.z;
}

// -------------------------------------------------- control + integration
// PD position servo per hinge, torque-limited (matches MuJoCo <position>
// actuator with kp/kv and jnt_actfrcrange clamping), plus soft joint limits.
G1_FN void g1_pd_tau(const G1Config& cfg, const G1Real* qpos,
                     const G1Real* qvel, const G1Real* ctrl, G1Real* tau,
                     G1Real* bimp) {
  for (int i = 0; i < 6; ++i) tau[i] = 0;
  for (int h = 0; h < G1_NJ; ++h) {
    G1Real q = qpos[7 + h], qd = qvel[6 + h];
    G1Real t = g1_act_kp[h] * (ctrl[h] - q) - g1_act_kv[h] * qd;
    G1Real lim = g1_act_frclim[h];
    if (t >  lim) t =  lim;
    if (t < -lim) t = -lim;
    G1Real lo = g1_jnt_range[2*h], hi = g1_jnt_range[2*h + 1];
    G1Real b = g1_act_kv[h] + g1_damping[h];
    if (q < lo) { t += cfg.limit_kp * (lo - q) - cfg.limit_kd * qd; b += cfg.limit_kd; }
    if (q > hi) { t += cfg.limit_kp * (hi - q) - cfg.limit_kd * qd; b += cfg.limit_kd; }
    tau[6 + h] = t;
    bimp[h] = b;
  }
}

// Semi-implicit Euler (velocity first), quaternion integrated with the
// updated body-frame angular velocity. Matches MuJoCo Euler with
// eulerdamp disabled (damping handled explicitly inside g1_aba).
G1_FN void g1_integrate(G1Real dt, const G1Real* qacc, G1Real* qpos,
                        G1Real* qvel) {
  for (int i = 0; i < G1_NV; ++i) qvel[i] += dt * qacc[i];
  qpos[0] += dt * qvel[0];
  qpos[1] += dt * qvel[1];
  qpos[2] += dt * qvel[2];
  Quat q = quat_integrate(Quat{qpos[3], qpos[4], qpos[5], qpos[6]},
                          ld3(qvel + 3), dt);
  qpos[3] = q.w; qpos[4] = q.x; qpos[5] = q.y; qpos[6] = q.z;
  for (int h = 0; h < G1_NJ; ++h) qpos[7 + h] += dt * qvel[6 + h];
}

// One full physics step: FK -> contacts -> PD -> ABA -> integrate.
G1_FN void g1_step(const G1Config& cfg, G1Real* qpos, G1Real* qvel,
                   const G1Real* ctrl, G1Real* anchor, G1Ws& w,
                   G1Real* fn_out = nullptr) {
  for (int i = 0; i < G1_NB; ++i) w.fext[i] = sv(v3(0,0,0), v3(0,0,0));
  g1_fk_vel(qpos, qvel, w);
  g1_contacts(cfg, w, anchor, fn_out);
  G1Real tau[G1_NV], qacc[G1_NV], bimp[G1_NJ];
  g1_pd_tau(cfg, qpos, qvel, ctrl, tau, bimp);
  g1_aba(cfg, qvel, tau, w, qacc, bimp);
  g1_integrate(cfg.dt, qacc, qpos, qvel);
}
