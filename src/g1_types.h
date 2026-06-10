// g1_types.h -- shared scalar/type/config definitions for the G1 simulator.
#pragma once

#if defined(__CUDACC__)
  #define G1_FN __host__ __device__ inline
#else
  #define G1_FN inline
#endif

// Model constants live in headers as constexpr arrays.
#if defined(__CUDACC__)
  #define G1_MODEL_CONST __constant__
#else
  #define G1_MODEL_CONST static constexpr
#endif

// Precision: 8 = double (validation vs MuJoCo oracle), 4 = float (GPU speed).
#ifndef G1_PRECISION
  #define G1_PRECISION 4
#endif
#if G1_PRECISION == 8
  typedef double G1Real;
#else
  typedef float G1Real;
#endif

// ----------------------------------------------------------------- config
// Compliant ground-contact model (plane z=0). Deliberately NOT MuJoCo's
// convex solver: spring-damper normal + viscous-capped Coulomb friction.
// Gains sized for the foot's articulated effective mass at the contact
// point (~1 kg via ankle armature lever): omega*dt ~ 0.28, dn*dt/m ~ 0.4.
// Static penetration ~2 mm at 8 points under the ~35 kg robot.
struct G1Config {
  G1Real gravity_z;     // -9.81
  G1Real contact_kn;    // normal stiffness  [N/m]
  G1Real contact_dn;    // normal damping    [N s/m]
  G1Real contact_kt;    // tangential anchor-spring stiffness [N/m]
  G1Real contact_dt;    // tangential damping [N s/m]
  G1Real limit_kp;      // soft joint-limit stiffness [N m/rad]
  G1Real limit_kd;      // soft joint-limit damping  [N m s/rad]
  G1Real dt;            // physics substep
};

G1_FN G1Config g1_default_config() {
  G1Config c;
  c.gravity_z  = G1Real(-9.81);
  c.contact_kn = G1Real(2e4);
  c.contact_dn = G1Real(200);
  c.contact_kt = G1Real(1e4);
  c.contact_dt = G1Real(200);
  c.limit_kp   = G1Real(200);
  c.limit_kd   = G1Real(5);
  c.dt         = G1Real(2e-3);
  return c;
}
