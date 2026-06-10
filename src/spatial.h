// spatial.h -- minimal spatial (Featherstone) algebra, host/device.
// Conventions:
//   * Spatial vectors are [angular; linear] in BODY-LOCAL coordinates.
//   * SE3 X = (R, p) is the pose of frame B expressed in frame A
//     (point map: x_A = R x_B + p).
//   * Quaternions are (w, x, y, z), MuJoCo order.
#pragma once
#include "g1_types.h"
#include <cmath>

struct V3 { G1Real x, y, z; };
G1_FN V3 v3(G1Real x, G1Real y, G1Real z) { V3 r{x, y, z}; return r; }
G1_FN V3 operator+(V3 a, V3 b) { return v3(a.x+b.x, a.y+b.y, a.z+b.z); }
G1_FN V3 operator-(V3 a, V3 b) { return v3(a.x-b.x, a.y-b.y, a.z-b.z); }
G1_FN V3 operator*(G1Real s, V3 a) { return v3(s*a.x, s*a.y, s*a.z); }
G1_FN G1Real dot(V3 a, V3 b) { return a.x*b.x + a.y*b.y + a.z*b.z; }
G1_FN V3 cross(V3 a, V3 b) {
  return v3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

struct M3 { G1Real m[9]; };                       // row-major
G1_FN V3 mul(const M3& A, V3 v) {
  return v3(A.m[0]*v.x + A.m[1]*v.y + A.m[2]*v.z,
            A.m[3]*v.x + A.m[4]*v.y + A.m[5]*v.z,
            A.m[6]*v.x + A.m[7]*v.y + A.m[8]*v.z);
}
G1_FN V3 mulT(const M3& A, V3 v) {                // A^T v
  return v3(A.m[0]*v.x + A.m[3]*v.y + A.m[6]*v.z,
            A.m[1]*v.x + A.m[4]*v.y + A.m[7]*v.z,
            A.m[2]*v.x + A.m[5]*v.y + A.m[8]*v.z);
}
G1_FN M3 matmul(const M3& A, const M3& B) {
  M3 C;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      C.m[3*i+j] = A.m[3*i+0]*B.m[0*3+j] + A.m[3*i+1]*B.m[1*3+j]
                 + A.m[3*i+2]*B.m[2*3+j];
  return C;
}

struct Quat { G1Real w, x, y, z; };
G1_FN Quat qnorm(Quat q) {
  G1Real n = sqrt(q.w*q.w + q.x*q.x + q.y*q.y + q.z*q.z);
  G1Real s = G1Real(1) / n;
  Quat r{q.w*s, q.x*s, q.y*s, q.z*s};
  return r;
}
G1_FN Quat qmul(Quat a, Quat b) {
  Quat r{a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
         a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
         a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
         a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w};
  return r;
}
G1_FN M3 quat_to_mat(Quat q) {
  G1Real w=q.w, x=q.x, y=q.y, z=q.z;
  M3 R;
  R.m[0]=1-2*(y*y+z*z); R.m[1]=2*(x*y-w*z);   R.m[2]=2*(x*z+w*y);
  R.m[3]=2*(x*y+w*z);   R.m[4]=1-2*(x*x+z*z); R.m[5]=2*(y*z-w*x);
  R.m[6]=2*(x*z-w*y);   R.m[7]=2*(y*z+w*x);   R.m[8]=1-2*(x*x+y*y);
  return R;
}
G1_FN M3 axis_angle_mat(V3 a, G1Real q) {        // |a| = 1 (Rodrigues)
  G1Real c = cos(q), s = sin(q), t = 1 - c;
  M3 R;
  R.m[0]=c+t*a.x*a.x;     R.m[1]=t*a.x*a.y-s*a.z; R.m[2]=t*a.x*a.z+s*a.y;
  R.m[3]=t*a.x*a.y+s*a.z; R.m[4]=c+t*a.y*a.y;     R.m[5]=t*a.y*a.z-s*a.x;
  R.m[6]=t*a.x*a.z-s*a.y; R.m[7]=t*a.y*a.z+s*a.x; R.m[8]=c+t*a.z*a.z;
  return R;
}
// q <- q (x) exp(omega_local * dt / 2): integrate body-frame angular velocity
// (matches MuJoCo mju_quatIntegrate semantics for free joints).
G1_FN Quat quat_integrate(Quat q, V3 w, G1Real dt) {
  G1Real angle = sqrt(dot(w, w)) * dt;
  if (angle < G1Real(1e-12)) return q;
  G1Real s = sin(angle/2) / (angle/dt);          // sin(a/2)/|w|
  Quat dq{G1Real(cos(angle/2)), s*w.x, s*w.y, s*w.z};
  return qnorm(qmul(q, dq));
}

// ------------------------------------------------------------------- SE3
struct SE3 { M3 R; V3 p; };
G1_FN SE3 se3_compose(const SE3& A, const SE3& B) {  // A * B
  SE3 C;
  C.R = matmul(A.R, B.R);
  C.p = mul(A.R, B.p) + A.p;
  return C;
}

struct SV { V3 a, l; };                          // [angular; linear]
G1_FN SV sv(V3 a, V3 l) { SV r{a, l}; return r; }
G1_FN SV operator+(SV u, SV w) { return sv(u.a + w.a, u.l + w.l); }
G1_FN SV operator-(SV u, SV w) { return sv(u.a - w.a, u.l - w.l); }
G1_FN SV operator*(G1Real s, SV u) { return sv(s*u.a, s*u.l); }
G1_FN G1Real svdot(SV u, SV w) { return dot(u.a, w.a) + dot(u.l, w.l); }

// Motion vector, frame B -> frame A, X = pose of B in A.
G1_FN SV motion_act(const SE3& X, SV m) {
  V3 a = mul(X.R, m.a);
  return sv(a, mul(X.R, m.l) + cross(X.p, a));
}
// Motion vector, frame A -> frame B.
G1_FN SV motion_act_inv(const SE3& X, SV m) {
  return sv(mulT(X.R, m.a), mulT(X.R, m.l - cross(X.p, m.a)));
}
// Force vector, frame B -> frame A.
G1_FN SV force_act(const SE3& X, SV f) {
  V3 l = mul(X.R, f.l);
  return sv(mul(X.R, f.a) + cross(X.p, l), l);
}
// v (cross) m  (motion x motion)
G1_FN SV cross_motion(SV v, SV m) {
  return sv(cross(v.a, m.a), cross(v.a, m.l) + cross(v.l, m.a));
}
// v (cross*) f  (motion x force)
G1_FN SV cross_force(SV v, SV f) {
  return sv(cross(v.a, f.a) + cross(v.l, f.l), cross(v.a, f.l));
}

// ------------------------------------------------------- 6x6 (articulated)
struct M6Sym { G1Real m[21]; };                     // lower-triangular packed

G1_FN void m6_zero(M6Sym& A) { for (int i = 0; i < 21; ++i) A.m[i] = 0; }

G1_FN SV m6_mul(const M6Sym& A, SV u) {
  G1Real x[6] = {u.a.x, u.a.y, u.a.z, u.l.x, u.l.y, u.l.z};
  G1Real y[6] = {0,0,0,0,0,0};
  int idx = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j < i; ++j) {
      G1Real A_ij = A.m[idx++];
      y[i] += A_ij * x[j];
      y[j] += A_ij * x[i];
    }
    y[i] += A.m[idx++] * x[i]; // j == i
  }
  return sv(v3(y[0], y[1], y[2]), v3(y[3], y[4], y[5]));
}

// A -= (U U^T) / d
G1_FN void m6_sub_outer(M6Sym& A, SV U, G1Real inv_d) {
  G1Real u[6] = {U.a.x, U.a.y, U.a.z, U.l.x, U.l.y, U.l.z};
  int idx = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j <= i; ++j) {
      A.m[idx++] -= u[i] * u[j] * inv_d;
    }
  }
}

// Rigid-body spatial inertia about the body-frame origin
G1_FN M6Sym spatial_inertia(G1Real mass, V3 c, const G1Real Icom[9]) {
  M6Sym I;
  G1Real cx[9] = {0, -c.z, c.y,  c.z, 0, -c.x,  -c.y, c.x, 0};
  int idx = 0;
  for (int i = 0; i < 6; ++i) {
    for (int j = 0; j <= i; ++j) {
      G1Real val = 0;
      if (i < 3 && j < 3) {
        G1Real s = 0;
        for (int k = 0; k < 3; ++k) s += cx[3*i+k] * cx[3*k+j];
        val = Icom[3*i+j] - mass * s;
      } else if (i >= 3 && j < 3) {
        val = -mass * cx[3*(i-3) + j];
      } else {
        val = (i == j) ? mass : G1Real(0);
      }
      I.m[idx++] = val;
    }
  }
  return I;
}

G1_FN M3 m6_sym_get_J(const M6Sym& I) {
  return {I.m[0], I.m[1], I.m[3],
          I.m[1], I.m[2], I.m[4],
          I.m[3], I.m[4], I.m[5]};
}
G1_FN M3 m6_sym_get_H(const M6Sym& I) {
  return {I.m[6], I.m[10], I.m[15],
          I.m[7], I.m[11], I.m[16],
          I.m[8], I.m[12], I.m[17]};
}
G1_FN M3 m6_sym_get_M(const M6Sym& I) {
  return {I.m[9],  I.m[13], I.m[18],
          I.m[13], I.m[14], I.m[19],
          I.m[18], I.m[19], I.m[20]};
}

G1_FN M3 matmul_ABT(const M3& A, const M3& B) {
  M3 C;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      C.m[3*i+j] = A.m[3*i+0]*B.m[3*j+0] + A.m[3*i+1]*B.m[3*j+1] + A.m[3*i+2]*B.m[3*j+2];
  return C;
}

G1_FN M3 px_mul(V3 p, const M3& A) {
  M3 C;
  for (int j=0; j<3; ++j) {
    C.m[0*3+j] = -p.z * A.m[1*3+j] + p.y * A.m[2*3+j];
    C.m[1*3+j] =  p.z * A.m[0*3+j] - p.x * A.m[2*3+j];
    C.m[2*3+j] = -p.y * A.m[0*3+j] + p.x * A.m[1*3+j];
  }
  return C;
}

G1_FN M3 mul_px(const M3& A, V3 p) {
  M3 C;
  for (int i=0; i<3; ++i) {
    C.m[i*3+0] =  A.m[i*3+1] * p.z - A.m[i*3+2] * p.y;
    C.m[i*3+1] = -A.m[i*3+0] * p.z + A.m[i*3+2] * p.x;
    C.m[i*3+2] =  A.m[i*3+0] * p.y - A.m[i*3+1] * p.x;
  }
  return C;
}

// Transform articulated inertia from child frame i to parent frame A
G1_FN void m6_psum_transform(M6Sym& IA_parent, const M6Sym& Ii, const SE3& X) {
  M3 Ji = m6_sym_get_J(Ii);
  M3 Hi = m6_sym_get_H(Ii);
  M3 Mi = m6_sym_get_M(Ii);
  
  M3 tJ = matmul_ABT(matmul(X.R, Ji), X.R);
  M3 tH = matmul_ABT(matmul(X.R, Hi), X.R);
  M3 tM = matmul_ABT(matmul(X.R, Mi), X.R);
  
  M3 px_tM = px_mul(X.p, tM);
  M3 HA;
  for (int i=0; i<9; ++i) HA.m[i] = tH.m[i] + px_tM.m[i];
  
  M3 tH_T;
  for(int i=0;i<3;++i) for(int j=0;j<3;++j) tH_T.m[3*i+j] = tH.m[3*j+i];
  
  M3 px_tHT = px_mul(X.p, tH_T);
  M3 HA_px = mul_px(HA, X.p);
  M3 JA;
  for (int i=0; i<9; ++i) JA.m[i] = tJ.m[i] + px_tHT.m[i] - HA_px.m[i];
  
  IA_parent.m[0] += JA.m[0];
  IA_parent.m[1] += JA.m[3]; IA_parent.m[2] += JA.m[4];
  IA_parent.m[3] += JA.m[6]; IA_parent.m[4] += JA.m[7]; IA_parent.m[5] += JA.m[8];
  
  IA_parent.m[6] += HA.m[0]; IA_parent.m[7] += HA.m[3]; IA_parent.m[8] += HA.m[6];
  IA_parent.m[10]+= HA.m[1]; IA_parent.m[11]+= HA.m[4]; IA_parent.m[12]+= HA.m[7];
  IA_parent.m[15]+= HA.m[2]; IA_parent.m[16]+= HA.m[5]; IA_parent.m[17]+= HA.m[8];
  
  IA_parent.m[9] += tM.m[0];
  IA_parent.m[13]+= tM.m[3]; IA_parent.m[14]+= tM.m[4];
  IA_parent.m[18]+= tM.m[6]; IA_parent.m[19]+= tM.m[7]; IA_parent.m[20]+= tM.m[8];
}

// Solve SPD 6x6 system A x = b (in-place Cholesky on a copy).
G1_FN SV m6_solve(const M6Sym& A, SV b) {
  G1Real L[21];
  for (int i = 0; i < 21; ++i) L[i] = A.m[i];
  for (int j = 0; j < 6; ++j) {
    int jj = j*(j+1)/2 + j;
    for (int k = 0; k < j; ++k) L[jj] -= L[j*(j+1)/2 + k] * L[j*(j+1)/2 + k];
    L[jj] = sqrt(L[jj]);
    G1Real inv = G1Real(1) / L[jj];
    for (int i = j+1; i < 6; ++i) {
      int ij = i*(i+1)/2 + j;
      for (int k = 0; k < j; ++k) L[ij] -= L[i*(i+1)/2 + k] * L[j*(j+1)/2 + k];
      L[ij] *= inv;
    }
  }
  G1Real y[6] = {b.a.x, b.a.y, b.a.z, b.l.x, b.l.y, b.l.z};
  for (int i = 0; i < 6; ++i) {                  // forward
    for (int k = 0; k < i; ++k) y[i] -= L[i*(i+1)/2 + k] * y[k];
    y[i] /= L[i*(i+1)/2 + i];
  }
  for (int i = 5; i >= 0; --i) {                 // backward
    for (int k = i+1; k < 6; ++k) y[i] -= L[k*(k+1)/2 + i] * y[k];
    y[i] /= L[i*(i+1)/2 + i];
  }
  return sv(v3(y[0], y[1], y[2]), v3(y[3], y[4], y[5]));
}
