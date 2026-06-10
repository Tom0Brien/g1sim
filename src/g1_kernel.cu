// g1_kernel.cu -- massively parallel G1 stepping: one thread per environment.
//
// GPU build (not testable in this dev container -- no GPU/nvcc; see README):
//   nvcc -O3 -arch=sm_80 --expt-relaxed-constexpr -use_fast_math [...]
//        (full command: see Makefile target `gpu`)
// CPU shim build (functionally exercises this exact code path):
//   make cpu-kernel
//
// Global state is SoA with environment as the fastest axis
// (array[i * nenv + e]) so per-dof load/store loops coalesce across threads.
// All model constants are constexpr (baked at compile time; on device they
// require --expt-relaxed-constexpr).

#ifdef FAKE_CUDA
  #include "fake_cuda.h"
#else
  #include <cuda_runtime.h>
#endif
#include "g1_core.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

#ifdef FAKE_CUDA
  #define G1_LAUNCH(kern, nblocks, nthreads, ...)                      \
    do {                                                               \
      long long _nb = (long long)(nblocks);                            \
      unsigned _nt = (nthreads);                                       \
      _Pragma("omp parallel for schedule(static)")                     \
      for (long long _b = 0; _b < _nb; ++_b)                           \
        for (unsigned _t = 0; _t < _nt; ++_t) {                        \
          gridDim = dim3((unsigned)_nb); blockDim = dim3(_nt);         \
          blockIdx = dim3((unsigned)_b); threadIdx = dim3(_t);         \
          kern(__VA_ARGS__);                                           \
        }                                                              \
    } while (0)
#else
  #define G1_LAUNCH(kern, nblocks, nthreads, ...) \
    kern<<<(nblocks), (nthreads)>>>(__VA_ARGS__)
#endif

#define CUDA_CHECK(x)                                                      \
  do {                                                                     \
    cudaError_t _e = (x);                                                  \
    if (_e != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),  \
              __FILE__, __LINE__);                                         \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// ---------------------------------------------------------------- helpers
G1_FN unsigned g1_hash(unsigned x) {
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16; return x;
}
G1_FN G1Real g1_urand(unsigned& s) {            // uniform [0, 1)
  s = g1_hash(s + 0x9E3779B9u);
  return G1Real(s) * G1Real(2.3283064365386963e-10);
}

// ---------------------------------------------------------------- kernels
// Reset every env to the "stand" keyframe with per-joint uniform noise of
// +-noise rad (and the base dropped from +drop m). Runs on device because
// the constexpr keyframe lives in the compiled binary, not host memory.
__global__ void g1_reset_kernel(int nenv, G1Real* qpos, G1Real* qvel,
                                G1Real* anchor, unsigned seed, G1Real noise,
                                G1Real drop) {
  int e = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= nenv) return;
  unsigned s = g1_hash(seed ^ (unsigned)(e + 1));
  for (int i = 0; i < G1_NQ; ++i) {
    G1Real v = G1Real(g1_qpos_stand[i]);
    if (i == 2) v += drop;
    if (i >= 7) v += noise * (2 * g1_urand(s) - 1);
    qpos[(size_t)i * nenv + e] = v;
  }
  for (int i = 0; i < G1_NV; ++i) qvel[(size_t)i * nenv + e] = 0;
  for (int i = 0; i < 2 * G1_NC; ++i)
    anchor[(size_t)i * nenv + e] = G1_ANCHOR_FREE;
}

__global__ void g1_reset_done_kernel(int nenv, G1Real* qpos, G1Real* qvel,
                                     G1Real* anchor, const uint8_t* done,
                                     unsigned seed, G1Real noise,
                                     G1Real drop) {
  int e = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= nenv) return;
  if (!done[e]) return;

  unsigned s = g1_hash(seed ^ (unsigned)(e + 1));
  for (int i = 0; i < G1_NQ; ++i) {
    G1Real v = G1Real(g1_qpos_stand[i]);
    if (i == 2) v += drop;
    if (i >= 7) v += noise * (2 * g1_urand(s) - 1);
    qpos[(size_t)i * nenv + e] = v;
  }
  for (int i = 0; i < G1_NV; ++i) qvel[(size_t)i * nenv + e] = 0;
  for (int i = 0; i < 2 * G1_NC; ++i)
    anchor[(size_t)i * nenv + e] = G1_ANCHOR_FREE;
}

// nsub physics substeps per launch. ctrl holds PD position targets.
__global__ void g1_step_kernel(int nenv, int nsub, G1Config cfg,
                               G1Real* __restrict__ qpos,
                               G1Real* __restrict__ qvel,
                               G1Real* __restrict__ anchor,
                               const G1Real* __restrict__ ctrl) {
  int e = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (e >= nenv) return;
  G1Real qp[G1_NQ], qv[G1_NV], ct[G1_NU], an[2 * G1_NC];
  for (int i = 0; i < G1_NQ; ++i)     qp[i] = qpos[(size_t)i * nenv + e];
  for (int i = 0; i < G1_NV; ++i)     qv[i] = qvel[(size_t)i * nenv + e];
  for (int i = 0; i < G1_NU; ++i)     ct[i] = ctrl[(size_t)i * nenv + e];
  for (int i = 0; i < 2 * G1_NC; ++i) an[i] = anchor[(size_t)i * nenv + e];
  G1Ws w;
  for (int s = 0; s < nsub; ++s) g1_step(cfg, qp, qv, ct, an, w);
  for (int i = 0; i < G1_NQ; ++i)     qpos[(size_t)i * nenv + e] = qp[i];
  for (int i = 0; i < G1_NV; ++i)     qvel[(size_t)i * nenv + e] = qv[i];
  for (int i = 0; i < 2 * G1_NC; ++i) anchor[(size_t)i * nenv + e] = an[i];
}

extern "C" {
  void g1_cuda_step(int nenv, int nsub, G1Real* qpos, G1Real* qvel, G1Real* anchor, const G1Real* ctrl, cudaStream_t stream) {
    int tpb = 128;
    int nblocks = (nenv + tpb - 1) / tpb;
    G1Config cfg = g1_default_config();
#ifdef FAKE_CUDA
    g1_step_kernel(nenv, nsub, cfg, qpos, qvel, anchor, ctrl);
#else
    g1_step_kernel<<<nblocks, tpb, 0, stream>>>(nenv, nsub, cfg, qpos, qvel, anchor, ctrl);
#endif
  }

  void g1_cuda_reset_done(int nenv, G1Real* qpos, G1Real* qvel, G1Real* anchor, const uint8_t* done, unsigned seed, G1Real noise, G1Real drop, cudaStream_t stream) {
    int tpb = 128;
    int nblocks = (nenv + tpb - 1) / tpb;
#ifdef FAKE_CUDA
    g1_reset_done_kernel(nenv, qpos, qvel, anchor, done, seed, noise, drop);
#else
    g1_reset_done_kernel<<<nblocks, tpb, 0, stream>>>(nenv, qpos, qvel, anchor, done, seed, noise, drop);
#endif
  }
}

// -------------------------------------------------------------- benchmark
int main(int argc, char** argv) {
  int nenv = 4096, nsteps = 500, nsub = 1, dump = 0;
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--nenv"))   nenv = atoi(argv[++i]);
    if (!strcmp(argv[i], "--steps"))  nsteps = atoi(argv[++i]);
    if (!strcmp(argv[i], "--nsub"))   nsub = atoi(argv[++i]);
    if (!strcmp(argv[i], "--dump"))   { dump = 1; nenv = 1; }
  }
  G1Config cfg = g1_default_config();
  size_t nq = (size_t)G1_NQ * nenv, nv = (size_t)G1_NV * nenv;
  size_t nu = (size_t)G1_NU * nenv, na = (size_t)2 * G1_NC * nenv;
  G1Real *qpos, *qvel, *anchor, *ctrl;
  CUDA_CHECK(cudaMalloc(&qpos, nq * sizeof(G1Real)));
  CUDA_CHECK(cudaMalloc(&qvel, nv * sizeof(G1Real)));
  CUDA_CHECK(cudaMalloc(&anchor, na * sizeof(G1Real)));
  CUDA_CHECK(cudaMalloc(&ctrl, nu * sizeof(G1Real)));

  // PD targets: the keyframe's hinge angles (qpos_stand[7:]) for every env.
  {
    G1Real* h = (G1Real*)malloc(nu * sizeof(G1Real));
    for (int i = 0; i < G1_NU; ++i)
      for (int e = 0; e < nenv; ++e)
        h[(size_t)i * nenv + e] = G1Real(g1_qpos_stand[7 + i]);
    CUDA_CHECK(cudaMemcpy(ctrl, h, nu * sizeof(G1Real),
                          cudaMemcpyHostToDevice));
    free(h);
  }

  int tpb = 128, nblocks = (nenv + tpb - 1) / tpb;
  G1_LAUNCH(g1_reset_kernel, nblocks, tpb, nenv, qpos, qvel, anchor,
            dump ? 0u : 1234u, dump ? G1Real(0) : G1Real(0.05),
            G1Real(0.02));
  CUDA_CHECK(cudaDeviceSynchronize());

  if (dump) {                       // single deterministic env -> stdout
    G1Real* h = (G1Real*)malloc(nq * sizeof(G1Real));
    for (int s = 0; s < nsteps; ++s) {
      G1_LAUNCH(g1_step_kernel, nblocks, tpb, nenv, nsub, cfg, qpos, qvel,
                anchor, ctrl);
      CUDA_CHECK(cudaDeviceSynchronize());
      CUDA_CHECK(cudaMemcpy(h, qpos, nq * sizeof(G1Real),
                            cudaMemcpyDeviceToHost));
      for (int i = 0; i < G1_NQ; ++i) printf("%.9g ", (double)h[i]);
      printf("\n");
    }
    free(h);
    return 0;
  }

  // warmup, then timed loop
  G1_LAUNCH(g1_step_kernel, nblocks, tpb, nenv, nsub, cfg, qpos, qvel,
            anchor, ctrl);
  CUDA_CHECK(cudaDeviceSynchronize());
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int s = 0; s < nsteps; ++s)
    G1_LAUNCH(g1_step_kernel, nblocks, tpb, nenv, nsub, cfg, qpos, qvel,
              anchor, ctrl);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaGetLastError());
  auto t1 = std::chrono::high_resolution_clock::now();
  double sec = std::chrono::duration<double>(t1 - t0).count();
  double envsteps = (double)nenv * nsteps * nsub;

  // sanity: pull a few env heights, assert all finite and standing
  G1Real* h = (G1Real*)malloc(nq * sizeof(G1Real));
  CUDA_CHECK(cudaMemcpy(h, qpos, nq * sizeof(G1Real), cudaMemcpyDeviceToHost));
  int bad = 0;
  double hmin = 1e30, hmax = -1e30;
  for (int e = 0; e < nenv; ++e) {
    double z = (double)h[(size_t)2 * nenv + e];
    if (!(z > 0.2 && z < 1.2)) ++bad;
    if (z < hmin) hmin = z;
    if (z > hmax) hmax = z;
  }
  printf("g1sim %s  %d envs x %d steps x %d substeps\n",
#ifdef FAKE_CUDA
         "[CPU shim]",
#else
         "[CUDA]",
#endif
         nenv, nsteps, nsub);
  printf("  %.3f s  ->  %.3g env-steps/s\n", sec, envsteps / sec);
  printf("  pelvis height after %.2f s sim: min %.3f max %.3f, %d/%d outside"
         " [0.2, 1.2]\n", nsteps * nsub * (double)cfg.dt, hmin, hmax, bad,
         nenv);
  free(h);
  cudaFree(qpos); cudaFree(qvel); cudaFree(anchor); cudaFree(ctrl);
  return bad ? 2 : 0;
}
