// fake_cuda.h -- minimal CPU shim so g1_kernel.cu compiles & runs without a
// GPU (build with -DFAKE_CUDA -fopenmp -x c++). Kernel launches go through
// the G1_LAUNCH macro in g1_kernel.cu; thread/block indices are thread_local
// so an OpenMP team emulates the grid. Device pointers are host pointers.
#pragma once
#include <cstdlib>
#include <cstring>

struct dim3 {
  unsigned x, y, z;
  dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
};

#define __global__
#define __device__
#define __host__
#define __forceinline__ inline

inline thread_local dim3 threadIdx, blockIdx, blockDim, gridDim;

using cudaError_t = int;
constexpr cudaError_t cudaSuccess = 0;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost,
                      cudaMemcpyDeviceToDevice };

template <class T>
inline cudaError_t cudaMalloc(T** p, size_t n) { *p = (T*)malloc(n); return 0; }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) {
  memcpy(d, s, n); return 0;
}
inline cudaError_t cudaMemset(void* d, int v, size_t n) { memset(d, v, n); return 0; }
inline cudaError_t cudaFree(void* p) { free(p); return 0; }
inline cudaError_t cudaDeviceSynchronize() { return 0; }
inline cudaError_t cudaGetLastError() { return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return "fake_cuda"; }
