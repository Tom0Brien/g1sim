# Unitree G1 specialized GPU simulator
ARCH ?= sm_80          # set to your GPU (sm_89 Ada, sm_90 Hopper, ...)
CXX  ?= g++
NVCC ?= nvcc
PYTHON ?= python

ifeq ($(OS),Windows_NT)
  HOSTLIB_OUT = build/libg1host.dll
  CPU_KERNEL_OUT = build/g1bench_cpu.exe
else
  HOSTLIB_OUT = build/libg1host.so
  CPU_KERNEL_OUT = build/g1bench_cpu
endif

all: hostlib cpu-kernel

build:
	$(PYTHON) -c "import os; os.makedirs('build', exist_ok=True)"

# Double-precision host library (oracle validation via ctypes)
hostlib: build
	$(CXX) -O2 -fPIC -shared -DG1_PRECISION=8 -Isrc src/host_lib.cpp \
	  -o $(HOSTLIB_OUT) -Wall -Wextra

# Kernel code path compiled for CPU through the fake-CUDA shim (float)
cpu-kernel: build
	$(CXX) -O3 -march=native -fopenmp -std=c++17 -DFAKE_CUDA \
	  -Isrc -Imodel -x c++ src/g1_kernel.cu -o $(CPU_KERNEL_OUT) \
	  -Wall -Wextra

# Real GPU build (requires CUDA toolkit; not testable in the dev container)
gpu: build
	$(NVCC) -O3 -arch=$(ARCH) --expt-relaxed-constexpr -use_fast_math \
	  -std=c++17 -Isrc src/g1_kernel.cu -o build/g1bench

# Regenerate src/g1_model.h + model/g1_stripped.xml from the menagerie MJCF
model:
	$(PYTHON) model/gen_model.py

test: hostlib cpu-kernel
	$(PYTHON) tests/test_vs_mujoco.py
	$(PYTHON) tests/test_kernel_vs_host.py

bench-cpu: cpu-kernel
	$(CPU_KERNEL_OUT) --nenv 4096 --steps 250

clean:
	$(PYTHON) -c "import shutil; shutil.rmtree('build', ignore_errors=True)"

.PHONY: all hostlib cpu-kernel gpu model test bench-cpu clean
