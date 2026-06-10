import ctypes
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(HERE, "..", "build", "libg1cuda.so")

if not os.path.exists(LIB_PATH):
    raise RuntimeError(f"Could not find {LIB_PATH}. Run 'make gpu' first.")

lib = ctypes.CDLL(LIB_PATH)

# void g1_cuda_step(int nenv, int nsub, G1Real* qpos, G1Real* qvel, G1Real* anchor, const G1Real* ctrl, cudaStream_t stream)
lib.g1_cuda_step.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
]

# void g1_cuda_reset_done(int nenv, G1Real* qpos, G1Real* qvel, G1Real* anchor, const uint8_t* done, unsigned seed, G1Real noise, G1Real drop, cudaStream_t stream)
lib.g1_cuda_reset_done.argtypes = [
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_uint,
    ctypes.c_float,
    ctypes.c_float,
    ctypes.c_void_p,
]

class G1Sim:
    def __init__(self, nenv: int, device: torch.device):
        self.nenv = nenv
        self.device = device
        
        # Dimensions based on g1_model.h
        self.NQ = 36  # Base (7) + 29 joints
        self.NV = 35  # Base (6) + 29 joints
        self.NU = 29
        self.NC = 8
        
        # SoA layout: (dim, nenv)
        self.qpos = torch.zeros((self.NQ, self.nenv), dtype=torch.float32, device=self.device)
        self.qvel = torch.zeros((self.NV, self.nenv), dtype=torch.float32, device=self.device)
        self.anchor = torch.full((2 * self.NC, self.nenv), 1e30, dtype=torch.float32, device=self.device)
        self.ctrl = torch.zeros((self.NU, self.nenv), dtype=torch.float32, device=self.device)
        
        # Helper mask
        self.done = torch.ones(self.nenv, dtype=torch.uint8, device=self.device)
        
    def reset_done(self, done: torch.Tensor, seed: int = 42, noise: float = 0.05, drop: float = 0.02):
        """Resets environments where done is True/1."""
        assert done.shape == (self.nenv,)
        assert done.dtype == torch.uint8
        assert done.device == self.device
        assert done.is_contiguous()
        
        stream = torch.cuda.current_stream(self.device).cuda_stream
        
        lib.g1_cuda_reset_done(
            self.nenv,
            self.qpos.data_ptr(),
            self.qvel.data_ptr(),
            self.anchor.data_ptr(),
            done.data_ptr(),
            seed,
            noise,
            drop,
            stream
        )
        
    def reset_all(self, seed: int = 42, noise: float = 0.05, drop: float = 0.02):
        """Resets all environments."""
        self.done.fill_(1)
        self.reset_done(self.done, seed, noise, drop)
        
    def step(self, nsub: int = 1):
        """Steps all environments forward by nsub substeps."""
        assert self.ctrl.is_contiguous()
        assert self.qpos.is_contiguous()
        
        stream = torch.cuda.current_stream(self.device).cuda_stream
        
        lib.g1_cuda_step(
            self.nenv,
            nsub,
            self.qpos.data_ptr(),
            self.qvel.data_ptr(),
            self.anchor.data_ptr(),
            self.ctrl.data_ptr(),
            stream
        )
