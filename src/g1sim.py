import ctypes
import os
import torch
import mujoco

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
        
        # RL specific states
        self.commands = torch.zeros((3, self.nenv), dtype=torch.float32, device=self.device)
        self.last_actions = torch.zeros((self.NU, self.nenv), dtype=torch.float32, device=self.device)
        
        # Helper mask
        self.done = torch.ones(self.nenv, dtype=torch.uint8, device=self.device)
        
        # Nominal stand pose from XML
        m = mujoco.MjModel.from_xml_path(os.path.join(HERE, "..", "model", "g1_stripped.xml"))
        self.default_qpos = torch.tensor(m.key_qpos[0], dtype=torch.float32, device=self.device).unsqueeze(1)
        self.default_joint_pos = self.default_qpos[7:36]
        self.global_gravity = torch.zeros((3, self.nenv), dtype=torch.float32, device=self.device)
        self.global_gravity[2, :] = -1.0
        
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
        
        self.last_actions.copy_(self.ctrl)
        
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

    def get_obs(self) -> torch.Tensor:
        """
        Computes the RL observation space:
        1. Commands (3)
        2. Projected gravity (3)
        3. Base linear velocity (3)
        4. Base angular velocity (3)
        5. Joint position error (29)
        6. Joint velocities (29)
        7. Previous actions (29)
        Returns: (nenv, 99)
        """
        # Base quaternion q = [w, x, y, z]
        q = self.qpos[3:7, :]
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        
        # Inverse vector part
        q_vec_inv = torch.stack([-qx, -qy, -qz], dim=0)
        
        def quat_rotate_inverse(v):
            uv = torch.cross(q_vec_inv, v, dim=0)
            uuv = torch.cross(q_vec_inv, uv, dim=0)
            return v + 2.0 * (qw * uv + uuv)
            
        proj_gravity = quat_rotate_inverse(self.global_gravity)
        base_lin_vel = quat_rotate_inverse(self.qvel[0:3, :])
        
        # Assemble observation (dim, nenv) -> (nenv, dim)
        obs = torch.cat([
            self.commands,                     # 3
            proj_gravity,                      # 3
            base_lin_vel,                      # 3
            self.qvel[3:6, :],                 # 3
            self.qpos[7:36, :] - self.default_joint_pos,  # 29
            self.qvel[6:35, :],                # 29
            self.last_actions                  # 29
        ], dim=0).T
        
        return obs
