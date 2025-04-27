import numpy as np
from .primitive import Rays
from utils import extrinsic_matrix,generate_rays

class Camera:

    def __init__(self, look_at, up, cam, fov, W, H, jitter):
        
        self.look_at = np.array(look_at)
        self.up = np.array(up)
        self.cam = np.array(cam)
        self.fov = np.float64(fov)
        self.W = W
        self.H = H
        self.cam_rays = None
        self.jitter = jitter
        if not jitter:
            self.cam_rays = self.generate_cam_rays()

    def generate_cam_rays(self):
        
        ext_matrix = extrinsic_matrix(self.look_at,self.up,self.cam)
        rays_dir = generate_rays(self.fov,self.H,self.W, self.jitter)
        
        rays_dir = np.dot(ext_matrix,rays_dir.T).T
        rays_dir = rays_dir[:,:3] - self.cam
        rays_dir = rays_dir/np.linalg.norm(rays_dir,axis=-1,keepdims=True)

        return Rays(self.cam[np.newaxis],rays_dir)
    
