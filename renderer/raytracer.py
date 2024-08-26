import numpy as np
from sampling import *
from utils import *
from primitives import Rays,Geometry,Scene

class RayTracer(object): 

    def __init__(self,scene,sampler, camconfig= {'H':64,'W':64,'fov':60,
                           'look':np.array([-10,1,1]),'up':np.array([0,1,0]),
                           'cam':np.array([10,1,1])}) -> None:
        self.scene = Scene(**scene)
        self.sampler = Sampling(**sampler)
        self.H = camconfig["H"]
        self.W = camconfig["W"]
        self.fov = camconfig["fov"]
        self.look = camconfig["look"]
        self.up = camconfig["up"]
        self.cam = camconfig["cam"]
    
    def generate_cam_rays(self):
        
        ext_matrix = extrinsic_matrix(self.look,self.up,self.cam)
        rays_dir = generate_rays(self.fov,self.H,self.W)
        
        rays_dir = np.dot(ext_matrix,rays_dir).T
        rays_dir = rays_dir[:3] - self.cam
        rays_dir = rays_dir/np.linalg.norm(rays_dir,axis=-1,keepdims=True)

        return Rays(self.cam,rays_dir)

    def intersect_with_scene(self,rays):
        dist, normals, ids = self.scene.intersect(rays)
        normals = np.where(normals != np.array([np.inf, np.inf, np.inf]),
                           normals, np.array([0, 0, 0]))

        hit_points = rays(dist)

        # NOTE: When ids == -1 (i.e., no hit), you get a valid BRDF ([0,0,0,0]), L_e ([0,0,0]), and objects id (-1)!
        brdf_params = np.concatenate((np.array([obj.brdf_params for obj in self.scene.geometries]),
                                      np.array([0, 0, 0, 1])[np.newaxis, :]))[ids]
        l_e = np.concatenate((np.array([obj.Le for obj in self.geometries]),
                              np.array([0, 0, 0])[np.newaxis, :]))[ids]
        l_e = np.where(np.logical_and(l_e != np.array([0, 0, 0]), (ids != -1)[:, np.newaxis]), l_e, 0)
        
        return hit_points,normals,brdf_params,l_e
    def render(self,rays):
        
        hit_points,normals,brdf_params,L_e = self.intersect_with_scene(rays)
        
        # initialize the output image and Directly render light sources
        L = np.zeros(normals.shape, dtype=np.float64)
        L += L_e
        
        self.sampler.set_initial_params(hit_points,normals,brdf_params, normals, brdf_params,rays.Ds)
        
        for l,light in enumerate(self.scene.lights):
            self.sampler.set_light(light)
            ### sample new rays directions
            shadow_rays,prob = self.sampler.shadow_rays()
            ### check which intersect light source, take contributions
            _,_,_,light_e = self.intersect_with_scene(shadow_rays)
            ### Calc direct illumination
            L += self.sampler.illumination(light_e,shadow_rays.Ds,prob)

        return L.reshape((self.H, self.W, 3))