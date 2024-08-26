from .primitives import *
import numpy as np

class Scene(object):
    def __init__(self):
        """ Initialize the scene. """
        
        # Scene objects. Set using add_geometries()
        self.geometries = []

        # Light sources. Set using add_lights()
        self.lights = []


    def add_geometries(self, geometries):
        """ 
        Adds a list of geometries to the scene.
        
        For geometries with non-zero emission,
        additionally add them to the light list.
        """
        for i in range(len(geometries)):
            if (geometries[i].Le != np.array([0, 0, 0])).any():
                self.add_lights([geometries[i]])

        self.geometries.extend(geometries)

    def add_lights(self, lights):
        """ Adds a list of lights to the scene. """
        self.lights.extend(lights)

    def intersect(self, rays):
        """
        Intersects a bundle of ray with the objects in the scene.
        Returns a tuple of hit information - hit_distances, hit_normals, hit_ids.
        """
        hit_ids = np.array([-1])
        hit_distances = np.array([np.inf])
        hit_normals = np.array([np.inf, np.inf, np.inf])

        for i,obj in enumerate(self.geometries):
            distances,normals = obj.intersect(rays)
            ### all the distances ts should be non negative
            distances = np.where(distances < 0,np.inf,distances)
            
            mask = (distances < hit_distances)
            hit_distances = np.where(mask,distances,hit_distances)
            hit_normals = np.where(mask[...,np.newaxis],normals,hit_normals)
            hit_ids = np.where(mask,i,hit_ids)
            
            #plt.matshow(np.abs(hit_normals.reshape((self.h, self.w, 3))))
            #plt.title("Normals")
            #plt.show()

        return hit_distances, hit_normals, hit_ids