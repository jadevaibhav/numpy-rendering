from sampling import *
from primitives import Scene, Rays, Geometry, Sphere, Mesh
from renderer import RayTracer
import numpy as np

if __name__ == '__main__':
    scene = Scene()
    sampler = LightSampling()
    scene.add_geometries([
            Sphere(60, np.array([213 + 65, 450, 227 + 105 / 2 - 100]),
                   Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh("cbox_floor.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_ceiling.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_back.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_greenwall.npz",
                 brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh("cbox_redwall.npz",
                 brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh("cbox_smallbox.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh("cbox_largebox.npz",
                 brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ])
    
    renderer = RayTracer(scene=scene,sampler=sampler,camconfig=
                        { 'H': 512,'W': 512,'fov': 39,
                        'look': np.array([278, 273, -769], dtype=np.float64),
                        'up': np.array([0, 1, 0], dtype=np.float64),
                        'cam': np.array([278, 273, -770], dtype=np.float64) 
                        })
    renderer.progressive_render_display(jitter=False)
