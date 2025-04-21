import argparse
import yaml
from renderer import RayTracer
from sampling import build_sampler
from primitives import *

def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_camera(cfg):
    return Camera(
        eye=cfg['eye'],
        look_at=cfg['look_at'],
        up=cfg['up'],
        fov=cfg['fov'],
        W=cfg['width'],
        H=cfg['height'], 
        jitter = cfg["jitter"]
    )

def build_scene(cfg):
    objects = []

    for geom in cfg['scene']['geometries']:
        brdf_params = geom.get('brdf_params', [1.0, 1.0, 1.0, 1.0])
        Le = geom.get('Le', [0.0, 0.0, 0.0])

        if geom['type'] == 'Sphere':
            obj = Sphere(
                r=geom['radius'],
                c=geom['center'],
                brdf_params=brdf_params,
                Le=Le
            )
        elif geom['type'] == 'Mesh':
            obj = load_mesh(
                filepath=geom['path'],
                scale=geom.get('scale', 1.0),
                translation=geom.get('translation', [0, 0, 0]),
                brdf_params=brdf_params,
                Le=Le
            )
        else:
            raise ValueError(f"Unsupported geometry type: {geom['type']}")
        
        objects.append(obj)

    return Scene(objects)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    camera = build_camera(cfg['camera'])
    sampler = build_sampler(cfg['sampler'])
    scene = build_scene(cfg)

    raytracer = RayTracer(
        scene=scene,
        camera=camera,
        sampler=sampler,
        max_depth=cfg['render']['max_depth'],
    )

    raytracer.render(
        samples=cfg['render']['samples'],
        output_path=cfg['render']['output_path'],
        progressive_display=cfg['render'].get('progressive_display', False)
    )

if __name__ == '__main__':
    main()
