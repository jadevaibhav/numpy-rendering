import numpy as np
import sys
import os
import argparse
import yaml 

# --- Add project root to Python path ---
# This helps Python find your modules if run.py is not in the same directory
# as primitives.py, sampling.py, etc. Uncomment and adjust if needed.
# Example: If run.py is in root, and modules are in 'src/':
# project_root = os.path.abspath(os.path.dirname(__file__))
# src_path = os.path.join(project_root, 'src')
# sys.path.insert(0, src_path)
# Or ensure your PYTHONPATH environment variable includes the modules directory.
# print("Python Path:", sys.path) # Uncomment to debug path issues

# --- Import necessary classes ---
try:
    # Primitives
    from primitives import Scene, Sphere, Mesh, Camera 
    from renderer import RayTracer
    from sampling import (
        Sampling, 
        LightSampling,
        CosineSampling,
        BRDFSampling,
        MISampling,
        UniformSphereSampling,
        # # Import constants needed for UniformSphereSampling init
        # UNIFORM_SPH_SAMPLING, # Ensure these are defined in sampling.py
        # UNIFORM_HEMI_SPH_SAMPLING
    )
   
    from utils import *
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the necessary modules (primitives.py, camera.py, raytracer.py, sampling.py, utils.py) are in the Python path.")
    sys.exit(1)

# --- Sampler Registry (remains the same) ---
# Maps string names to sampler classes and any required init kwargs
SAMPLER_REGISTRY = {
    # "uniform_sphere": {
    #     "class": UniformSphereSampling,
    #     # Pass the required 'sampling_type' for this specific sampler
    #     "init_kwargs": {"sampling_type": UNIFORM_SPH_SAMPLING}
    # },
    # "uniform_hemisphere": {
    #     "class": UniformSphereSampling,
    #     # Pass the required 'sampling_type' for this specific sampler
    #     "init_kwargs": {"sampling_type": UNIFORM_HEMI_SPH_SAMPLING}
    #  },
    "cosine": {"class": CosineSampling, "init_kwargs": {}},
    "light": {"class": LightSampling, "init_kwargs": {}},
    "brdf": {"class": BRDFSampling, "init_kwargs": {}},
    "mis": {"class": MISampling, "init_kwargs": {}}
}

# --- Geometry Registry ---
# Maps geometry type strings from YAML to Python classes
GEOMETRY_REGISTRY = {
    "Sphere": Sphere,
    "Mesh": Mesh
    # Add other geometry types here (e.g., "Plane": Plane)
}

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Python OOP Raytracer with YAML Config")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file."
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default=None, # Default is None, will use config file value
        choices=list(SAMPLER_REGISTRY.keys()) + [None], # Allow None
        help="Override the sampling strategy specified in the config file."
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=None, # Default is None, will use config file value
        help="Override the samples per pixel specified in the config file."
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="./", # Default to current directory
        help="Directory containing the mesh (.npz) files referenced in the config."
    )
    # parser.add_argument("--output", type=str, default=None, help="Override output filename.")

    return parser.parse_args()

def load_config(config_path):
    """Loads the YAML configuration file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded successfully from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file {config_path}: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred loading config: {e}")
        sys.exit(1)

def create_camera_from_config(config):
    """Creates a Camera object from the loaded configuration."""
    cam_config = config.get('camera')
    if not cam_config:
        print("Error: 'camera' section missing in configuration file.")
        sys.exit(1)

    try:
        # Convert lists from YAML to numpy arrays
        look_at = np.array(cam_config['look_at'], dtype=np.float64)
        up = np.array(cam_config['up'], dtype=np.float64)
        cam_pos = np.array(cam_config['position'], dtype=np.float64)

        # Ensure Camera class is imported correctly
        camera = Camera(
            look_at=look_at,
            up=up,
            cam=cam_pos,
            fov=float(cam_config['fov']),
            W=int(cam_config['width']),
            H=int(cam_config['height']),
            jitter=bool(cam_config.get('jitter', False)) # Optional jitter
        )
        print("Camera created successfully.")
        return camera
    except KeyError as e:
        print(f"Error: Missing key in 'camera' configuration: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error creating camera from config: {e}")
        sys.exit(1)

def create_scene_from_config(config, mesh_dir):
    """Creates a Scene object and populates it from the loaded configuration."""
    scene_config = config.get('scene')
    if not scene_config:
        print("Error: 'scene' section missing in configuration file.")
        sys.exit(1)

    geometries_config = scene_config.get('geometries', [])
    if not geometries_config:
        print("Warning: No geometries defined in the 'scene' section.")

    scene = Scene()
    print(f"Processing {len(geometries_config)} geometries from config...")

    for i, geom_data in enumerate(geometries_config):
        geom_type_str = geom_data.get('type')
        params = geom_data.get('parameters', {})

        if not geom_type_str:
            print(f"Warning: Geometry entry {i+1} missing 'type'. Skipping.")
            continue

        geom_class = GEOMETRY_REGISTRY.get(geom_type_str)
        if not geom_class:
            print(f"Warning: Unknown geometry type '{geom_type_str}' in entry {i+1}. Skipping.")
            continue

        try:
            # Prepare arguments for the geometry class constructor
            geom_args = {}
            if geom_class == Sphere:
                geom_args['r'] = float(params['radius'])
                geom_args['c'] = np.array(params['center'], dtype=np.float64)
                # Optional emission (Le), default to black
                if 'emission' in params:
                    geom_args['Le'] = np.array(params['emission'], dtype=np.float64)
                # Optional BRDF params, default might be handled in Sphere class
                if 'brdf' in params:
                     geom_args['brdf_params'] = np.array(params['brdf'], dtype=np.float64)

            elif geom_class == Mesh:
                mesh_filename = params['filename']
                # Ensure mesh_dir is an absolute path or relative to script execution dir
                full_mesh_path = os.path.abspath(os.path.join(mesh_dir, mesh_filename))
                geom_args['filename'] = full_mesh_path
                geom_args['brdf_params'] = np.array(params['brdf'], dtype=np.float64)
                 # Optional emission (Le), default to black
                if 'emission' in params:
                    geom_args['Le'] = np.array(params['emission'], dtype=np.float64)

            # Add logic for other geometry types here...

            # Instantiate the geometry
            geometry_instance = geom_class(**geom_args)
            scene.add_geometries([geometry_instance]) # add_geometries expects a list
            print(f"  Added {geom_type_str}: {params.get('filename', params.get('center'))}")

        except KeyError as e:
            print(f"Warning: Missing parameter '{e}' for geometry type '{geom_type_str}' in entry {i+1}. Skipping.")
            continue
        except FileNotFoundError:
             print(f"Warning: Mesh file not found for '{geom_type_str}' in entry {i+1}: {full_mesh_path}. Skipping.")
             continue
        except Exception as e:
            print(f"Warning: Error creating geometry type '{geom_type_str}' in entry {i+1}: {e}. Skipping.")
            continue

    print(f"Scene setup complete. Geometries loaded: {len(scene.geometries)}, Lights found: {len(scene.lights)}")

    # Check for light source requirement after loading all geometries
    sampler_type_in_config = config.get('sampler', {}).get('type', 'mis') # Default to mis if not specified
    if not scene.lights and sampler_type_in_config in ["light", "mis"]:
        print(f"Error: Configured sampler '{sampler_type_in_config}' requires at least one light source with non-zero emission, but none found in scene config.")
        sys.exit(1)

    return scene


def get_sampler_from_config(config, cli_override=None):
    """Gets the sampler instance based on config or CLI override."""
    sampler_config = config.get('sampler', {})
    # Use CLI override if provided, otherwise use config, default to 'mis' if totally missing
    sampler_type = cli_override if cli_override is not None else sampler_config.get('type', 'mis')

    if sampler_type not in SAMPLER_REGISTRY:
        print(f"Error: Unknown sampler type '{sampler_type}'. Available: {list(SAMPLER_REGISTRY.keys())}")
        sys.exit(1)

    try:
        sampler_info = SAMPLER_REGISTRY[sampler_type]
        sampler_class = sampler_info["class"]
        # Combine specific init_kwargs from registry with potential future kwargs from config
        sampler_kwargs = sampler_info["init_kwargs"].copy()
        # Add any sampler-specific options from config['sampler'] here if needed
        # sampler_kwargs.update(sampler_config.get('options', {}))

        sampler = sampler_class(**sampler_kwargs)
        print(f"Using sampler: {sampler.__class__.__name__} (Type: {sampler_type})")
        return sampler
    except Exception as e:
        print(f"An unexpected error occurred during sampler instantiation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    # 0. Parse Command Line Arguments
    args = parse_arguments()

    # 1. Load Configuration from YAML
    config = load_config(args.config)

    # --- Get effective parameters (CLI overrides config) ---
    effective_spp = args.spp if args.spp is not None else config.get('renderer', {}).get('spp', 20) # Default 20 if missing
    # Determine sampler type: CLI override > config file > default ('mis')
    effective_sampler_type_cli = args.sampler # This is None if not provided
    # effective_output = args.output if args.output is not None else config.get('renderer', {}).get('output_filename', None)
    num_bounces = config.get('renderer', {}).get('num_bounces', 1)

    print(f"Effective Samples per pixel: {effective_spp}")
    print(f"Mesh directory: {os.path.abspath(args.mesh_dir)}")

    # 2. Create Camera from Config
    camera = create_camera_from_config(config)

    # 3. Create Scene from Config
    # Pass the absolute path to mesh_dir for robust file finding
    scene = create_scene_from_config(config, os.path.abspath(args.mesh_dir))

    # 4. Instantiate Selected Sampling Strategy (using config or override)
    # Pass the CLI override value (which might be None)
    sampler = get_sampler_from_config(config, cli_override=effective_sampler_type_cli)

    # 5. Instantiate the RayTracer
    # Pass camera and sampler directly
    renderer = RayTracer(scene=scene, sampler=sampler, camera=camera)
    print("RayTracer initialized.")

    # 6. Start Rendering
    print(f"Starting progressive rendering with {effective_spp} spp...")
    # Pass effective spp and num_bounces from config/defaults
    renderer.progressive_render_display(total_spp=effective_spp, num_bounces=num_bounces)
    # Optionally save based on effective_output here

    print("Script finished.")
