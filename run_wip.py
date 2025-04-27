import numpy as np
import sys
import os
import argparse # Import argparse

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
    from primitives import Scene, Sphere, Mesh, Camera
    from renderer import RayTracer
    # Import all sampler classes (enum constants removed)
    from sampling import (
        Sampling, # Base class (optional import)
        LightSampling,
        CosineSampling,
        BRDFSampling,
        MISampling,
        UniformSphereSampling,
        # Import constants needed for UniformSphereSampling init
        #UNIFORM_SPH_SAMPLING,
        #UNIFORM_HEMI_SPH_SAMPLING
    )
    # from utils import * # If needed
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure the necessary modules (primitives.py, camera.py, raytracer.py, sampling.py, utils.py) are in the Python path.")
    sys.exit(1)

# --- Sampler Registry ---
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
    "cosine": {
        "class": CosineSampling,
        "init_kwargs": {} # Assumes constructor takes no specific args beyond **kwargs
    },
    "light": {
        "class": LightSampling,
        "init_kwargs": {} # Assumes constructor takes no specific args beyond **kwargs
    },
    "brdf": {
        "class": BRDFSampling,
        "init_kwargs": {} # Assumes constructor takes no specific args beyond **kwargs
    },
    "mis": {
        "class": MISampling,
        "init_kwargs": {} # Assumes constructor takes no specific args beyond **kwargs
    }
    # Add other samplers here if needed
}

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Python OOP Raytracer")
    parser.add_argument(
        "--sampler",
        type=str,
        required=True,
        choices=SAMPLER_REGISTRY.keys(),
        help="Select the sampling strategy to use."
    )
    parser.add_argument(
        "--spp",
        type=int,
        default=20, # Default samples per pixel
        help="Number of samples per pixel for progressive rendering."
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default="./", # Default to current directory
        help="Directory containing the mesh (.npz) files."
    )
    # Add other command-line arguments if needed (e.g., output filename, resolution)
    # parser.add_argument("--width", type=int, default=512, help="Image width")
    # parser.add_argument("--height", type=int, default=512, help="Image height")
    # parser.add_argument("--output", type=str, default=None, help="Output image filename.")

    return parser.parse_args()

if __name__ == '__main__':
    # 0. Parse Command Line Arguments
    args = parse_arguments()
    print(f"Selected sampler: {args.sampler}")
    print(f"Samples per pixel: {args.spp}")
    print(f"Mesh directory: {os.path.abspath(args.mesh_dir)}")

    print("Setting up scene...")
    # 1. Create the Scene
    scene = Scene()

    # 2. Define Camera Configuration
    # Potentially override W/H from args if added to parser
    cam_config = {
        'look_at': np.array([278, 273, 0], dtype=np.float64),
        'up': np.array([0, 1, 0], dtype=np.float64),
        'cam': np.array([278, 273, -800], dtype=np.float64),
        'fov': 39.0,
        'W': 512, # Or args.width
        'H': 512, # Or args.height
        'jitter': False
    }
    camera = Camera(**cam_config)

    # 3. Add Geometries to the Scene using the specified mesh directory
    mesh_path_prefix = args.mesh_dir
    try:
        geometries = [
            Sphere(r=60, c=np.array([278, 450, 250]), Le=1.25 * np.array([15.6, 15.6, 15.6])),
            Mesh(os.path.join(mesh_path_prefix, "cbox_floor.npz"), brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh(os.path.join(mesh_path_prefix, "cbox_ceiling.npz"), brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh(os.path.join(mesh_path_prefix, "cbox_back.npz"), brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh(os.path.join(mesh_path_prefix, "cbox_greenwall.npz"), brdf_params=np.array([0.16, 0.76, 0.16, 1])),
            Mesh(os.path.join(mesh_path_prefix, "cbox_redwall.npz"), brdf_params=np.array([0.76, 0.16, 0.16, 1])),
            Mesh(os.path.join(mesh_path_prefix, "cbox_smallbox.npz"), brdf_params=np.array([0.76, 0.76, 0.76, 1])),
            Mesh(os.path.join(mesh_path_prefix, "cbox_largebox.npz"), brdf_params=np.array([0.76, 0.76, 0.76, 1]))
        ]
        scene.add_geometries(geometries)
    except FileNotFoundError as e:
        print(f"Error loading mesh file: {e}")
        print(f"Please ensure mesh files (.npz) are in the directory: {os.path.abspath(mesh_path_prefix)}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during geometry loading: {e}")
        sys.exit(1)

    print(f"Scene setup complete. Geometries: {len(scene.geometries)}, Lights: {len(scene.lights)}")
    if not scene.lights:
         print("Warning: No light sources found in the scene!")
         # Depending on the sampler, this might be an error
         if args.sampler in ["light", "mis"]:
             print(f"Error: Sampler '{args.sampler}' requires at least one light source.")
             sys.exit(1)


    # 4. Instantiate Selected Sampling Strategy using the Registry
    try:
        sampler_info = SAMPLER_REGISTRY[args.sampler]
        sampler_class = sampler_info["class"]
        sampler_kwargs = sampler_info["init_kwargs"] # Get specific kwargs needed for init

        # Instantiate the sampler using its specific required kwargs
        # Assumes sampler constructors now only require specific init args (like sampling_type)
        # and accept **kwargs for the base class.
        sampler = sampler_class(**sampler_kwargs)

    except KeyError:
        # This should not happen due to argparse choices, but good practice
        print(f"Error: Unknown sampler '{args.sampler}' specified.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during sampler instantiation: {e}")
        print("Ensure the sampler constructors in sampling.py have been updated correctly.")
        sys.exit(1)

    print(f"Using sampler: {sampler.__class__.__name__}")

    # 5. Instantiate the RayTracer
    renderer = RayTracer(scene=scene, sampler=sampler, camera=camera)
    print("RayTracer initialized.")

    # 6. Start Rendering
    print(f"Starting progressive rendering with {args.spp} spp...")
    # Pass spp from command line arguments
    renderer.progressive_render_display(total_spp=args.spp, num_bounces=1) # num_bounces=1 for direct light

    print("Script finished.")
