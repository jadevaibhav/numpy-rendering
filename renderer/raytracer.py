import numpy as np
import matplotlib.pyplot as plt
from sampling import Sampling 
from utils import *
from primitives import Rays, Scene, Mesh, Camera 

class RayTracer(object):
    """
    RayTracer class responsible for rendering the scene.

    Attributes:
        scene (Scene): The scene object containing geometries and lights.
        sampler (Sampling): The sampling strategy object.
        camera (Camera): The camera object defining the viewpoint.
        H (int): Height of the output image.
        W (int): Width of the output image.
    """

    def __init__(self, scene: Scene, sampler: Sampling, camera: Camera) -> None:
         
        self.scene = scene
        self.sampler = sampler
        self.camera = camera
        # Get image dimensions from the camera
        self.H = camera.H
        self.W = camera.W

    def generate_cam_rays(self) -> Rays:
        """
        Generates primary rays originating from the camera.

        Returns:
            Rays: A Rays object containing the origins and directions
                  of the primary rays.
        """
        # Delegate ray generation to the camera object
        # If jitter, we generate ray for each render, otherwise pre-compute
        if self.camera.jitter:
            return self.camera.generate_cam_rays()
        else: 
            return self.camera.cam_rays

    def intersect_with_scene(self, rays: Rays) -> tuple:
        """
        Intersects a bundle of rays with the scene.

        Args:
            rays: The Rays object to intersect with the scene.

        Returns:
            tuple: A tuple containing:
                - hit_points (np.ndarray): Points of intersection.
                - normals (np.ndarray): Surface normals at the hit points.
                - brdf_params (np.ndarray): BRDF parameters of the hit surfaces.
                - l_e (np.ndarray): Emitted light (Le) from the hit surfaces.
                - hit_ids (np.ndarray): Indices of the hit geometries (-1 for no hit).
        """
        # Intersect rays with the scene geometries
        dist, normals, ids = self.scene.intersect(rays)

        # Calculate hit points using the distances
        hit_points = rays(dist) 

        # Replace infinite normals (no hit) with zero vectors
        normals = np.where(np.isinf(normals), np.array([0.0, 0.0, 0.0]), normals)

        # --- Retrieve Material Properties ---
        # Create arrays of BRDF parameters and Le for all geometries
        # Add default values at the end for the case where a ray hits nothing (id = -1)
        all_brdf_params = np.array([obj.brdf_params for obj in self.scene.geometries])
        default_brdf = np.array([[0.0, 0.0, 0.0, 1.0]]) # Default BRDF for no hit
        brdf_params_lookup = np.concatenate((all_brdf_params, default_brdf), axis=0)

        all_l_e = np.array([obj.Le for obj in self.scene.geometries])
        default_l_e = np.array([[0.0, 0.0, 0.0]]) # Default Le for no hit
        l_e_lookup = np.concatenate((all_l_e, default_l_e), axis=0)

        # Get the parameters corresponding to the hit objects using the hit ids
        brdf_params = brdf_params_lookup[ids]
        l_e = l_e_lookup[ids]

        # Ensure Le is zero for non-emissive surfaces that were hit (redundant if Le is correctly set in primitives)
        # l_e = np.where(ids[:, np.newaxis] != -1, l_e, np.array([0.0, 0.0, 0.0])) # Keep Le only for actual hits

        return hit_points, normals, brdf_params, l_e, ids 

    def render(self, rays: Rays) -> np.ndarray:
        """
        Performs one pass of rendering for the given rays (e.g., direct illumination).

        Args:
            rays: The primary rays from the camera.

        Returns:
            np.ndarray: The calculated radiance for each ray, reshaped to (H, W, 3).
        """
        # 1. Find first intersection (eye rays)
        hit_points, normals, brdf_params, L_e, hit_ids = self.intersect_with_scene(rays)

        # Initialize the output image (radiance)
        L = np.zeros_like(normals, dtype=np.float64) # Shape (N, 3)
        L += L_e

        # --- Direct Illumination ---
        # Only calculate direct illumination for rays that actually hit something
        hit_mask = (hit_ids != -1)
        if not np.any(hit_mask): # If no rays hit anything
             return L.reshape((self.H, self.W, 3))

        # Filter data for points that were hit
        valid_hit_points = hit_points[hit_mask]
        valid_normals = normals[hit_mask]
        valid_brdf_params = brdf_params[hit_mask]
        valid_incoming_dirs = rays.Ds[hit_mask] # Incoming direction to the hit point

        # Initialize sampler with parameters of the valid hit points
        self.sampler.set_initial_params(valid_hit_points, valid_normals, valid_brdf_params, valid_incoming_dirs)

        L_direct = np.zeros_like(valid_normals, dtype=np.float64)

        # Loop through each light source for direct illumination calculation
        for light in self.scene.lights:
            self.sampler.set_light(light) # Configure sampler for the current light

            # a. Sample directions (generate shadow rays)
            shadow_rays, prob = self.sampler.shadow_rays() # Sampler generates N_valid rays

            # b. Check visibility: Intersect shadow rays with the scene
            # Important: Need to handle self-intersection (offset origin slightly)
            # The shadow_ray origin is already offset in the Sampling base class
            shadow_hit_dist, _, shadow_hit_ids = self.scene.intersect(shadow_rays)

            # Determine which shadow rays actually hit the *current* light source
            # Find the index of the current light in the scene's geometry list
            try:
                light_geom_index = self.scene.geometries.index(light)
            except ValueError:
                continue # Should not happen if lights are added correctly

            # light_e_values will be the light's Le if the shadow ray hits it, else [0,0,0]
            light_e_values = np.zeros_like(L_direct)
            # Check if the shadow ray hit *anything* before potentially hitting the light
            # and if the thing hit was the light source itself.
            # A simple check is if shadow_hit_ids == light_geom_index
            # A more robust check might involve comparing distances if multiple objects overlap
            hit_current_light_mask = (shadow_hit_ids == light_geom_index)
            light_e_values[hit_current_light_mask] = light.Le

            # c. Calculate illumination contribution using the sampler
            # Pass only the emission from the *current* light source
            L_direct += self.sampler.illumination(light_e_values, shadow_rays.Ds, prob)

        # Add direct illumination contribution back to the main radiance array
        # Use the hit_mask to place the results correctly
        L[hit_mask] += L_direct

        # Reshape the final radiance to image dimensions
        return L.reshape((self.H, self.W, 3))

    def progressive_render_display(self, total_spp=20, num_bounces=1, jitter=None):
        """
        Performs progressive rendering and displays the result interactively.

        Args:
            total_spp (int): Total samples per pixel.
            num_bounces (int): Number of bounces for path tracing (currently only direct).
                               (Note: The current render method only does direct illumination).
            jitter (bool, optional): Whether to jitter camera rays. If None, uses camera's default.
        """
        # Use camera's jitter setting if not overridden
        if jitter is None:
            jitter = self.camera.jitter
        else:
            self.camera.jitter = jitter # Update camera's jitter setting if specified here

        # Matplotlib setup for interactive display
        plt.figure()
        plt.ion() # Turn interactive mode on
        plt.axis('off') # Turn off axes
        plt.title(f"Rendering Progress (0/{total_spp} spp)")

        # Initialize accumulated radiance and image display handle
        L_accumulated = np.zeros((self.H, self.W, 3), dtype=np.float64)
        # Display initial black image
        image_data = plt.imshow(np.clip(L_accumulated, 0, 1))
        plt.show()
        plt.pause(0.01) # Allow plot window to appear

        # --- Progressive Rendering Loop ---
        for i in range(total_spp):
            print(f"Rendering sample {i + 1}/{total_spp}...")
            # 1. Generate camera rays for this sample
            # Consider jitter for anti-aliasing if enabled
            primary_rays = self.generate_cam_rays()

            # 2. Render this sample (currently only direct illumination)
            # TODO: Extend `render` for multi-bounce path tracing if needed
            L_sample = self.render(primary_rays)

            # 3. Accumulate radiance
            L_accumulated += L_sample

            # 4. Update display with the average radiance
            L_display = L_accumulated / (i + 1)
            # Apply gamma correction and clipping for display
            L_display_gamma = np.clip(L_display**(1/2.2), 0, 1)

            image_data.set_data(L_display_gamma)
            plt.title(f"Rendering Progress ({i + 1}/{total_spp} spp)")
            plt.draw() # Redraw the current figure
            plt.pause(0.001) # Pause allows the plot to update

        print("Rendering finished.")
        plt.ioff() # Turn interactive mode off

        # Save the final image (average radiance, gamma corrected)
        final_image = np.clip(L_accumulated / total_spp, 0, 1)
        final_image_gamma = final_image**(1/2.2)

        try:
            # Construct filename based on parameters
            sampler_name = self.sampler.__class__.__name__ # Get sampler class name
            light_radius = self.scene.lights[0].r if self.scene.lights and hasattr(self.scene.lights[0], 'r') else 'unknown'
            filename = f"render-{sampler_name}-bounces-{num_bounces}-lightR-{light_radius}-spp-{total_spp}.png"
            plt.imsave(filename, final_image_gamma)
            print(f"Image saved as {filename}")
        except Exception as e:
            print(f"Error saving image: {e}")
            # Fallback filename
            plt.imsave("render_final.png", final_image_gamma)
            print("Image saved as render_final.png")


        plt.title(f"Final Render ({total_spp} spp)")
        plt.imshow(final_image_gamma) # Show final gamma-corrected image
        plt.show(block=True) # Keep the window open until closed manually
