import numpy as np
from primitives import Rays
from utils import *

# --- Constants for UniformSphereSampling type ---
# These are kept as they are specific init arguments for UniformSphereSampling
UNIFORM_SPH_SAMPLING = "uniform_sphere"
UNIFORM_HEMI_SPH_SAMPLING = "uniform_hemisphere"


class Sampling(object):
    """
    Base class for different importance sampling strategies.

    Attributes:
        shadow_ray_o_offset (float): Small offset for shadow rays to avoid self-intersection.
        hit_points (np.ndarray | None): Intersection points on surfaces. Set by set_initial_params.
        normals (np.ndarray | None): Surface normals at hit points. Set by set_initial_params.
        brdf_params (np.ndarray | None): BRDF parameters of surfaces. Set by set_initial_params.
        rays_w (np.ndarray | None): Incoming ray directions (view directions). Set by set_initial_params.
        light (Geometry | None): The light source being sampled (if applicable). Set by set_light.
    """
    shadow_ray_o_offset = 3e-6 # Offset to avoid self-intersection

    def __init__(self, **kwargs) -> None:
        """
        Initializes the base sampler. Runtime data is set later.
        Accepts keyword arguments for potential future configurations.
        """
        # Initialize runtime-dependent attributes to None
        self.hit_points = None
        self.normals = None
        self.brdf_params = None
        self.rays_w = None
        self.light = None
        # Store any other configuration kwargs if needed
        self._config_kwargs = kwargs
        # sampling_type might be set by subclasses or passed via kwargs
        self.sampling_type = kwargs.get('sampling_type', None)


    def set_initial_params(self, hit_points: np.ndarray, normals: np.ndarray, brdf_params: np.ndarray, rays_w: np.ndarray):
        """
        Sets the runtime parameters required for sampling calculations.
        Called by the RayTracer before sampling for a set of hits.

        Args:
            hit_points: Intersection points.
            normals: Surface normals at hit points.
            brdf_params: BRDF parameters of hit surfaces.
            rays_w: Incoming ray directions towards the hit points.
        """
        self.rays_w = rays_w
        self.hit_points = hit_points
        self.normals = normals
        self.brdf_params = brdf_params
        # Reset any state that depends on these params if necessary
        self._on_params_set()

    def _on_params_set(self):
        """Placeholder method for subclasses to react to parameters being set."""
        pass # Overridden by MISampling, for example

    def set_light(self, light):
        """
        Sets the specific light source for samplers that target lights.

        Args:
            light: A geometry object representing the light source.
        """
        self.light = light
        # Reset any state that depends on the light if necessary
        self._on_light_set()

    def _on_light_set(self):
         """Placeholder method for subclasses to react to light being set."""
         pass # Overridden by MISampling, for example

    def shadow_rays(self) -> tuple:
        """
        Generates shadow rays based on the sampling strategy.

        Returns:
            tuple: (Rays object containing shadow ray origins and directions,
                    probability density of sampling those directions)
        Raises:
            ValueError: If initial parameters or light (if needed) are not set.
        """
        if self.hit_points is None or self.normals is None:
            raise ValueError("Initial parameters (hit_points, normals) must be set before generating shadow rays.")

        new_ray_dir, prob = self.sample()
        # Offset origin slightly along the normal to avoid self-intersection
        origins = self.hit_points + self.shadow_ray_o_offset * self.normals
        
        return Rays(origins, new_ray_dir), prob

    def sample(self, mask=None) -> tuple:
        """
        Abstract method to sample new ray directions from the hit points.
        Masking logic needs careful implementation in subclasses if used.

        Args:
            mask (np.ndarray, optional): Boolean mask to select hit points for sampling. Defaults to None (all points).

        Returns:
            tuple: (Sampled directions (np.ndarray), Probability density (np.ndarray or float))

        Raises:
            NotImplementedError: If called on the base class.
        """
        raise NotImplementedError("Sample method must be implemented by subclasses.")

    def eval_prob_dist(self, dirs: np.ndarray, mask=None) -> np.ndarray:
        """
        Abstract method to evaluate the probability density of given directions
        under this sampler's distribution.

        Args:
            dirs: The directions to evaluate.
            mask (np.ndarray, optional): Boolean mask to select corresponding hit points. Defaults to None.

        Returns:
            np.ndarray: Probability densities for the given directions.

        Raises:
            NotImplementedError: If called on the base class.
        """
        raise NotImplementedError("eval_prob_dist method must be implemented by subclasses.")

    def illumination(self, light_e: np.ndarray, dirs: np.ndarray, prob: np.ndarray, mask=None) -> np.ndarray:
        """
        Calculates the illumination contribution using the BRDF, sampled directions,
        light energy, and probability density.

        Args:
            light_e: Incoming light energy (Le * visibility) reaching the hit points
                     from the sampled directions.
            dirs: The sampled directions.
            prob: The probability density of sampling those directions.
            mask (np.ndarray, optional): Boolean mask to select relevant parameters. Defaults to None.

        Returns:
            np.ndarray: The calculated illumination (radiance) contribution.
        """
        # Apply mask if provided - Note: Masking needs careful handling if sizes change
        # If mask is None, create a mask that selects all elements
        effective_mask = slice(None) if mask is None else mask

        if self.brdf_params is None or self.rays_w is None or self.normals is None:
             raise ValueError("Initial parameters must be set before calculating illumination.")

        # Ensure prob is broadcastable or has the correct shape
        if isinstance(prob, (float, int)):
             prob = np.full((dirs[effective_mask].shape[0], 1), prob, dtype=np.float64)
        elif prob.ndim == 1:
             prob = prob[:, np.newaxis]

        # Avoid division by zero or very small probabilities
        prob = np.maximum(prob, 1e-9) # Add epsilon to avoid division by zero

        # Get relevant parameters using the mask
        current_brdf_params = self.brdf_params[effective_mask]
        current_rays_w = self.rays_w[effective_mask]
        current_normals = self.normals[effective_mask]
        current_dirs = dirs[effective_mask]

        alpha = current_brdf_params[:, -1:] # Keep dimension for broadcasting
        rho_d = current_brdf_params[:, :3] # Diffuse reflectance

        reflected_w = reflect_along_normal(current_rays_w, current_normals)
        specular_dot = np.sum(reflected_w * current_dirs, axis=-1, keepdims=True)
        # Clamp specular_dot to avoid issues with pow for negative bases
        specular_dot = np.maximum(0.0, specular_dot)

        # Phong BRDF calculation (simplified - check original formula if needed)
        # Using alpha directly as the exponent here.
        # Note: Original code used alpha+1 / 2pi * spec_dot**alpha for specular part
        # and rho_d/pi for diffuse. Let's stick to that.
        brdf_diffuse = rho_d / np.pi
        brdf_specular = rho_d * (alpha + 1) / (2 * np.pi) * (specular_dot ** alpha)

        # Choose BRDF based on alpha (alpha == 1 means purely diffuse in original code?)
        # Assuming alpha > 1 means Phong, alpha == 1 means Lambertian. Needs clarification based on BRDF definition.
        # Let's assume alpha == 1 implies diffuse, alpha > 1 implies Phong (mix?)
        # A common interpretation is alpha controls shininess, rho_d controls color/intensity.
        # If alpha == 1 was meant as purely diffuse:
        is_diffuse = np.isclose(alpha, 1.0)
        brdf = np.where(is_diffuse, brdf_diffuse, brdf_specular) # Check if this mix is correct

        # Cosine term (Lambert's law part)
        cos_theta = np.sum(current_normals * current_dirs, axis=-1, keepdims=True)
        cos_theta = np.maximum(0.0, cos_theta) # Clamp to non-negative

        # Final illumination calculation: L = Le * BRDF * cos(theta) / pdf
        L_ill = light_e[effective_mask] * brdf * cos_theta / prob

        # Handle cases where prob was zero (or near zero) - resulting L_ill might be inf/nan
        L_ill = np.nan_to_num(L_ill, nan=0.0, posinf=0.0, neginf=0.0)

        # If a mask was used, we need to return an array matching the original size
        if mask is not None:
            full_L_ill = np.zeros_like(light_e)
            full_L_ill[effective_mask] = L_ill
            return full_L_ill
        else:
            return L_ill


class UniformSphereSampling(Sampling):
    """
    Implements uniform spherical or hemi-spherical sampling.
    The type (sphere or hemisphere) is determined during initialization.
    """
    def __init__(self, sampling_type: str, **kwargs) -> None:
        """
        Initializes the uniform sampler.

        Args:
            sampling_type (str): Must be UNIFORM_SPH_SAMPLING or UNIFORM_HEMI_SPH_SAMPLING.
            **kwargs: Additional keyword arguments for the base class.
        """
        if sampling_type not in [UNIFORM_SPH_SAMPLING, UNIFORM_HEMI_SPH_SAMPLING]:
             raise ValueError(f"Invalid sampling_type '{sampling_type}' for UniformSphereSampling")
        super().__init__(sampling_type=sampling_type, **kwargs)

    def eval_prob_dist(self, dirs: np.ndarray, mask=None) -> float:
        """Evaluates the constant probability density."""
        if self.sampling_type == UNIFORM_SPH_SAMPLING:
            # PDF for uniform sampling over the entire sphere surface
            return 1.0 / (4.0 * np.pi)
        else: # UNIFORM_HEMI_SPH_SAMPLING
            return 1.0 / (2.0 * np.pi)

    def sample(self, mask=None) -> tuple:
        """Samples directions uniformly over a sphere or hemisphere."""
        if self.hit_points is None:
             raise ValueError("Initial parameters must be set before sampling.")

        num_samples = self.hit_points.shape[0] if mask is None else np.sum(mask)
        if num_samples == 0:
             return np.empty((0, 3)), np.empty((0, 1))

        # Generate random points uniformly on a unit sphere
        rv1 = np.random.rand(num_samples) # u1 = cos(theta) for sphere, u1 = z for hemisphere mapping? Let's use standard sphere point picking
        rv2 = np.random.rand(num_samples) # u2 = phi / (2*pi)

        z = 1.0 - 2.0 * rv1 # z coordinate uniformly in [-1, 1]
        r = np.sqrt(np.maximum(0.0, 1.0 - z*z)) # Radius in xy plane
        phi = 2.0 * np.pi * rv2 # Azimuthal angle

        w_x = r * np.cos(phi)
        w_y = r * np.sin(phi)
        w_z = z
        w = np.stack([w_x, w_y, w_z], axis=-1)
        # w is already normalized due to construction

        prob = 1.0 / (4.0 * np.pi) # PDF for sphere sampling

        if self.sampling_type == UNIFORM_HEMI_SPH_SAMPLING:
            # If hemisphere sampling, align with normals
            if self.normals is None:
                 raise ValueError("Normals must be set for hemisphere sampling.")

            current_normals = self.normals if mask is None else self.normals[mask]
            if current_normals.shape[0] != num_samples:
                 raise ValueError("Mask and normals dimensions mismatch.")

            # Ensure samples are in the hemisphere defined by the normal
            dot_product = np.sum(current_normals * w, axis=-1, keepdims=True)
            # Flip vectors that are in the wrong hemisphere
            w = np.where(dot_product < 0, -w, w)
            prob = 1.0 / (2.0 * np.pi) # PDF for hemisphere sampling

        # If mask was used, create full-size arrays
        if mask is not None:
             full_w = np.zeros_like(self.hit_points)
             full_prob = np.zeros((self.hit_points.shape[0], 1))
             full_w[mask] = w
             full_prob[mask] = prob
             return full_w, full_prob
        else:
             return w, prob


class CosineSampling(Sampling):
    """Implements cosine-weighted hemisphere sampling."""
    def __init__(self, **kwargs) -> None:
        super().__init__(sampling_type="cosine", **kwargs) # Set type explicitly

    def eval_prob_dist(self, dirs: np.ndarray, mask=None) -> np.ndarray:
        """Evaluates the cosine-weighted probability density: max(0, dot(N, D)) / pi."""
        if self.normals is None:
            raise ValueError("Normals must be set to evaluate cosine probability.")

        current_normals = self.normals if mask is None else self.normals[mask]
        current_dirs = dirs if mask is None else dirs[mask]

        dot_product = np.sum(current_normals * current_dirs, axis=-1, keepdims=True)
        prob = np.maximum(0.0, dot_product) / np.pi

        if mask is not None:
            full_prob = np.zeros((dirs.shape[0], 1))
            full_prob[mask] = prob
            return full_prob
        else:
            return prob

    def sample(self, mask=None) -> tuple:
        """Samples directions according to a cosine distribution around the normal."""
        if self.hit_points is None or self.normals is None:
             raise ValueError("Initial parameters must be set before sampling.")

        num_samples = self.hit_points.shape[0] if mask is None else np.sum(mask)
        if num_samples == 0:
             return np.empty((0, 3)), np.empty((0, 1))

        # Malley's method: uniform sampling on a disk, then project onto hemisphere
        rv1 = np.random.rand(num_samples)
        rv2 = np.random.rand(num_samples)

        # Sample point on unit disk
        r = np.sqrt(rv1)
        phi = 2.0 * np.pi * rv2

        # Local coordinates (x, y, z) where z is up (aligned with normal initially)
        x_local = r * np.cos(phi)
        y_local = r * np.sin(phi)
        z_local = np.sqrt(np.maximum(0.0, 1.0 - rv1)) # z = sqrt(1 - r^2)
        w_local = np.stack([x_local, y_local, z_local], axis=-1)

        # Create orthonormal basis around the normal
        current_normals = self.normals if mask is None else self.normals[mask]
        w = rotate_vectors(w_local,current_normals)

        # PDF is cos(theta) / pi = z_local / pi = dot(N, w) / pi
        prob = w_local[:, 2:3] / np.pi # Use z_local directly

        return w, prob

    # Illumination can often be inherited if the base class handles BRDF * cos / pdf correctly
    # However, the original CosineSampling had a simplified illumination. Let's override
    # to match that simplification if needed, otherwise use base.
    # Sticking to base class illumination calculation for consistency for now.
    # def illumination(self, light_e: np.ndarray, dirs: np.ndarray, prob: np.ndarray, mask=None) -> np.ndarray:
    #     # Original simplified: L = Le * BRDF / pi (prob = cos/pi, so BRDF * cos / prob = BRDF * pi)
    #     # This assumes BRDF itself doesn't include the cosine term.
    #     # Base class illumination: L = Le * BRDF * cos / prob
    #     # If prob = cos / pi, then L = Le * BRDF * cos / (cos / pi) = Le * BRDF * pi
    #     # Let's use the base class implementation which is more general.
    #     return super().illumination(light_e, dirs, prob, mask)


class LightSampling(Sampling):
    """
    Implements sampling directions towards a spherical light source,
    uniformly over the solid angle subtended by the light.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(sampling_type="light", **kwargs)

    def _get_solid_angle_and_center_dir(self, current_hit_points):
        """Helper to calculate solid angle and direction to light center."""
        if self.light is None or not hasattr(self.light, 'c') or not hasattr(self.light, 'r'):
            raise ValueError("Spherical light source (with center 'c' and radius 'r') must be set.")

        vec_to_center = self.light.c - current_hit_points
        dist_sq = np.sum(vec_to_center**2, axis=-1, keepdims=True)
        dist = np.sqrt(dist_sq)

        # Ensure distance is not zero
        dist = np.maximum(dist, 1e-9)
        vec_to_center = vec_to_center / dist # Normalize direction to center

        # Check if hit point is inside the light sphere
        # If inside, solid angle is 4pi? Or handle differently? Assume outside.
        # Clamp radius^2 / dist_sq to [0, 1] for sqrt
        cos_theta_max_sq = 1.0 - np.clip(self.light.r**2 / dist_sq, 0.0, 1.0)
        cos_theta_max = np.sqrt(cos_theta_max_sq)

        # Solid angle = 2 * pi * (1 - cos(theta_max))
        solid_angle = 2.0 * np.pi * (1.0 - cos_theta_max)
        # Avoid zero solid angle if point is far away and light is small
        solid_angle = np.maximum(solid_angle, 1e-9)

        return solid_angle, cos_theta_max, vec_to_center

    def eval_prob_dist(self, dirs: np.ndarray, mask=None) -> np.ndarray:
        """Evaluates the probability density (1 / solid_angle) if dir is within cone, else 0."""
        if self.hit_points is None:
            raise ValueError("Initial parameters must be set.")

        current_hit_points = self.hit_points if mask is None else self.hit_points[mask]
        current_dirs = dirs if mask is None else dirs[mask]

        if current_hit_points.shape[0] == 0:
             return np.empty((0, 1))

        solid_angle, cos_theta_max, vec_to_center = self._get_solid_angle_and_center_dir(current_hit_points)

        # Check if the direction is within the cone towards the light
        dot_product = np.sum(current_dirs * vec_to_center, axis=-1, keepdims=True)
        is_within_cone = dot_product >= cos_theta_max

        prob = np.where(is_within_cone, 1.0 / solid_angle, 0.0)

        return prob

    def sample(self, mask=None) -> tuple:
        """Samples directions uniformly within the solid angle subtended by the light."""
        if self.hit_points is None:
             raise ValueError("Initial parameters must be set before sampling.")

        current_hit_points = self.hit_points if mask is None else self.hit_points[mask]
        num_samples = current_hit_points.shape[0]

        if num_samples == 0:
             return np.empty((0, 3)), np.empty((0, 1))

        solid_angle, cos_theta_max, vec_to_center = self._get_solid_angle_and_center_dir(current_hit_points)

        # Sample uniformly within the cone defined by cos_theta_max
        # Sample cos(theta) uniformly in [cos_theta_max, 1]
        rv1 = np.random.rand(num_samples, 1)
        cos_theta = 1.0 - rv1 * (1.0 - cos_theta_max)
        sin_theta = np.sqrt(np.maximum(0.0, 1.0 - cos_theta**2))

        # Sample phi uniformly in [0, 2*pi]
        rv2 = np.random.rand(num_samples, 1)
        phi = 2.0 * np.pi * rv2

        # Local coordinates (relative to vec_to_center being the z-axis)
        x_local = sin_theta * np.cos(phi)
        y_local = sin_theta * np.sin(phi)
        z_local = cos_theta
        w_local = np.concatenate([x_local, y_local, z_local], axis=-1)

        from utils import rotate_vectors 
        # Rotate w_local (aligned with Z) to be aligned with vec_to_center
        
        w = rotate_vectors(w_local,vec_to_center)
        prob = 1.0 / solid_angle # PDF is 1 / solid_angle

        # if mask is not None:
        #      full_w = np.zeros_like(self.hit_points)
        #      full_prob = np.zeros((self.hit_points.shape[0], 1))
        #      full_w[mask] = w
        #      full_prob[mask] = prob
        #      return full_w, full_prob
        # else:
        return w, prob


class BRDFSampling(Sampling):
    """
    Implements sampling based on the Phong BRDF model.
    Handles diffuse (alpha=1) and specular lobes.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(sampling_type="brdf", **kwargs)

    def eval_prob_dist(self, dirs: np.ndarray, mask=None) -> np.ndarray:
        """Evaluates the probability density based on the Phong lobe."""
        if self.brdf_params is None or self.rays_w is None or self.normals is None:
            raise ValueError("Initial parameters must be set.")

        current_brdf = self.brdf_params if mask is None else self.brdf_params[mask]
        current_rays_w = self.rays_w if mask is None else self.rays_w[mask]
        current_normals = self.normals if mask is None else self.normals[mask]
        current_dirs = dirs if mask is None else dirs[mask]

        alpha = current_brdf[:, -1:] # Phong exponent

        # Handle diffuse case (alpha == 1 -> cosine sampling)
        is_diffuse = np.isclose(alpha, 1.0)
        prob_diffuse = np.maximum(0.0, np.sum(current_normals * current_dirs, axis=-1, keepdims=True)) / np.pi

        # Handle specular case (alpha > 1)
        w_r = reflect_along_normal(current_rays_w, current_normals)
        cos_alpha = np.sum(w_r * current_dirs, axis=-1, keepdims=True)
        cos_alpha = np.maximum(0.0, cos_alpha)
        # PDF for Phong lobe sampling: (alpha + 1) / (2 * pi) * cos(alpha_spec)^alpha
        prob_specular = (alpha + 1.0) / (2.0 * np.pi) * (cos_alpha ** alpha)

        prob = np.where(is_diffuse, prob_diffuse, prob_specular)
        
        return prob

    def sample(self, mask=None) -> tuple:
        """Samples directions based on the Phong BRDF (cosine or specular lobe)."""
        if self.brdf_params is None or self.rays_w is None or self.normals is None:
            raise ValueError("Initial parameters must be set.")

        current_brdf = self.brdf_params if mask is None else self.brdf_params[mask]
        current_rays_w = self.rays_w if mask is None else self.rays_w[mask]
        current_normals = self.normals if mask is None else self.normals[mask]
        num_samples = current_brdf.shape[0]

        if num_samples == 0:
            return np.empty((0, 3)), np.empty((0, 1))

        alpha = current_brdf[:, -1:] # Phong exponent
        is_diffuse = np.isclose(alpha, 1.0)

        # --- Generate samples in a local frame ---
        rv1 = np.random.rand(num_samples, 1)
        rv2 = np.random.rand(num_samples, 1)
        phi = 2.0 * np.pi * rv2

        # Cosine sampling (for diffuse or as base for specular)
        cos_theta_cos = np.sqrt(1.0 - rv1) # cos(theta) = sqrt(1-u1) for cosine sampling
        sin_theta_cos = np.sqrt(rv1)       # sin(theta) = sqrt(u1)

        # Phong sampling (importance sampling the specular lobe)
        cos_theta_phong = rv1**(1.0 / (alpha + 1.0)) # cos(alpha_spec) = u1^(1/(alpha+1))
        sin_theta_phong = np.sqrt(np.maximum(0.0, 1.0 - cos_theta_phong**2))

        # Choose cos_theta based on diffuse/specular
        cos_theta = np.where(is_diffuse, cos_theta_cos, cos_theta_phong)
        sin_theta = np.where(is_diffuse, sin_theta_cos, sin_theta_phong)

        # Local coordinates
        x_local = sin_theta * np.cos(phi)
        y_local = sin_theta * np.sin(phi)
        z_local = cos_theta
        w_local = np.concatenate([x_local, y_local, z_local], axis=-1)

        # --- Determine the reference axis for rotation ---
        # Diffuse: Rotate from Z-axis to Normal
        # Specular: Rotate from Z-axis to Reflection direction
        w_r = reflect_along_normal(current_rays_w, current_normals)
        reference_axis = np.where(is_diffuse, current_normals, w_r)

        # --- Rotate local samples to world frame ---
        w = rotate_vectors(w_local,reference_axis)

        # --- Calculate PDF ---
        # PDF for cosine sampling: cos(theta_N) / pi = dot(N, w) / pi
        prob_diffuse = np.maximum(0.0, np.sum(current_normals * w, axis=-1, keepdims=True)) / np.pi
        # PDF for Phong sampling: (alpha + 1) / (2 * pi) * cos(alpha_spec)^alpha = (a+1)/(2pi) * dot(R, w)^a
        cos_alpha_spec = np.maximum(0.0, np.sum(w_r * w, axis=-1, keepdims=True))
        prob_specular = (alpha + 1.0) / (2.0 * np.pi) * (cos_alpha_spec ** alpha)

        prob = np.where(is_diffuse, prob_diffuse, prob_specular)
        
        return w, prob

    # Override illumination to potentially use the simplified BRDF calculation from original code
    # Original: L = Le * BRDF_term / prob, where BRDF_term was different for diffuse/specular
    # Base class uses: L = Le * BRDF_func * cos / prob
    # Let's stick to the base class for consistency unless the specific BRDF calculation here is required.
    # def illumination(self, light_e: np.ndarray, dirs: np.ndarray, prob: np.ndarray, mask=None) -> np.ndarray:
    #     # ... implement specific BRDF calculation from original if needed ...
    #     return super().illumination(light_e, dirs, prob, mask)


class MISampling(Sampling):
    """
    Implements Multiple Importance Sampling (MIS) using a balance heuristic,
    combining LightSampling and BRDFSampling.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(sampling_type="mis", **kwargs)
        # Sub-samplers will be instantiated later when parameters/light are set
        self.light_sampler = None
        self.brdf_sampler = None
        self.coin_toss = None # Will be generated in _on_params_set

    def _on_params_set(self):
        """Instantiate sub-samplers and generate coin toss when params are set."""
        if self.hit_points is None: return # Should not happen if called after set_initial_params

        # Generate coin toss based on the number of hit points
        self.coin_toss = np.random.rand(self.hit_points.shape[0]) < 0.5 # 50/50 split

        # Instantiate BRDF sampler (doesn't need light)
        self.brdf_sampler = BRDFSampling()
        mask_brdf = ~self.coin_toss
        self.brdf_sampler.set_initial_params(self.hit_points, 
                                             self.normals, 
                                             self.brdf_params, 
                                             self.rays_w)

        # Instantiate Light sampler if light is already set
        if self.light is not None:
            self._ensure_light_sampler()

    def _on_light_set(self):
         """Instantiate light sampler if parameters are already set."""
         if self.hit_points is not None: # Check if params are set
              self._ensure_light_sampler()

    def _ensure_light_sampler(self):
        """Creates the light sampler instance if it doesn't exist."""
        if self.light_sampler is None and self.light is not None and self.hit_points is not None:
             self.light_sampler = LightSampling()
             # Pass the currently set parameters and light
             mask_light = self.coin_toss
             self.light_sampler.set_initial_params(self.hit_points, 
                                                   self.normals, 
                                                   self.brdf_params, 
                                                   self.rays_w)
             self.light_sampler.set_light(self.light)


    def sample(self, mask=None) -> tuple:
        """
        Samples using either light or BRDF sampling based on a coin toss,
        and calculates the combined MIS probability using the balance heuristic.
        """
        if self.light_sampler is None or self.brdf_sampler is None or self.coin_toss is None:
            raise ValueError("MISampler not fully initialized. Ensure set_initial_params and set_light are called.")
        if mask is not None:
            raise NotImplementedError("Masking is not fully supported in MISampling sample method yet.")

        # --- Sample using one strategy based on coin toss ---
        dirs = np.zeros_like(self.hit_points)
        prob_sample = np.zeros((self.hit_points.shape[0], 1)) # PDF of the strategy used for sampling

        # Sample using Light strategy where coin_toss is True
        mask_light = self.coin_toss
        if np.any(mask_light):
             dirs[mask_light], prob_sample[mask_light] = self.light_sampler.sample(mask=mask_light)

        # Sample using BRDF strategy where coin_toss is False
        mask_brdf = ~self.coin_toss
        if np.any(mask_brdf):
             dirs[mask_brdf], prob_sample[mask_brdf] = self.brdf_sampler.sample(mask=mask_brdf)

        # --- Evaluate PDF using the *other* strategy ---
        prob_other = np.zeros_like(prob_sample)

        # Evaluate BRDF PDF for directions sampled by Light strategy
        if np.any(mask_light):
             prob_other[mask_light] = self.brdf_sampler.eval_prob_dist(dirs, mask=mask_light)

        # Evaluate Light PDF for directions sampled by BRDF strategy
        if np.any(mask_brdf):
             prob_other[mask_brdf] = self.light_sampler.eval_prob_dist(dirs, mask=mask_brdf)

        # --- Combine probabilities using Balance Heuristic ---
        # pdf_mis = 0.5 * pdf_light + 0.5 * pdf_brdf
        # Where coin_toss is True, pdf_sample = pdf_light, pdf_other = pdf_brdf
        # Where coin_toss is False, pdf_sample = pdf_brdf, pdf_other = pdf_light
        pdf_light = np.where(mask_light[:, np.newaxis], prob_sample, prob_other)
        pdf_brdf = np.where(mask_brdf[:, np.newaxis], prob_sample, prob_other)

        # Combined PDF using balance heuristic (n1=1, n2=1, w1=0.5, w2=0.5 -> pdf = 0.5*pdf1 + 0.5*pdf2)
        # Power heuristic might be better: w_i = pdf_i / (pdf_light + pdf_brdf) -> pdf = pdf_light*w_light + pdf_brdf*w_brdf
        # Let's use the simple balance heuristic from the original code structure:
        prob_mis = 0.5 * pdf_light + 0.5 * pdf_brdf
        # Avoid division by zero if both pdfs are zero
        prob_mis = np.maximum(prob_mis, 1e-9)

        return dirs, prob_mis

    def illumination(self, light_e: np.ndarray, dirs: np.ndarray, prob: np.ndarray, mask=None) -> np.ndarray:
        """
        Calculates illumination using the MIS probability.
        The base class illumination function should work correctly if provided with
        the combined MIS probability 'prob'.
        """
        if self.light_sampler is None or self.brdf_sampler is None:
             raise ValueError("MISampler not fully initialized.")
        if mask is not None:
             raise NotImplementedError("Masking is not fully supported in MISampling illumination method yet.")

        # The base class calculates: Le * BRDF * cos / prob
        # Here 'prob' is the combined MIS PDF calculated in sample()
        return super().illumination(light_e, dirs, prob, mask=None) # Pass mask=None as MIS sample doesn't support it yet
