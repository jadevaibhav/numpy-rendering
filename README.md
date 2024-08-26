# numpy-rendering

numpy-based python implementation of physics-based rendering algorithms for learning: 
1. Ray tracing: Direct illumination of object meshes with directional light and point light source
2. Ray tracing: Direct illumination with anti-aliasing and ambient occlusion light source
3. Ray tracing: Direct illumination with object light source, using MC importance sampling (Light, BRDF, and multiple importance) 
4. Path tracing: Indirect illumination with implicit and explicit path tracing
5. Photon Mapping: Indirect illumination with light path sampling

I have structured this repo to support multiple physics-based rendering algorithms, with a sampling technique of your choice. You can use any sampling technique or implement one on your own by extending the Sampling class.
Any sampling technique should have 3 components:
1. A way to sample from the given distribution
2. A way to calculate the probability density of the sample (direction)
3. A way to compute the illumination at the given point

I am still working on making this a modular implementation supporting as many components of the rendering pipeline as possible. All feedback is welcome!
