**I built a physically-based global illumination renderer from scratch. No game engine. No shortcuts. Just math and GPU power.** 🔥

The classic Cornell Box — rendered with real light physics, not faked.

## How it works:

 ┌─────────────┐
 │  Scene Setup │  Cornell Box geometry + materials
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Subdivide   │  Adaptive mesh → 15K+ triangles
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Form Factors│  How much light can each surface
 │  (GPU/OptiX) │  pair exchange? Monte Carlo on GPU
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Visibility  │  Real ray-traced shadows via
 │  (RTX cores) │  NVIDIA OptiX 9.1
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Radiosity   │  Light bounces until convergence
 │  Solve       │  50,000+ iterations
 └──────┬──────┘
        ▼
 ┌─────────────┐
 │  Tone Map +  │  Exposure, gamma, smooth shading
 │  Render      │  Final ray-traced PNG output
 └──────┘

## Why radiosity?

In 1984, Cindy Goral at Cornell published the first-ever radiosity paper in computer graphics: "Modeling the interaction of light between diffuse surfaces." It was a heat-transfer method borrowed from thermal engineering — applied to pixels for the first time. The Cornell Box was literally invented to validate it.

40 years later, most renderers use path tracing (Monte Carlo sampling of the rendering equation). It's flexible but noisy. Radiosity is the opposite: brute-force, deterministic, patch-to-patch. Every surface talks to every other surface. No hemicube approximation. No hemisphere sampling shortcuts. Just the raw double-area integral between every pair of triangles.

It's the OG approach. And it produces that buttery-smooth indirect light that path tracers need thousands of samples to match.

## What makes it real:

- Classical patch-to-patch form factors — the original brute-force method, no hemicube or hemisphere shortcuts
- Every surface exchanges light with every other surface — no faking it
- Visibility checking on GPU via OptiX ray tracing!
- Light bounces naturally between walls → red/green color bleed
- Soft shadows emerge from the physics, not from blur tricks
- Resolution-independent convergence — works at any mesh density
- Real material properties from the original Cornell Box spec: https://bowers.cornell.edu/cornell-box

**The stack**: C++ / CUDA / OptiX 9.1 / RTX 4070 SUPER
**Core code**: ~800 lines. That's it.

No Unity. No Unreal. No Blender. Just the raw equations of light. ✨

---

*References:*
- *Goral, Torrance, Greenberg, Battaile (1984). "Modeling the interaction of light between diffuse surfaces." SIGGRAPH '84.*
- *Material reflectances & geometry from the Cornell Box project: https://bowers.cornell.edu/cornell-box*