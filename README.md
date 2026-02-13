# Radiosity Cornell Box

Classic global illumination via progressive radiosity (shooting method) with
GPU-accelerated form-factor computation using NVIDIA OptiX ray tracing.

![Cornell Box Render](output/cornell_render.png)

## Features

- Procedural Cornell Box scene with adaptive mesh subdivision
- Progressive radiosity solver with convergence control
- GPU form factors via OptiX Monte Carlo area-to-area integration
- CPU fallback (centroid-to-centroid, no visibility)
- Smooth per-vertex color OBJ export
- OptiX ray-traced PNG output with barycentric color interpolation

## Prerequisites

- **CMake** ≥ 3.21
- **Visual Studio 2022** (or another C++17 / CUDA-capable toolchain)
- **CUDA Toolkit** (tested with 12.x)
- **NVIDIA OptiX SDK 9.1** — install to the default path or set `OptiX_INSTALL_DIR` in CMakeLists.txt
- **OpenMP** (included with MSVC)

To build without a GPU, set `-DUSE_OPTIX=OFF` in the configure step. The solver
will fall back to CPU-only form factors.

## Build

```bash
cmake --preset release
cmake --build --preset release
```

Debug build:

```bash
cmake --preset debug
cmake --build --preset debug
```

## Run

```bash
./build/bin/Release/radiosity.exe
```

Options:

| Flag | Description |
|---|---|
| `--output PATH` | Output directory (default: `output`) |
| `--no-validate` | Skip mesh validation |
| `--help` | Show help |

Output files are written to the output directory:
- `cornell_radiosity.obj` — smoothed vertex-color OBJ
- `cornell_render.png` — ray-traced PNG (OptiX only)

## Configuration

All tuning constants live in [`src/app/Config.h`](src/app/Config.h):

| Constant | Default | Description |
|---|---|---|
| `kSubdivisionTargetArea` | 0.001 | Target triangle area for subdivision |
| `kVisibilitySamples` | 32 | Monte Carlo samples per form-factor target |
| `kIndirectBoostFactor` | 1.3 | Indirect light multiplier |
| `kToneMapExposure` | 1.4 | Exposure before gamma |
| `kToneMapGamma` | 0.8 | Gamma exponent |
| `kRenderWidth/Height` | 3840 | PNG render resolution |

## Project Structure

```
src/
  main.cpp                  Entry point, radiosity solver, tone mapping
  app/Config.h              Tuning constants and CLI parsing
  scene/CornellBox.h        Procedural Cornell Box geometry
  mesh/MeshData.h           Mesh data structures
  mesh/Subdivision.h        Adaptive triangle subdivision
  mesh/PatchBuilder.h       Per-triangle geometry and material setup
  math/Vec3.h               3D vector type
  math/MathUtils.h          Triangle area, normal, centroid
  math/HemisphereSampling.h Cosine-weighted hemisphere sampling (CUDA)
  export/OBJExporter.h      Smoothed vertex-color OBJ export
  gpu/OptiXContext.h        OptiX context, GAS build, form-factor launch
  gpu/Renderer.h            OptiX ray-traced PNG renderer
  gpu/HemisphereFormFactorKernels.cu  Form-factor visibility kernel
  gpu/RenderKernels.cu      Primary-ray render kernel
third_party/
  stb_image_write.h         PNG writer (single-header library)
```

## License

See repository root for license information.
