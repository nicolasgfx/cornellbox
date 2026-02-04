# Radiosity Cornell Box Renderer - Phase 1 Complete

A clean, academic implementation of a radiosity renderer for the classic Cornell Box scene.

**Current Status:** ✅ **Phase 1 Complete** - Geometry & Data Layout

---

## Quick Start

### Build

```powershell
# Configure CMake
cd build
cmake ..

# Build Release
cmake --build . --config Release

# Copy executable
cd ..
Copy-Item build\bin\Release\radiosity.exe output\bin\
```

### Run

```powershell
# Run with different profiles
.\output\bin\radiosity.exe --low      # 32 triangles
.\output\bin\radiosity.exe --medium   # 128 triangles
.\output\bin\radiosity.exe --high     # 512 triangles

# Or use the run script
powershell -ExecutionPolicy Bypass -File run.ps1 -Profile medium
```

### View Output

OBJ files are generated in `output/scenes/`:
- `cornell_phase1_low.obj`
- `cornell_phase1_medium.obj`
- `cornell_phase1_high.obj`

Open these in any 3D viewer (Blender, MeshLab, etc.) to visualize the Cornell box.

---

## Phase 1 Implementation

### What's Implemented

✅ **Geometry System**
- Procedural Cornell box generation
- Triangle-only representation (no quads)
- Automatic normal orientation enforcement
- Three performance profiles with subdivision

✅ **Data Structures**
- Structure-of-Arrays (SoA) for patch attributes
- GPU-friendly memory layout (ready for Phase 2)
- Vertex adjacency for color baking

✅ **Material System**
- Red left wall, green right wall, white other surfaces
- Ceiling area light with emission
- Reflectance and emission per triangle

✅ **Export**
- OBJ/MTL export with vertex colors
- Area-weighted color baking
- Compatible with standard 3D viewers

### Code Structure

```
src/
├── main.cpp                    # Application entry point
├── app/
│   └── Config.h               # CLI argument parsing
├── math/
│   ├── Vec3.h                 # Vector math
│   └── MathUtils.h            # Triangle geometry utilities
├── mesh/
│   ├── MeshData.h             # Core data structures (Vertex, TriIdx, PatchSoA)
│   ├── Subdivision.h          # Triangle subdivision
│   └── PatchBuilder.h         # SoA construction & validation
├── scene/
│   ├── CornellBox.h           # Procedural Cornell box builder
│   └── OBJLoader.h            # OBJ file parser (for future use)
└── export/
    ├── VertexColor.h          # Color baking utilities
    └── OBJExporter.h          # OBJ/MTL writer
```

### Key Design Decisions

1. **Triangle == Patch:** Every triangle is exactly one radiosity patch
2. **SoA Layout:** All per-patch data stored in separate arrays for GPU efficiency
3. **Separation of Concerns:** 
   - Geometry (vertices/indices) for OptiX traversal only
   - PatchSoA for all radiosity computation
4. **No Half-Edge (Yet):** Simple vertex adjacency sufficient for Phase 1
5. **Normals Enforced:** Walls inward, boxes outward, validated automatically

---

## Validation

All Phase 1 checklist items pass:

- ✅ Triangle-only geometry
- ✅ Single connected mesh
- ✅ SoA patch data structure
- ✅ Normal orientation correct
- ✅ Performance profiles working
- ✅ Vertex color baking
- ✅ OBJ/MTL export functional
- ✅ No anti-patterns present

See `output/PHASE1_VALIDATION_REPORT.md` for detailed validation results.

---

## Performance Profiles

| Profile | Triangles | Vertices | Description |
|---------|-----------|----------|-------------|
| Low     | 32        | 28       | Original geometry |
| Medium  | 128       | 192      | 1 subdivision level (4×) |
| High    | 512       | 768      | 2 subdivision levels (16×) |

---

## Next Steps - Phase 2: Visibility

Phase 2 will add:
- OptiX ray tracing setup
- Triangle-to-triangle visibility computation
- Visibility cache on disk
- AO-like debug visualization

Phase 1 code will not require refactoring for Phase 2.

---

## Requirements

- CMake 3.18+
- C++17 compiler (MSVC, GCC, or Clang)
- Windows/Linux

**Phase 2 will additionally require:**
- CUDA 12.4+
- OptiX 9.1.0+
- NVIDIA GPU

---

## Project Structure

```
radiosity/
├── build/              # CMake build directory
├── output/
│   ├── bin/           # Executable
│   ├── scenes/        # Generated OBJ files
│   ├── cache/         # (Phase 2) Visibility cache
│   └── logs/          # (Future) Debug logs
├── src/               # Source code (see above)
├── prompts/           # Project planning documents
├── doc/               # Documentation
├── third_party/       # (Phase 2) OptiX SDK
├── CMakeLists.txt     # Build configuration
├── build.ps1          # Build script
└── run.ps1            # Run script
```

---

## License

Academic/Educational Use

---

**Status:** Phase 1 Complete ✅ | Ready for Phase 2 🚀


A progressive radiosity renderer with NVIDIA OptiX ray tracing acceleration.

## Features

- **Classical Radiosity**: Progressive refinement algorithm with Monte Carlo form factor computation
- **OptiX Acceleration**: GPU-accelerated visibility testing using NVIDIA OptiX 9.1.0
- **Per-Vertex Interpolation**: Smooth radiosity gradients across mesh surfaces
- **Visibility Caching**: Pre-computed form factors stored for rapid iteration
- **Cornell Box Scenes**: Standard test scenes with color bleeding demonstration

## Requirements

### Software
- **Visual Studio 2022** (Community or higher)
- **CMake 3.18+** (included with VS 2022)
- **CUDA Toolkit 12.4** (for OptiX support)

### Hardware
- **NVIDIA GPU** with OptiX support (RTX series recommended)
- **Windows 10/11** x64

**Note:** OptiX SDK 9.1.0 headers are included in `third_party/optix/` - no separate installation required!

## Directory Structure

```
radiosity/
├── src/                    # Source code
│   ├── main.cpp           # Main entry point
│   ├── math/              # Vector/matrix math
│   ├── geometry/          # Mesh data structures
│   ├── core/              # Patch and scene
│   ├── radiosity/         # Radiosity solver
│   ├── visibility/        # OptiX integration
│   └── output/            # OBJ/PLY export
│
├── third_party/            # Third-party dependencies
│   └── optix/             # OptiX SDK 9.1.0 (local copy)
│       ├── include/       # OptiX headers
│       ├── lib/           # GLFW libraries
│       └── bin/           # GLFW runtime DLL
│
├── build/                  # CMake build directory (generated)
│   └── RadiosityRenderer.sln  # Visual Studio solution
│
├── output/                 # All program outputs
│   ├── bin/               # Executables and PTX
│   │   ├── radiosity.exe
│   │   └── optix_kernels.ptx
│   ├── scenes/            # Generated .obj/.mtl files
│   ├── cache/             # Visibility cache (.cache files)
│   └── logs/              # Test output and logs
│
├── doc/                    # Documentation
├── prompts/                # Development notes
│
├── CMakeLists.txt         # CMake configuration
├── build.ps1              # Build script
├── run.ps1                # Run script
├── clean.ps1              # Clean script
└── test_ptx_rebuild.ps1   # PTX rebuild test
```

## Quick Start

### 1. Environment Setup

**OptiX SDK:** Already included in `third_party/optix/` - no setup needed!

**CUDA Toolkit:** Verify CUDA 12.4 is installed:

```powershell
# Verify CUDA is installed
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"
```

If not installed, download from [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads).

### 2. Build

```powershell
# Clean build (recommended for first time)
.\build.ps1 -Clean

# Incremental build
.\build.ps1

# Debug build
.\build.ps1 -Config Debug

# Verbose output
.\build.ps1 -Verbose
```

**Build output:**
- `output/bin/radiosity.exe` - Main executable
- `output/bin/optix_kernels.ptx` - GPU kernels (auto-generated)

### 3. Run

```powershell
# Run with filtered output
.\run.ps1

# Run with full output
.\run.ps1 -Verbose

# Clear cache and run
.\run.ps1 -NoCache
```

**Note:** The executable must run from `output/bin/` to find the PTX file.

### 4. View Results

Generated files are in `output/scenes/`:

```
cornell_box.obj                  # Base geometry
cornell_box_radiosity.obj        # Solved radiosity
cornell_box_visibility_debug.obj # Visibility visualization
```

**View with:**
- **Blender**: File → Import → Wavefront (.obj)
- **Online**: [3dviewer.net](https://3dviewer.net/)
- **MeshLab**: `meshlab cornell_box_radiosity.obj`

## Build System Features

### Automatic PTX Rebuild

The build system automatically recompiles OptiX kernels when CUDA source changes:

```powershell
# Test PTX auto-rebuild
.\test_ptx_rebuild.ps1
```

**How it works:**
1. CMake tracks `src/visibility/optix_kernels.cu` as a dependency
2. When modified, `nvcc` recompiles to `output/bin/optix_kernels.ptx`
3. `radiosity.exe` depends on PTX target, ensuring correct build order

### Clean Output Structure

**Before restructure:**
- Files scattered across 4 locations
- 30+ files in build directory
- Manual PTX copying required

**After restructure:**
- All outputs in `output/` tree
- 13 files in build directory
- Automatic PTX placement

### Build Scripts

| Script | Purpose |
|--------|---------|
| `build.ps1` | Configure and build project |
| `run.ps1` | Execute renderer from correct directory |
| `clean.ps1` | Remove build artifacts |
| `test_ptx_rebuild.ps1` | Verify PTX dependency tracking |

## Development Workflow

### Typical Iteration

```powershell
# 1. Edit source code
code src/main.cpp

# 2. Build
.\build.ps1

# 3. Run
.\run.ps1 -Verbose

# 4. View output
blender output/scenes/cornell_box_radiosity.obj
```

### Cache Management

Visibility caching significantly speeds up repeated runs:

```powershell
# Use cache (default)
.\run.ps1

# Clear cache (force recompute)
.\run.ps1 -NoCache

# Manual cache management
Remove-Item output/cache/*.cache
```

**Cache location:** `output/cache/cornell_box_14x7_visibility.cache`

**Cache contents:** Visibility fractions (0-1), NOT form factors

### Full Rebuild

```powershell
# Clean everything
.\clean.ps1 -All

# Rebuild from scratch
.\build.ps1 -Clean

# Verify
.\run.ps1 -Verbose
```

## Configuration Options

### CMakeLists.txt

```cmake
# OptiX support (default: ON)
option(USE_OPTIX "Enable OptiX ray tracing" ON)

# Build type (default: Release)
set(CMAKE_BUILD_TYPE Release)  # or Debug

# CUDA architecture (default: sm_86 for RTX 30xx)
-arch=sm_86  # Change for different GPU generations
```

### Radiosity Solver (main.cpp)

```cpp
radiosity::RadiosityRenderer::Config config;
config.convergenceThreshold = 0.0001f;  // Lower = more bounces
config.maxIterations = 100;             // Max iterations
config.verbose = true;                  // Show progress
```

### Scene Complexity

```cpp
// In exportRadiositySolution():
box.build(14, 7);  // 14x14 walls, 7x7 boxes (~20 min)
box.build(8, 4);   // 8x8 walls, 4x4 boxes (~2 min)
box.build(4, 2);   // 4x4 walls, 2x2 boxes (~10 sec)
```

## Troubleshooting

### Build Errors

**"OptiX not found"**
```powershell
# Set OptiX path
$env:OPTIX_ROOT = "C:\ProgramData\NVIDIA Corporation\OptiX SDK 9.1.0"
.\build.ps1 -Clean
```

**"CUDA not found"**
```powershell
# Verify CUDA installation
Test-Path "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4"

# Update path in CMakeLists.txt if different version
```

**"MSBuild not found"**
```powershell
# Install Visual Studio 2022 with C++ workload
# Or update path in build.ps1
```

### Runtime Errors

**"PTX file not found"**
```powershell
# Ensure running from output/bin
Push-Location output/bin
.\radiosity.exe
Pop-Location

# Or use run script
.\run.ps1
```

**"OptiX initialization failed"**
- Update NVIDIA drivers
- Verify GPU supports OptiX (RTX series or GTX 10xx+)
- Check CUDA/OptiX version compatibility

### Performance Issues

**Slow visibility computation:**
- Reduce scene complexity: `box.build(4, 2)` for testing
- Enable caching: `DEBUG_ENABLE_CACHE = true`
- Verify OptiX is initialized (look for "OptiX initialized" message)

**Slow build times:**
- Use incremental builds: `.\build.ps1` (not `-Clean`)
- PTX only rebuilds when CUDA source changes
- Use `/m` flag for parallel builds (already enabled)

## Testing

### Unit Tests (Built-in)

```powershell
.\run.ps1 -Verbose
```

Tests include:
- Vector3 math operations
- IndexedMesh construction
- Patch initialization
- Cornell Box generation
- Form factor computation
- OptiX visibility testing
- Radiosity convergence

### PTX Rebuild Test

```powershell
.\test_ptx_rebuild.ps1
```

Verifies:
- CUDA source modification triggers PTX rebuild
- PTX output in correct location
- Timestamp updates correctly

## Performance

### Scene Complexity vs Time

| Configuration | Patches | Time (GTX 1660 Ti) | Time (RTX 3080) |
|--------------|---------|-------------------|-----------------|
| box.build(4, 2) | ~100 | ~10 sec | ~3 sec |
| box.build(8, 4) | ~400 | ~2 min | ~30 sec |
| box.build(14, 7) | ~1200 | ~20 min | ~5 min |
| box.build(21, 10) | ~2700 | ~60 min | ~15 min |

### Optimization Tips

1. **Enable caching** for repeated runs with same scene
2. **Start small** (4x4 walls) for testing, scale up when working
3. **Use Release build** (5-10x faster than Debug)
4. **Lower convergence threshold** if slight inaccuracy acceptable

## Known Issues

1. **PTX Architecture**: Default `sm_86` targets RTX 30xx series. Older GPUs need different arch:
   - RTX 20xx: `sm_75`
   - GTX 10xx: `sm_61`
   - Update in CMakeLists.txt: `-arch=sm_XX`

2. **Path Separators**: Windows-only. For Linux/macOS, update path handling in main.cpp

3. **Memory Usage**: Large scenes (>3000 patches) may exceed GPU memory. Reduce subdivision if crashes occur.

## References

- [OptiX Programming Guide](https://raytracing-docs.nvidia.com/optix8/guide/index.html)
- [CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/)
- [Radiosity Theory](doc/radiosity_course_theory_first_introduction.md)

## License

This project is for educational purposes.

## Author

Developed during Week 1-3 of radiosity renderer implementation course.

---

**Last Updated:** February 2026  
**Version:** 2.0 (Post-build-revision)
