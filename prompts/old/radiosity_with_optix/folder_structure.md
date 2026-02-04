# Folder Structure - src/

## Overview
Organized by functionality for clean separation of concerns and easy debugging.

```
src/
├── core/              # Core radiosity algorithm implementation
├── geometry/          # Geometric primitives and spatial structures
├── math/              # Mathematical utilities and data structures
├── scene/             # Scene definition, materials, lights
├── visibility/        # Ray tracing and visibility testing (OptiX)
├── debug/             # Debug visualization and logging tools
├── output/            # Image writing and color processing
└── main.cpp           # Program entry point
```

## Directory Descriptions

### `core/` - Radiosity Algorithm Core
Contains the main radiosity computation logic.

**Files to create:**
- `Patch.h/cpp` - Surface element (patch) representation
  - Position, normal, area
  - Radiosity value (emitted + reflected)
  - Reflectance (albedo)
  - Emission (for lights)
  
- `FormFactor.h/cpp` - Form factor computation
  - Geometric form factor calculation
  - Hemisphere sampling utilities
  - Form factor matrix/storage
  
- `RadiosityRenderer.h/cpp` - Main renderer class
  - Scene setup and patch subdivision
  - Iterative solver (progressive refinement)
  - Convergence checking
  - DEBUG: Iteration logging, energy tracking
  
- `Scene.h/cpp` - Scene container
  - Patch collection management
  - Spatial acceleration (later)
  - Scene statistics

### `geometry/` - Geometric Primitives
Basic geometric structures for the scene.

**Files to create:**
- `Triangle.h/cpp` - Triangle primitive
  - 3 vertices
  - Normal computation
  - Area calculation
  - Point sampling for patch centers
  
- `Mesh.h/cpp` - Triangle mesh
  - Collection of triangles
  - Vertex and index buffers
  - Subdivision for patch generation
  
- `BoundingBox.h/cpp` - Axis-aligned bounding box
  - For spatial queries
  - Ray-box intersection

### `math/` - Mathematical Utilities
Header-only math library (or thin wrappers around GLM).

**Files to create:**
- `Vector3.h` - 3D vector operations
  - Add, subtract, multiply, divide
  - Dot product, cross product
  - Normalize, length
  - Component-wise operations
  
- `Matrix4.h` - 4x4 matrix transformations
  - Identity, translation, rotation, scale
  - Matrix multiplication
  - Inverse (if needed)
  
- `MathUtils.h` - Utility functions
  - Clamp, lerp, smoothstep
  - Random number generation
  - Epsilon comparisons
  - Color space conversions

### `scene/` - Scene Definition
Cornell Box and material definitions.

**Files to create:**
- `CornellBox.h/cpp` - Cornell Box scene
  - Hardcoded geometry creation
  - Standard dimensions and colors
  - Light source definition
  - Patch subdivision for each wall
  
- `Material.h/cpp` - Surface material
  - Diffuse reflectance (RGB)
  - Emission (RGB, for lights)
  - Possibly specular (later)
  
- `Light.h/cpp` - Light source
  - Area light definition
  - Emission intensity
  - Geometry (quad for Cornell Box ceiling light)

### `visibility/` - Ray-Based Visibility
OptiX integration for visibility testing.

**Files to create:**
- `OptiXContext.h/cpp` - OptiX initialization
  - Context setup
  - Pipeline creation
  - Shader binding table
  - Module loading
  
- `VisibilityTester.h/cpp` - Visibility queries
  - Ray generation between patches
  - Binary visibility test (occluded/visible)
  - Batch ray casting for efficiency
  - DEBUG: Ray visualization data
  
- `optix_kernels.cu` - CUDA/OptiX device code
  - Ray generation program
  - Closest hit program
  - Miss program
  - Any-hit program (for shadow rays)

### `debug/` - Debug & Visualization
Tools to make bugs visible.

**Files to create:**
- `DebugRenderer.h/cpp` - Debug output manager
  - Patch wireframe rendering
  - Patch coloring modes
  - Ray visualization
  - Text overlay (iteration count, etc.)
  
- `PatchVisualizer.h/cpp` - Patch-specific debug
  - Color patches by:
    - Radiosity magnitude
    - Unshot radiosity
    - Number of visible neighbors
    - Form factor sum
  - Highlight specific patches
  
- `FormFactorVisualizer.h/cpp` - Form factor debug
  - Heatmap visualization
  - Show form factors from selected patch
  - Reciprocity validation visualization
  
- `Logger.h/cpp` - Logging system
  - Timestamped log messages
  - Log levels (DEBUG, INFO, WARNING, ERROR)
  - File and console output

### `output/` - Image Output
Writing final and intermediate images.

**Files to create:**
- `ImageWriter.h/cpp` - Image file I/O
  - Write to PPM (simple format)
  - Write to PNG (using STB)
  - Support for HDR output (later)
  
- `ColorMapper.h/cpp` - Color processing
  - Tone mapping (HDR to LDR)
  - Gamma correction
  - False color visualization for debug
  - Heatmap generation

### `main.cpp` - Entry Point
Main program flow and configuration.

**Contents:**
- Parse command-line arguments (debug mode, output path, etc.)
- Initialize OptiX context
- Create Cornell Box scene
- Subdivide into patches
- Run radiosity solver
- Output images (final + debug)
- Cleanup

## Implementation Order

**Phase 1: Math & Geometry Foundation**
1. `math/Vector3.h`
2. `math/Matrix4.h`
3. `math/MathUtils.h`
4. `geometry/Triangle.h/cpp`
5. `geometry/Mesh.h/cpp`

**Phase 2: Scene Setup**
6. `scene/Material.h/cpp`
7. `scene/Light.h/cpp`
8. `scene/CornellBox.h/cpp`

**Phase 3: Radiosity Core**
9. `core/Patch.h/cpp`
10. `core/Scene.h/cpp`
11. `core/FormFactor.h/cpp`

**Phase 4: Visibility (OptiX)**
12. `visibility/OptiXContext.h/cpp`
13. `visibility/optix_kernels.cu`
14. `visibility/VisibilityTester.h/cpp`

**Phase 5: Solver**
15. `core/RadiosityRenderer.h/cpp`

**Phase 6: Output & Debug**
16. `output/ColorMapper.h/cpp`
17. `output/ImageWriter.h/cpp`
18. `debug/Logger.h/cpp`
19. `debug/DebugRenderer.h/cpp`
20. `debug/PatchVisualizer.h/cpp`

**Phase 7: Integration**
21. `main.cpp`

## Naming Conventions

- **Classes**: PascalCase (`RadiosityRenderer`, `FormFactor`)
- **Files**: Match class name (`RadiosityRenderer.h`)
- **Variables**: camelCase (`patchIndex`, `formFactor`)
- **Constants**: UPPER_SNAKE_CASE (`MAX_ITERATIONS`, `PI`)
- **Namespaces**: lowercase (`radiosity::core`, `radiosity::math`)

## Build System

Create `CMakeLists.txt` at project root to handle:
- C++17 standard
- CUDA compilation for `.cu` files
- OptiX include paths
- GLM linkage
- Debug/Release configurations

## Debug Philosophy

Each module should have:
1. **Verbose logging** - Can be toggled at compile time
2. **Validation checks** - Assert correctness (e.g., energy conservation)
3. **Visualization hooks** - Export debug data for visualization
4. **Unit testability** - Small, focused functions

## Next Actions

1. ✅ Create folder structure
2. ⏭️ Install OptiX (see optix_wsl_installation.md)
3. ⏭️ Create CMakeLists.txt
4. ⏭️ Implement Phase 1 (Math & Geometry)
