# Radiosity Renderer Coding Plan - Cornell Box

## Project Goal
Build a classical radiosity renderer for the Cornell Box scene from scratch, starting with a "debuggy" version that makes it easy to identify and fix issues.

## Phase 1: Prerequisites & Setup

### 1.1 Environment Setup
- [ ] Install OptiX SDK for Windows (Week 3)
- [x] Install CUDA Toolkit 12.3+ for Windows
- [x] Install Visual Studio 2019/2022 with C++ tools
- [x] Install CMake 3.18+ (included with Visual Studio)
- [ ] Install image I/O library (STB or similar) - Week 5
- [x] Verify GPU drivers - NVIDIA GeForce RTX 4070 (12GB) with Windows drivers

### 1.2 Folder Structure (Updated - Week 2 Complete)
```
src/
├── core/              # Core radiosity algorithm
│   ├── Scene.h        # ✅ RadiosityScene combining patches + indexed mesh
│   ├── Patch.h        # ✅ Radiosity patch with B, B_unshot, emission, reflectance
│   ├── FormFactor.h   # Week 3 - Form factor calculation
│   └── RadiosityRenderer.h  # Week 4 - Progressive refinement solver
├── geometry/          # Geometric primitives (OptiX-ready)
│   └── IndexedMesh.h  # ✅ Shared vertices + index buffer + patchIds
├── math/              # Math utilities
│   ├── Vector3.h      # ✅ 3D vector operations
│   ├── Matrix4.h      # ✅ 4x4 transformations
│   └── MathUtils.h    # ✅ PI, EPSILON, random sampling, tone mapping
├── scene/             # Scene definitions
│   ├── CornellBox.h   # ✅ Standard Cornell Box (552.8×548.8×559.2mm)
│   └── Material.h     # ✅ Cornell Box materials (white, red, green, light)
├── visibility/        # Ray-based visibility testing (Week 3)
│   ├── VisibilityTester.h
│   └── OptiXContext.h
├── debug/             # Debug visualization (Week 5)
│   ├── DebugRenderer.h
│   ├── PatchVisualizer.h
│   └── FormFactorVisualizer.h
├── output/            # Output formats
│   ├── OBJWriter.h    # ✅ Wavefront OBJ/MTL export for visualization
│   └── ImageWriter.h  # Week 5 - PPM/PNG output
└── main.cpp           # ✅ Entry point with test suite
```

**Note**: Using header-only design for simplicity. All classes are in .h files.

## Phase 2: Core Implementation (Debuggy Version)

### 2.1 Math Foundation
- [ ] Implement basic Vector3 class (add, subtract, dot, cross, normalize)
- [ ] Implement Matrix4 ✅ COMPLETED (Week 1)
- [x] Implement basic Vector3 class (add, subtract, dot, cross, normalize)
- [x] Implement Matrix4 for transformations (translation, rotation, scale)
- [x] Add math utility functions (clamp, lerp, smoothstep, radians, degrees)
- [x] Random sampling (uniform, cosine-weighted hemisphere)
- [x] Tone mapping (Reinhard)
- **Validated**: All tests passing, used throughout codebase

### 2.2 Geometry System ✅ COMPLETED (Week 2 - Revised Architecture)
- [x] **IndexedMesh** class with OptiX-ready data layout
  - Shared vertex buffer (`vector<Vector3> vertices`)
  - Index buffer (`vector<uint32_t> indices`) for triangle definitions
  - Per-triangle patch mapping (`vector<uint32_t> patchIds`)
  - Direct GPU buffer access via `getVertexDataPtr()` and `getIndexDataPtr()`
  - **Memory efficiency**: ~50% savings vs old embedded-vertex approach
- [x] **MeshBuilder** for subdivided quad generation
- [x] Cornell Box scene definition (hardcoded geometry)
  - Left wall (red), right wall (green)
  - Back wall, floor, ceiling (white)
  - Two rotated boxes (short -18°, tall +18°)
  - Area light on ceiling (130mm square)
- **Learning**: OptiX requires indexed representation, not embedded vertices. Designed for GPU from start.

### 2.3 Patch System (Core Radi (Week 3 - IN PROGRESS)
- [ ] Implement basic form factor formula: F_ij = (cos θi * cos θj) / (π * r²)
- [ ] Add visibility term via OptiX ray tracing
- [ ] Form factor calculation strategies:
  - [ ] Point-to-point (patch center to patch center)
  - [ ] Hemicube method (more accurate for large patches)
  - [ ] Monte Carlo sampling (for validation)
- [ ] Form factor storage (sparse matrix or on-demand computation)
- [ ] DEBUG: Log form factors, visualize with color coding
- **Note**: Data structures ready - patches have center, normal, area

### 2.5 Ray-Based Visibility (OptiX Integration) (Week 3 - NEXT)
- [ ] Install OptiX SDK 7.7+ for Windows
- [ ] Initialize OptiX context in `visibility/OptiXContext.h`
- [ ] Upload IndexedMesh to GPU using existing buffer pointers
  - `mesh.getVertexDataPtr()` → OptiX vertex buffer
  - `mesh.getIndexDataPtr()` → OptiX index buffer
- [ ] Create simple ray tracing pipeline (closest-hit, miss programs)
- [ ] Implement visibility ray casting between patch centers
- [ ] Return visibility value (0.0 = occluded, 1.0 = fully visible)
- [ ] DEBUG: Visualize rays, show which patch pairs are visible
- **Ready**: Mesh already in OptiX-compatible format

### 2.6 Radiosity Solver (Week 4)
- [ ] Progressive refinement method (simpler than matrix inversion)
- [ ] Iterative shooting method:
  1. Find patch with most unshot radiosity (use `Patch::unshotMagnitude()`)
  2. Shoot radiosity to all visible patches
  3. Update radiosity values using form factors
  4. Update B_unshot for affected patches
  5. Repeat until convergence (energy threshold)
- [ ] DEBUG: Print iteration number, energy level, convergence status
- **Ready**: Patch structure has B and B_unshot already

### 2.7 Visualization & Output
- [x] OBJ export for external visualization (Blender, MeshLab)
- [ ] PPM/PNG image output (Week 5)
- [ ] Vertex color interpolation for smooth shading
- [ ] DEBUG modes (Week 5):
  - Show patch boundaries
  - Color-code patches by radiosity level
  - Show form factors as heatmap
  - Highlight shooting patch in each iteration
- **Working**: Currently exporting to OBJ with material colors
  3. Update radiosity values
  4. Repeat until convergence
- [ ] DEBUG: Print iteration number, energy level, convergence status

### 2.7 Visualization & Output
- [ ] Simple rasterizer for patch rendering
- [ ] Vertex interpolation for smooth shading
- [ ] Output to PPM or PNG image
- [ ] DEBUG modes:
  - Show patch boundaries
  - Color-code patches by radiosity level
  - Show form factors as heatmap
  - Highlight shooting patch in each iteration

## Phase 3: Debug Features (Built-in from Start)

### 3.1 Verbose Logging
- [ ] Log each iteration of radiosity solver
- [ ] Log form factor calculations (sample subset)
- [ ] Log visibility test results
- [ ] Execution time per phase

### 3.2 Intermediate Outputs
- [ ] Save image after every N iterations
- [ ] Create animation showing convergence
- [ ] Export patch data to text file for inspection

### 3.3 Validation Tests
- [ ] Energy conservation check (sum of energy in/out)
- [ ] Form factor reciprocity: A_i * F_ij = A_j * F_ji
- [ ] Sum of form factors from a patch ≈ 1.0 (closed environment)

### 3.4 Visual Debug Modes
- [ ] Mode 1: Render patch centers as dots
- [ ] Mode 2: Render patches with wireframe
- [ ] Mode 3: Color patches by number of visible neighbors
- [ ] Mode 4: Show radiosity magnitude as grayscale
- [ ] Mode 5: Normal visualization

## Phase 4: OptiX Setup Instructions

### 4.1 Prerequisites for Windows
```powershell
# 1. Verify GPU drivers
nvidia-smi

# 2. Install Visual Studio 2019 or 2022
# Download from: https://visualstudio.microsoft.com/downloads/
# Select: Desktop development with C++, Windows 10 SDK, CMake tools

# 3. Verify installations
cmake --version
```

### 4.2 Install CUDA Toolkit
```powershell
# Download CUDA Toolkit 12.3+ from:
# https://developer.nvidia.com/cuda-downloads
# Select: Windows → x86_64 → Version (10/11) → exe (local)
# Install to default location: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3

# Verify installation
nvidia-smi
nvcc --version
```

### 4.3 Install OptiX SDK
```powershell
# Download OptiX 7.7 or later from NVIDIA website
# https://developer.nvidia.com/designworks/optix/download
# Extract to C:\OptiX\OptiX-SDK-7.7.0

# Set environment variable (PowerShell)
[System.Environment]::SetEnvironmentVariable('OPTIX_ROOT', 'C:\OptiX\OptiX-SDK-7.7.0', 'User')

# Restart terminal for changes to take effect
```

### 4.4 Build System Setup
See [SETUP_WINDOWS.md](../../SETUP_WINDOWS.md) for detailed build instructions using:
- Visual Studio (GUI)
- CMake command line
- Ninja build system

## Phase 5: Build Configuration

### 5.1 CMakeLists.txt structure
- Main project configuration
- CUDA and OptiX finding
- Compiler flags for C++17
### ✅ Week 1: Foundation (COMPLETED)
- [x] Verify environment (CUDA Driver 13.0, RTX 4070)
- [x] Create folder structure
- [x] Implement math classes (Vector3, Matrix4, MathUtils)
- [x] Build system (CMake with Debug/Release configs)
- [x] All tests passing

### ✅ Week 2: Geometry & Scene (COMPLETED - Architecture Revised)
- [x] Cornell Box scene definition (552.8×548.8×559.2mm standard)
- [x] **OptiX-ready architecture redesign**:
  - Replaced Triangle/Mesh with IndexedMesh (shared vertices)
  - Integrated Patch system with geometry from start
  - Bidirectional patch↔triangle mapping
  - ~50% memory savings, GPU-friendly layout
- [x] Patch subdivision (configurable per surface)
- [x] Scene visualization (OBJ export tested with Blender)
- [x] Test suite validates all components
- **Output**: 16 patches, 1502 triangles, 30.3 KB memory footprint

### 🔄 Week 3: OptiX Integration & Form Factors (CURRENT)
- [ ] Install OptiX SDK 7.7+ for Windows
- [ ] Initialize OptiX context and upload mesh buffers
- [ ] Implement visibility testing with ray tracing
- [ ] Calculate form factors using visibility results
- [ ] Validate form factor properties (reciprocity, sum ≈ 1.0)
- [ ] Debug visualization of form factors
- **Ready**: Data structures prepared, direct buffer access available

### Week 4: Radiosity Solver (PLANNED)
- [ ] Progressive refinement implementation
- [ ] Iterative shooting method using B_unshot tracking
- [ ] Energy conservation validation
- [ ] Convergence monitoring with debug output
- [ ] Per-iteration image export for animation
- **Goal**: Working radiosity solution with soft shadows

### Week 5: Refinement & Debug Tools (PLANNED)
- [ ] Improve convergence (adaptive thresholds)
- [ ] Add debug visualization modes
- [ ] PPM/PNG image output
- [ ] Performance profiling and optimization
- [ ] Hemicube method for form factors (if needed)

### Week 6: Polish & Validation (PLANNED)
- [ ] Higher resolution patches (20×20 walls)
- [ ] Better tone mapping and color output
- [ ] Comparison with reference Cornell Box images
- [ ] Documentation and final report
   - Progressive refinement implementation
   - Iterative solving with debug output
   - Energy conservation validation

5. **Week 5: Refinement & Debug Tools**
   - Improve convergence
   - Add all debug visualization modes
   - Performance profiling ✅ RESOLVED
   - ✅ Verified: CUDA Driver 13.0, RTX 4070 accessible in WSL2
   - Week 3: Will install OptiX SDK and test compatibility
   
2. **Data Structure Design**: OptiX requires indexed representation 🔧 LEARNED
   - ✅ Redesigned: Switched from embedded vertices to IndexedMesh
   - ✅ Result: ~50% memory savings, GPU-ready from start
   - Lesson: Design for GPU upload from the beginning
   
3. **Form Factor Accuracy**: Numerical precision issues (Week 3)
   - Solution: Use float32 (sufficient for radiosity), validate with known cases
   - Validation: Check reciprocity and sum properties
   
4. **Convergence Speed**: Many iterations needed (Week 4)
   - Solution: Start with coarse patches (10×10), progressive refinement
   - Track B_unshot for efficient shooting method
   
5. **Memory Usage**: Large form factor matrix (Week 3-4)
   - Current: 16 patches = 256 form factors (manageable)
   - Future: Sparse storage or compute on-demand for distant patches
   - Consider hierarchical radiosity for high-res scen
3. **Modular Design**: Each component testable independently
4. **Simple First**: Get working version, optimize later
5. **Validation Built-in**: Check correctness at every stage
6. **Clear Data Flow**: Easy to inspect values at any point

## Expected Challenges & Solutions

1. **OptiX on Windows**: Requires CUDA Toolkit and OptiX SDK
   - Solution: Install CUDA 12.3+ and OptiX 7.7+ for Windows
   - ✅ RESOLVED: Migrated from WSL2 to native Windows
   - ✅ Verified: RTX 4070 with Windows drivers
   - Week 3: Installing OptiX SDK for Windows
   
2. **Form Factor Accuracy**: Numerical precision issues
### Week 1-2 (Foundation) ✅
- [x] Cornell Box geometry defined correctly (verified in Blender)
- [x] Materials have correct reflectance (red, green, white)
- [x] Patch subdivision works (16 patches, 1502 triangles)
- [x] Memory efficiency (<50 KB for test scene)
- [x] OptiX-ready data structures (indexed mesh, GPU buffer access)

### Week 3-4 (Radiosity Solution)
- [ ] Form factors calculated correctly (reciprocity validated)
- [ ] OptiX visibility testing works (binary hit/no-hit)
- [ ] Progressive refinement converges (energy threshold)
- [ ] Cornell Box renders with correct base colors
- [ ] Soft shadows visible on boxes and walls
- [ ] Energy conservation validated (<1% error)

### Week 5-6 (Quality & Polish)
- [ ] Color bleeding evident (red tint on white surfaces near red wall)
- [ ] Visual quality matches reference radiosity images
- [ ] Debug modes help identify issues quickly
- [ ] Higher resolution (20×20 patches) looks smooth
- [ ] Final images exportable to PNG with tone mapping
   - Solution: Sparse storage, compute on-demand for distant patches

## Success Criteria

- [ ] Cornell Box renders with correct colors (red/green walls)
- [ ] Soft shadows visible
- [ ] Color bleeding evident (red tint on nearby surfaces)
- [ ] Energy conservation validated
- [ ] Visual quality matches reference radiosity images
- [ ] Debug modes help identify issues quickly

## Key Learnings (Week 1-2)

### Architecture Design
- **OptiX requires indexed meshes**: Don't use embedded vertices
- **Design for GPU from start**: Avoid costly architecture redesigns
- **Bidirectional mapping is critical**: Patches ↔ Triangles must be maintained
- **Memory matters**: Indexed mesh saved ~50% vs embedded approach
- **Header-only is simple**: No .cpp files needed for this project size

### Implementation Notes
- **Vector3 needs sufficient operators**: Component-wise ops, clamping, printing
- **Matrix4 column-major**: Matches OpenGL/OptiX conventions
- **Patch subdivision**: Each quad → NxM patches → 2×N×M triangles
- **Cornell Box dimensions**: 552.8×548.8×559.2mm (standard specification)
- **Test early, test often**: Geometry bugs found via Blender visualization

### Current Status (End of Week 2)
- **10 files**: Math (3), Geometry (1), Core (2), Scene (2), Output (1), Main (1)
- **30.3 KB memory**: For 16 patches, 1502 triangles (10×10 walls, 5×5 boxes)
- **All tests passing**: Vector3, Matrix4, MathUtils, IndexedMesh, Patch, Scene, Material
- **Ready for Week 3**: Data structures OptiX-compatible, direct GPU buffer access

## References

- Cohen & Wallace: "Radiosity and Realistic Image Synthesis"
- OptiX 7.x Programming Guide (for WSL2 compatibility)
- Classic Cornell Box specification (original dimensions)
- Physically Based Rendering book (PBR) - Form factor equations
