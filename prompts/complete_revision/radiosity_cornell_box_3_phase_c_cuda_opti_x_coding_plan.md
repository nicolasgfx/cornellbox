# Radiosity Cornell Box – 3-Phase C++/CUDA/OptiX Coding Plan

This is a **Copilot-oriented coding plan** for an academic radiosity project in **C++**, with optional **CUDA + OptiX** acceleration.  
Primary values: **simplicity, compactness, correctness, debuggability**.

The plan is split into **3 phases** with explicit deliverables.

---

## 0. Guiding Principles

- **Triangles only**. No quads.
- **SoA for per-triangle (“patch”) attributes**. GPU/CPU share the **same memory layout** and use **memcpy** without transformation.
- Geometry for OptiX uses the standard:
  - `float3* vertices`
  - `uint3* indices`
- Radiosity is solved on **triangles** (triangle == patch).
- Cornell box normals:
  - **Box walls:** normals point **inwards**
  - **Inner boxes:** normals point **outwards**
  - Box is **open on one side**.
- Provide 3 geometry performance profiles from day 1:
  - `--low`, `--medium`, `--high`
- Always provide a **visual debug output** as `.obj + .mtl`.

---

## 1. Project Layout (Keep It Simple)

### 1.1 Proposed folder structure

Keep the existing top-level structure, but simplify `/src`.

```
RADIOSITY/
  build/
  doc/
  output/
    scenes/           # generated .obj/.mtl outputs
    cache/            # visibility caches
    logs/
  prompts/
  src/
    main.cpp
    app/              # CLI, config, run modes
    scene/            # loading cornell obj, profile generation, validation
    mesh/             # mesh data + connectivity helpers
    gpu/              # OptiX + CUDA utilities
    visibility/       # visibility pipeline + cache format
    radiosity/        # solver + form factors
    export/           # obj+mtl export, vertex color bake
    math/             # minimal vec3, utils
  third_party/
    optix/
  README.md
  run.sh
  CMakeLists.txt
```

### 1.2 Keep modules small

- Avoid deep nesting.
- Prefer **header-only small utilities** for math.
- No heavy abstractions; simple structs and POD arrays.

---

## 2. Shared Data Model (CPU/GPU Friendly)

### 2.1 Geometry (OptiX-friendly)

Use standard buffers:

```cpp
struct Vertex { float x, y, z; };   // matches float3 layout
struct TriIdx { uint32_t i0, i1, i2; };

std::vector<Vertex> vertices;
std::vector<TriIdx> indices;
```

### 2.2 Patch attributes as SoA (triangle == patch)

Create one entry per triangle (patchCount == indices.size()).

```cpp
struct PatchSoA {
  // geometry-derived
  std::vector<float> area;          // [N]
  std::vector<float> nx, ny, nz;    // [N]
  std::vector<float> cx, cy, cz;    // [N] centroid

  // material / radiosity
  std::vector<float> rho_r, rho_g, rho_b;  // reflectance
  std::vector<float> emit_r, emit_g, emit_b; // emission (light source triangles)

  // solver state
  std::vector<float> B_r, B_g, B_b;         // radiosity
  std::vector<float> Bu_r, Bu_g, Bu_b;      // unshot radiosity (progressive refinement) OR delta buffer

  // ids
  std::vector<uint32_t> tri_id;     // identity mapping (0..N-1), used for debugging
};
```

**Rule:** These arrays are copied to the GPU via raw memcpy, no rearrangement.

### 2.3 Minimal adjacency for vertex color baking

Half-edge is optional (see Phase 2/3). For simplicity, start with:

```cpp
std::vector<std::vector<uint32_t>> vertex_to_tris; // for each vertex, list of incident triangles
```

This is enough to compute area-weighted vertex colors.

---

## 3. CLI + Profiles + Build System

### 3.1 `run.sh`

Support:
- `--low`, `--medium`, `--high`
- `--nocache`
- `--profile visibility|ao|radiosity` (optional but helpful)
- `--rays N` (visibility rays per pair)
- `--samples N` (points per triangle, see Phase 2)

### 3.2 Debug flags

Compile-time:
- `RAD_DEBUG`
- `RAD_VALIDATE`

Runtime:
- `--nocache`
- `--dump-matrix-stats`

### 3.3 CMake

- Build CPU code always.
- Build CUDA/OptiX parts only if CUDA is available.
- Keep a single executable `radiosity`.

---

# PHASE 1 – The Basics (Clean Cornell Box + SoA)

## 1. Goal

- Load the classic Cornell box OBJ (`/mnt/data/cornell_box.obj`) and produce a **single connected triangle mesh**.
- Build SoA patch arrays.
- Enforce correct normals (walls inward, inner boxes outward).
- Generate `.obj + .mtl` output showing colored Cornell box.

## 2. Tasks

### 2.1 OBJ import (minimal)

Implement a minimal OBJ loader that reads:
- `v` positions
- `vn` normals (if available)
- `f` faces
- `usemtl` (optional)

If normals are not provided or unreliable:
- compute face normals from vertex positions.

### 2.2 Mesh validation

Add validation checks:
- indices in range
- no degenerate triangles
- connected mesh check (optional: BFS on triangle adjacency)

### 2.3 Normals orientation enforcement

Implement a rule-based normal fix for Cornell box:

- Identify triangle groups by spatial location (wall planes) using AABB thresholds.
- For each group, ensure normals point the expected direction:
  - walls inward
  - inner boxes outward

Implementation hint:
- For each triangle, compute normal `n`.
- Compare `dot(n, expectedDirection)`.
- If negative, swap two indices to flip winding.

### 2.4 Performance profiles (low/medium/high)

Keep the original Cornell box as baseline.

Define profiles by **tessellation/subdivision**, without changing the overall shape:

- **LOW:** original triangles from OBJ
- **MEDIUM:** subdivide each triangle 2× (4 triangles)
- **HIGH:** subdivide each triangle 4× (16 triangles)

Rules:
- subdivision must preserve mesh connectivity
- material/emission attributes must be propagated
- triangle == patch always

### 2.5 SoA construction

After final triangle list is prepared:
- compute per-triangle `area`, `centroid`, `normal`
- initialize `rho` from materials
- initialize `emit` for ceiling light triangles
- set `B = emit`, `Bu = emit`

### 2.6 Export `.obj + .mtl`

For Phase 1 output:
- assign vertex colors based on wall/box material color (not radiosity)
- export as `output/scenes/cornell_phase1_{profile}.obj`

**Expected outcome:** a Cornell box visible in any 3D viewer.

---

# PHASE 2 – Visibility (Offline OptiX Cache + AO-like Debug)

## 1. Goal

- Compute **triangle-to-triangle partial visibility** `V[i,j] in [0..1]` offline using OptiX.
- Cache the result on disk.
- Visualize visibility as an AO-like value in `.obj` export.

## 2. Visibility definition

For triangle pair (i, j):
- choose `n` pairs of sample points on triangle i and triangle j
- shoot `n` segment rays
- `V[i,j] = (#visible rays) / n`

Visibility is **visibility only**. No cosines, no distance term, no form factor.

## 3. Sampling points (constant, bandwidth-free)

### 3.1 Precomputed sample set

Use a fixed set of barycentric coefficients `(u_k, v_k)` with uniform coverage.

- store `(u_k, v_k)` as **`__constant__`** arrays in the OptiX module
- convert to surface points per triangle:

```cpp
p = v0 + u*(v1-v0) + v*(v2-v0);
```

### 3.2 Stratified patterns

Provide at least 3 patterns:
- 4 samples
- 16 samples
- 64 samples

All patterns must use the standard triangle uniform mapping:
- generate u,v in [0,1]
- reflect if `u+v>1`

For constant patterns, precompute them once (CPU) and paste them into the OptiX source as literal constants.

## 4. OptiX kernel design

### 4.1 Launch shape

Launch one thread per triangle pair block, e.g.
- 2D launch `(i, j)` with `i < j`
- or linear index mapping over pairs

Each thread:
- loads triangle i and j
- loops over `k=0..n-1` sample pairs
- shoots a segment ray for each pair
- accumulates visible count

### 4.2 Correctness rules (must hold)

- use hemisphere check with `dot(n_i, dir) > 0`
- `tmin = 1e-4`
- `tmax = |pB - pA| - eps`
- ignore target triangle (or patch) hits in any-hit
- use `TERMINATE_ON_FIRST_HIT`

### 4.3 Patch/triangle ID mapping

Do **not** use `optixGetPrimitiveIndex()` as patch id unless guaranteed.

Recommended:
- build GAS per triangle mesh
- use SBT data or instance IDs to recover triangle IDs

For a single GAS triangle mesh, primitive index is the triangle index **only if** it matches `indices[]` ordering. Validate this explicitly.

## 5. Cache format

Cache must exist for all 3 profiles.

### 5.1 File naming

```
output/cache/visibility_{profile}_r{rays}_s{samples}.bin
```

### 5.2 Storage

Store **compressed** visibility to reduce size:
- `uint8 V8 = round(V * 255)`

Optionally store a per-pair bitmask for exact per-sample visibility (still “visibility only”):
- for 16 samples: `uint16 mask` (1 bit per sample)
- this enables correct form factor integration later without multiplying averages

### 5.3 Cache validation header

Include header:
- magic
- version
- profile
- N triangles
- samples per pair
- rays per pair

## 6. AO-like debug output

Compute per triangle:

```text
occ[i] = average over j != i of (1 - V[i,j])
```

Or equivalently “visibility score”:

```text
visScore[i] = average over j != i of V[i,j]
```

Normalize to [0..1] and store as per-triangle scalar.

### 6.1 Bake triangle scalar to vertex colors

Use **area-weighted** reconstruction:

```cpp
vertexValue = sum( triValue * triArea ) / sum(triArea)
```

Export as vertex colors in `.obj`.

**Expected outcome:** `.obj` that looks like an AO/visibility heatmap; used to validate OptiX kernel correctness.

---

# PHASE 3 – Radiosity (Energy Transport using Precomputed Visibility)

## 1. Goal

- Compute radiosity in RGB using iterative energy transfer.
- Use the cached visibility results as occlusion input.
- Export final radiosity colored Cornell box.

## 2. Form factor computation (triangle-based)

### 2.1 What must be computed

Form factor approximation between triangle i and j:

\[
F_{ij} \approx \frac{1}{A_i} \sum_{k} \frac{\cos\theta_i^{(k)} \cos\theta_j^{(k)}}{\pi r_k^2} \; V_k \; \Delta A
\]

Important: visibility multiplies the kernel **per sample**.

### 2.2 Two supported options

#### Option A (preferred): per-sample visibility mask

If Phase 2 caches a **bitmask** per pair:
- compute geometric kernel per sample
- multiply by `maskBit(k)`
- average

This is correct and still keeps ray tracing “offline”.

#### Option B (simpler baseline): averaged visibility

If Phase 2 caches only `V[i,j]` as a scalar:
- compute an averaged geometric kernel `Kavg[i,j]` (using the same sample pairs, but without rays)
- approximate `F[i,j] ≈ V[i,j] * Kavg[i,j] * Aj/Ai`

This is an approximation; document it as such.

### 2.3 Distance softening (optional)

If form factors are too small, support effective distance:

```text
r2_eff = r2 + alpha * Aj
alpha ~ 0.5
```

Enable as debug toggle.

## 3. Solver

Implement progressive refinement (recommended for clarity and convergence):

1. Initialize:
- `B = emit`
- `Bu = emit`

2. Repeat until convergence:
- pick patch `p` with max `|Bu[p]|`
- for all j:
  - `B[j] += rho[j] * F[p,j] * Bu[p]`
  - `Bu[j] += rho[j] * F[p,j] * Bu[p]`
- set `Bu[p] = 0`

Stop condition:
- `max(|Bu|) < epsilon`

Keep RGB channels separate (SoA arrays).

## 4. Post-processing: vertex colors

After convergence:
- compute per-triangle RGB = `B_r/g/b`
- bake to vertex colors using area-weighted adjacency

Export as:

```
output/scenes/cornell_phase3_{profile}.obj
output/scenes/cornell_phase3_{profile}.mtl
```

**Expected outcome:** radiosity Cornell box with visible indirect light and color bleeding.

---

## 4. Cross-Phase Validation & Tests

Add these tests early and keep them enabled:

### 4.1 Form factor bounds

For each i:
- verify `sum_j F[i,j] <= 1 + eps`

### 4.2 Reciprocity (area-weighted)

For random pairs (i,j):

```text
Ai * Fij ≈ Aj * Fji
```

Use tolerance based on sampling.

### 4.3 Visibility symmetry (optional)

Geometric visibility is symmetric, but sampled visibility may differ slightly.
- Do not assume perfect equality unless sampling is symmetric.

### 4.4 Visual debug modes

Always keep export modes:
- Phase1: material colors
- Phase2: visibility/AO heatmap
- Phase3: final radiosity

---

## 5. Implementation Notes (keep it academic)

- Avoid complex mesh libraries.
- Keep kernels small and deterministic.
- Prefer readable reference implementations over micro-optimizations.
- Coalescing: SoA arrays for patch attributes; contiguous indices.
- Avoid half-edge initially; add later only if you need crease-aware vertex splitting or mesh refinement.

---

*End of coding plan.*

