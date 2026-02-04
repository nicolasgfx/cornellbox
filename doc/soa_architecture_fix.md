# SoA Architecture Fix - Phase 2 Refactoring

## Problem Identified

The initial Phase 2 implementation violated the **Structure-of-Arrays (SoA)** design principle specified in the coding plan:

- **Violation:** OptiX kernels were accessing `vertices[]` and `indices[]` arrays (AoS layout)
- **Impact:** Scattered memory access, poor cache utilization, wasted bandwidth
- **Root Cause:** Fell back to standard OptiX patterns without following radiosity-specific design

## Architectural Principle: Two Worlds

The system separates geometry into two distinct worlds with different memory access patterns:

### World A: Traversal (OptiX-managed)
- **Data:** `vertices[]`, `indices[]`
- **Layout:** Array-of-Structures (AoS) - acceptable
- **Usage:** OptiX GAS building and ray traversal **only**
- **Access:** Never touched by CUDA compute kernels
- **Why AoS is OK:** OptiX internally restructures and optimizes for hardware traversal

### World B: Compute (CUDA-streamed)
- **Data:** PatchSoA arrays (`cx, cy, cz, nx, ny, nz, area, rho, B, etc.`)
- **Layout:** Structure-of-Arrays (SoA) - **required**
- **Usage:** All radiosity/visibility/energy transport computation
- **Access:** Linear streaming by CUDA kernels
- **Why SoA is required:** Coalesced memory access across thread warps

## Bridge Between Worlds

The two worlds share a common indexing scheme:

```
triangle_index == patch_index
```

This allows:
- OptiX to traverse using its internal GAS representation
- CUDA kernels to read precomputed patch data using the same triangle IDs
- No pointer chasing, no indirection, no topology lookups on GPU

## Implementation Changes

### 1. LaunchParams Struct (optix_kernels.cu)

**Before (WRONG):**
```cuda
struct LaunchParams {
    float3* vertices;  // AoS - scattered access
    uint3* indices;
    float3* normals;
    ...
};
```

**After (CORRECT):**
```cuda
struct LaunchParams {
    // SoA layout - coalesced access
    float* cx, *cy, *cz;  // centroids
    float* nx, *ny, *nz;  // normals
    ...
};
```

### 2. Kernel Logic (optix_kernels.cu)

**Before (WRONG):**
```cuda
// Load triangle vertices
uint3 tri_i = params.indices[i];
float3 v0 = params.vertices[tri_i.x];  // scattered reads
float3 v1 = params.vertices[tri_i.y];
float3 v2 = params.vertices[tri_i.z];

// Reconstruct geometry
float3 centroid = (v0 + v1 + v2) / 3.0f;
```

**After (CORRECT):**
```cuda
// Read precomputed patch data (coalesced SoA reads)
float3 centroid_i = make_float3(params.cx[i], params.cy[i], params.cz[i]);
float3 normal_i = make_float3(params.nx[i], params.ny[i], params.nz[i]);
```

### 3. OptiXContext Upload (OptiXContext.h)

**Before (WRONG):**
```cpp
// Upload vertices/indices for kernel use
CUDA_CHECK(cudaMemcpy(d_vertices, mesh.vertices.data(), ...));
CUDA_CHECK(cudaMemcpy(d_indices, mesh.indices.data(), ...));

// Reconstruct normals
std::vector<float3> normals(numTri);
for (size_t i = 0; i < numTri; ++i) {
    normals[i] = make_float3(patches.nx[i], patches.ny[i], patches.nz[i]);
}
CUDA_CHECK(cudaMemcpy(d_normals, normals.data(), ...));
```

**After (CORRECT):**
```cpp
// === World A: vertices/indices used ONLY for GAS building ===
// (already uploaded in buildGAS() - never touched by compute kernels)

// === World B: Upload PatchSoA arrays directly (SoA layout) ===
CUDA_CHECK(cudaMemcpy(d_cx, patches.cx.data(), floatBytes, ...));
CUDA_CHECK(cudaMemcpy(d_cy, patches.cy.data(), floatBytes, ...));
CUDA_CHECK(cudaMemcpy(d_cz, patches.cz.data(), floatBytes, ...));
CUDA_CHECK(cudaMemcpy(d_nx, patches.nx.data(), floatBytes, ...));
CUDA_CHECK(cudaMemcpy(d_ny, patches.ny.data(), floatBytes, ...));
CUDA_CHECK(cudaMemcpy(d_nz, patches.nz.data(), floatBytes, ...));
```

### 4. Device Memory Management (OptiXContext.h)

**Before (WRONG):**
```cpp
CUdeviceptr d_vertices = 0;
CUdeviceptr d_indices = 0;
CUdeviceptr d_normals = 0;  // redundant reconstruction
```

**After (CORRECT):**
```cpp
// World A (Traversal): vertices/indices for OptiX GAS only
CUdeviceptr d_vertices = 0;
CUdeviceptr d_indices = 0;

// World B (Compute): PatchSoA arrays for CUDA kernels
CUdeviceptr d_cx = 0, d_cy = 0, d_cz = 0;  // centroids
CUdeviceptr d_nx = 0, d_ny = 0, d_nz = 0;  // normals
```

## Performance Impact

### Before (AoS):
- **Scattered reads:** Each warp thread accesses 3 random vertices → no coalescing
- **Redundant computation:** Centroids/normals recomputed in kernel
- **Wasted bandwidth:** Transfer vertices (3×N floats) + indices (3×N ints) + normals (3×N floats)
- **Expected throughput:** ~10× slower than optimal

### After (SoA):
- **Coalesced reads:** Warp threads read consecutive cx[i], cy[i], cz[i] → perfect coalescing
- **Precomputed data:** Centroids/normals computed once on CPU
- **Minimal bandwidth:** Transfer only needed data (6×N floats: cx,cy,cz,nx,ny,nz)
- **Expected throughput:** Near-optimal memory bandwidth utilization

## Validation

The architectural separation is now enforced:

1. ✅ `buildGAS()` uses vertices/indices (World A)
2. ✅ `computeVisibility()` uploads PatchSoA arrays (World B)
3. ✅ Kernel reads from `params.cx/cy/cz/nx/ny/nz` (World B)
4. ✅ No kernel code accesses vertices/indices
5. ✅ Header comments document the separation

## Rule for Future Development

**Critical principle:**

> **No CUDA kernel may iterate over vertex or index buffers.**
> All radiosity, visibility accumulation, and energy transport kernels operate exclusively on PatchSoA arrays.

This ensures:
- Coalesced memory access patterns
- Minimal data transfer
- Maximum GPU throughput
- Architectural correctness

## Files Modified

1. `src/gpu/optix_kernels.cu`
   - Updated LaunchParams to SoA layout
   - Removed triangle reconstruction code
   - Simplified to centroid-to-centroid rays

2. `src/gpu/OptiXContext.h`
   - Added World A/B separation comment
   - Split device pointers (d_vertices/d_indices vs d_cx/cy/cz/nx/ny/nz)
   - Updated computeVisibility() to upload PatchSoA
   - Removed redundant normal reconstruction

## Next Steps

1. **Build and test:** Use `fast_build.ps1` to compile with Ninja
2. **Verify output:** Check cache files and AO visualization
3. **Performance profiling:** Measure bandwidth utilization with Nsight Compute
4. **Extend to Phase 3:** Use same SoA pattern for form factor computation
