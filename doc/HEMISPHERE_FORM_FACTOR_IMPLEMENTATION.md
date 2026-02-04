# Hemisphere Form Factor Implementation - Phase 2 Revamp

**Date:** February 3, 2026  
**Status:** Core modules implemented, OptiX integration pending

---

## Overview

This document describes the **complete replacement** of the old centroid-based form factor approach with a new **hemispherical ray casting method** using OptiX and Monte Carlo sampling.

### Why Replace?

The old approach had critical flaws:
- Centroid-to-centroid approximation caused `1/r²` instability
- Cross-shaped artifacts from pairwise geometry
- Form factor normalization violated reciprocity
- Solver failed to converge reliably

### New Approach: Hemisphere Ray Casting

For each patch `i`, we cast rays over the positive hemisphere and accumulate hits:

```
F[i,j] ≈ (# hits from i to j) / N_rays
```

**Key advantages:**
- Visibility implicitly included (no separate visibility matrix)
- No `1/r²` singularities
- Energy conservative: `Σ_j F[i,j] ≤ 1`
- Self-hits automatically filtered

---

## File Structure

### Backups Created (.bak)
✅ `src/radiosity/FormFactor.h.bak`  
✅ `src/radiosity/Solver.h.bak`  
✅ `src/gpu/optix_kernels.cu.bak`

### New Core Modules

#### 1. **HemisphereSampling.h** - Sampling Utilities
**Path:** `src/math/HemisphereSampling.h`

**Functions:**
- `cosineWeightedHemisphere(u)` - Generate direction with PDF = cos(θ)/π
- `hammersley(i, N)` - Low-discrepancy 2D sequence
- `buildOrthonormalBasis(N)` - Construct {T, B, N} from normal
- `localToWorld(...)` - Transform local direction to world space
- `barycentricSample(...)` - Sample origin points on triangle

**Status:** ✅ Complete

---

#### 2. **HemisphereFormFactorKernels.cu** - OptiX Kernel
**Path:** `src/gpu/HemisphereFormFactorKernels.cu`

**Programs:**
- `__raygen__hemisphere_formfactor` - Cast hemisphere rays per patch
- `__closesthit__hemisphere` - Record hit primitive index
- `__anyhit__hemisphere` - Filter self-intersections
- `__miss__hemisphere` - Handle rays that escape

**Launch configuration:**
- Dimensions: `[numTriangles, originSamples * dirSamples]`
- Thread `(i, r)` computes one ray for patch `i`
- Atomic accumulation to `formFactors[i * N + j]`

**Status:** ✅ Complete (needs OptiX pipeline integration)

---

#### 3. **FormFactorCache.h** - CSR Storage Format
**Path:** `src/radiosity/FormFactorCache.h`

**Data structure:**
```cpp
struct FormFactorCSR {
    uint32_t numPatches;
    uint32_t raysPerPatch;
    std::string profile;
    
    std::vector<uint32_t> rowPtr;  // [N+1]
    std::vector<uint32_t> colIdx;  // [nnz]
    std::vector<float> values;     // [nnz]
}
```

**Functions:**
- `loadCSR()` / `saveCSR()` - Binary cache I/O
- `getCacheFilename(profile, O, D)` - Generate cache path
- `denseFlatToCSR()` - Convert OptiX output to CSR
- `validateCSR()` - Check row sums, reciprocity, self-hits

**Cache format:**
```
output/cache/ff_hemi_{profile}_O{O}_D{D}_R{R}.csr.bin
```

**Status:** ✅ Complete

---

#### 4. **FormFactorPrecompute.h** - Orchestration
**Path:** `src/radiosity/FormFactorPrecompute.h`

**Main function:**
```cpp
FormFactorCSR computeFormFactors(
    const MeshData& mesh,
    const PatchSoA& patches,
    const HemisphereConfig& config,
    OptiXContext::Context& optixContext
);
```

**Profile configurations:**
- **LOW:** O=1, D=256 → R=256 rays/patch
- **MEDIUM:** O=1, D=512 → R=512 rays/patch
- **HIGH:** O=4, D=512 → R=2048 rays/patch

**Workflow:**
1. Check cache (if `--ffcache`)
2. Allocate GPU buffers
3. Launch OptiX hemisphere kernel
4. Download results
5. Convert to CSR
6. Validate and save

**Status:** ⚠️ Skeleton complete, needs OptiX pipeline integration

---

#### 5. **Solver.h** - Updated for CSR
**Path:** `src/radiosity/Solver.h`

**Changes:**
- ✅ Now accepts `FormFactorCSR` instead of dense matrix
- ✅ Iterates over sparse row using `rowPtr[i]...rowPtr[i+1]`
- ✅ Removed visibility matrix multiplication (implicit in form factors)
- ✅ Old dense interface kept as `progressiveRefinement_OLD()`

**Algorithm (unchanged):**
```
Find patch p with max unshot radiosity
For each j in row p (sparse):
    ΔB_j = ρ_j * F[p,j] * Bu_p
    B_j += ΔB_j
    Bu_j += ΔB_j
Bu_p = 0
```

**Status:** ✅ Complete

---

#### 6. **FormFactor.h** - Deprecated
**Path:** `src/radiosity/FormFactor.h`

**Status:** ⚠️ Marked deprecated, kept for reference
- Old function renamed to `computeFormFactors_OLD()`
- Header comment directs to new system

---

## CMakeLists.txt Updates

✅ Added `HemisphereSampling.h` to headers  
✅ Added `FormFactorCache.h`, `FormFactorPrecompute.h`, `Solver.h` to headers  
✅ Added `HemisphereFormFactorKernels.cu` to CUDA sources  
✅ Created separate PTX compilation for hemisphere kernel  
✅ Added source directory to CUDA include path

**Build targets:**
- `optix_kernels.ptx` - Old visibility kernel
- `hemisphere_kernels.ptx` - New form factor kernel

---

## Integration Checklist

### Phase 1: OptiX Pipeline Extension (TODO)

The OptiXContext needs to support **two pipelines**:

1. **Visibility pipeline** (existing) - `optix_kernels.ptx`
2. **Hemisphere pipeline** (new) - `hemisphere_kernels.ptx`

**Required changes to OptiXContext.h:**

```cpp
class Context {
    // Add hemisphere pipeline members
    OptixModule hemisphereModule = nullptr;
    OptixProgramGroup hemisphereRaygenPG = nullptr;
    OptixProgramGroup hemisphereMissPG = nullptr;
    OptixProgramGroup hemisphereHitgroupPG = nullptr;
    OptixPipeline hemispherePipeline = nullptr;
    OptixShaderBindingTable hemisphereSBT = {};
    
    // Add method
    void buildHemispherePipeline(const char* ptxPath);
    void launchHemisphere(uint32_t N, uint32_t R, 
                          const HemisphereLaunchParams& params);
};
```

**Steps:**
1. Load `hemisphere_kernels.ptx`
2. Create program groups for raygen/miss/hit
3. Link pipeline
4. Setup SBT
5. Launch with dims `[N, R]`

---

### Phase 2: Main.cpp Integration (TODO)

**Add command-line flags:**
```cpp
--precompute-ff       Run hemisphere form factor precompute
--ffcache             Use cached form factors (default: true)
--nocache             Force recompute (ignore cache)
--origin-samples O    Origins per patch (default: 1)
--dir-samples D       Directions per origin (default: 256)
```

**Program flow:**
```cpp
// Phase 1: Load mesh, build PatchSoA
MeshData mesh = loadCornellBox(...);
PatchSoA patches = buildPatches(mesh);

// Phase 2: Compute/load form factors
HemisphereConfig config = getProfileConfig(profile);
FormFactorCSR formFactors = computeFormFactors(
    mesh, patches, config, optixContext);

// Phase 3: Solve
progressiveRefinement(patches, formFactors);

// Phase 4: Export
exportToOBJ(patches, "output.obj");
```

---

### Phase 3: Validation & Debug (CRITICAL)

**Must validate before trusting results:**

#### Row sum check
```cpp
for (uint32_t i = 0; i < N; ++i) {
    float S_i = sum_j F[i,j];
    assert(S_i >= 0.0f && S_i <= 1.01f);
}
```

**Expected:**
- Closed Cornell box: `S_i ≈ 0.8-1.0` (most energy hits walls)
- Open box: `S_i < 1.0` (rays escape)
- If `S_i > 1.0`: normalization error or kernel bug

#### Self-hit check
```cpp
for (uint32_t i = 0; i < N; ++i) {
    assert(F[i,i] < 1e-8f);
}
```

**If fails:** Any-hit program not filtering self-intersections

#### Visual debug export
Export `exposure_i = sum_j F[i,j]` as heatmap:
- Blue: low exposure (occluded)
- Red: high exposure (directly facing light)

Use `exportFormFactorDebug()` in FormFactorPrecompute.h

---

## Performance Targets

| Profile | Triangles | Rays/patch | Total rays | Expected time |
|---------|-----------|------------|------------|---------------|
| LOW     | ~100      | 256        | ~25K       | < 1 sec       |
| MEDIUM  | ~1,000    | 512        | ~512K      | < 5 sec       |
| HIGH    | ~10,000   | 2,048      | ~20M       | < 60 sec      |

**Optimization notes:**
- OptiX launch overhead: ~10ms
- Dominated by ray tracing (not CPU)
- Cache reuse critical for iterative workflow

---

## Known Limitations

1. **Sparse accumulation not implemented**
   - Currently requires dense matrix for OptiX output
   - Feasible for LOW/MEDIUM (< 2000 triangles)
   - HIGH profile needs sparse atomic adds (future work)

2. **Single GAS assumption**
   - Assumes `optixGetPrimitiveIndex()` matches patch ID
   - If multi-GAS, need index mapping buffer

3. **Constant sampling tables**
   - Direction samples recomputed per thread
   - Could precompute in `__constant__` memory (minor optimization)

---

## Testing Strategy

### Unit Tests (Manual)

1. **Single triangle hemisphere**
   - Create 1-triangle scene
   - Cast 256 rays
   - Verify all rays miss (S_0 ≈ 0)

2. **Two parallel triangles**
   - Facing triangles at distance d
   - Verify reciprocity: `A_i * F[i,j] ≈ A_j * F[j,i]`

3. **Cornell box (LOW)**
   - 100 triangles
   - Check row sums: `0.5 < S_i < 1.0`
   - Verify light → walls > 0
   - Verify floor → ceiling ≈ 0 (occluded)

### Regression Tests

Compare against **known good reference:**
- Import Cornell box radiosity from other renderer (e.g., PBRT)
- Compare final vertex colors (expect < 10% difference)

---

## Troubleshooting Guide

### Problem: All F[i,j] = 0
**Cause:** OptiX kernel not launching or SBT misconfigured  
**Fix:** Add printf to raygen, verify launch dims

### Problem: Row sums > 1.5
**Cause:** Monte Carlo estimator wrong (PDF mismatch)  
**Fix:** Check `cosineWeightedHemisphere()` implementation

### Problem: F[i,i] > 0 (self-hits)
**Cause:** Any-hit not filtering or epsilon too small  
**Fix:** Increase `sceneEpsilon`, verify payload passing

### Problem: Solver diverges
**Cause:** Form factors invalid (not validated)  
**Fix:** Run `validateCSR()` before solver

---

## References

- **Pharr et al., "Physically Based Rendering" (4th ed.)** - Monte Carlo sampling
- **Cohen & Wallace, "Radiosity and Realistic Image Synthesis"** - Form factor theory
- **Dutré et al., "Advanced Global Illumination"** - Hemisphere integration
- **OptiX Programming Guide 9.x** - Ray tracing API

---

## Next Steps (Priority Order)

1. ✅ **Backups created**
2. ✅ **Core modules implemented**
3. ⚠️ **OptiX pipeline integration** (CRITICAL BLOCKER)
4. ⚠️ **Main.cpp integration** (CLI flags + workflow)
5. ⚠️ **Validation tests** (row sums, self-hits)
6. ⚠️ **First working render** (LOW profile)
7. 🔲 **Sparse accumulation** (for MEDIUM/HIGH)
8. 🔲 **Optimization** (constant memory, caching)
9. 🔲 **Documentation** (user guide, examples)

---

## Appendix: File Locations

```
src/
├── math/
│   └── HemisphereSampling.h          ✅ NEW
├── gpu/
│   ├── OptiXContext.h                 ⚠️ NEEDS UPDATE
│   ├── optix_kernels.cu              ✅ BACKED UP
│   └── HemisphereFormFactorKernels.cu ✅ NEW
└── radiosity/
    ├── FormFactor.h                   ⚠️ DEPRECATED
    ├── FormFactor.h.bak               ✅ BACKUP
    ├── FormFactorCache.h              ✅ NEW
    ├── FormFactorPrecompute.h         ✅ NEW
    ├── Solver.h                       ✅ UPDATED
    └── Solver.h.bak                   ✅ BACKUP
```

---

**Implementation by:** GitHub Copilot (Claude Sonnet 4.5)  
**Based on:** Hemispherical Ray Casting Coding Plan
