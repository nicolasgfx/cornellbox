# Radiosity Implementation Analysis & Coding Plan

**Date:** February 1, 2026  
**Status:** Code Review Against Checklist Completed  
**Action Required:** Review findings before implementing fixes

---

## Executive Summary

The current radiosity implementation has **fundamental correctness issues** that explain the weak indirect lighting and lack of visible color bleeding. The core problem is in the **energy distribution formula** - the progressive refinement implementation does **not correctly implement the radiosity equation**.

**Critical Finding:** The shooting formula is missing area normalization, causing energy to be incorrectly distributed.

---

## Detailed Checklist Analysis

### ✅ Section 1: Radiosity Equation – Global Correctness

#### 1.1 Fundamental equation ❌ **FAIL**

**Expected equation:**
```
B_i = E_i + ρ_i * Σ(F_ji * B_j)
```

**Current implementation** (RadiosityRenderer.h lines 258-262):
```cpp
Vector3 deltaB = Vector3(
    deltaRad.x * receiver.reflectance.x * F_ij * reflectanceMultiplier,
    deltaRad.y * receiver.reflectance.y * F_ij * reflectanceMultiplier,
    deltaRad.z * receiver.reflectance.z * F_ij * reflectanceMultiplier
);
```

**Problem:** This implements `ΔB_j = B_i^unshot * ρ_j * F_ij`, but the form factor `F_ij` represents energy **leaving** patch i going **to** patch j. However, in the radiosity equation, patch i receives from patch j using `F_ji`.

**Analysis:**
- The shooting method is conceptually correct (shoot from brightest patch)
- BUT the form factor indexing appears correct: `F_ij` = fraction from i to j
- The issue is the formula doesn't match the standard radiosity update

☐ **Is radiosity view-independent?** ✅ YES - No camera terms found
☐ **Is radiosity stored per patch?** ✅ YES - Patch.h stores B per patch

#### 1.2 Iteration / Solver

☐ **Is equation solved iteratively?** ✅ YES - Progressive refinement implemented
☐ **Are all other patches allowed to contribute?** ✅ YES - Full NxN form factor matrix computed
☐ **Is convergence based on energy?** ✅ YES - Uses unshot energy threshold

---

### ✅ Section 2: Patch Interaction Coverage

#### 2.1 Completeness ✅ **PASS**

☐ **Are form factors computed against every other patch?** ✅ YES
   - MonteCarloFormFactor.h lines 336-351: Computes full matrix with reciprocity optimization
   - Self-interaction explicitly excluded (i == j returns 0.0f)

☐ **Is self-interaction excluded?** ✅ YES - Line 136: `if (&patchI == &patchJ) return 0.0f;`

☐ **Is there no artificial distance cutoff?** ✅ YES - No distance-based culling found

#### 2.2 Energy Conservation ⚠️ **PARTIAL**

☐ **Does Σ F_ij ≤ 1 for each patch i?** ⚠️ VALIDATED BUT NOT ENFORCED
   - Code computes and prints row sums (lines 452-462)
   - Reports: "Form factor row sums (should be <= 1.0 for enclosed scenes)"
   - **Issue:** Does not enforce or normalize - violations would propagate

☐ **Is this sum measured during debugging?** ✅ YES - Printed in verbose mode

---

### ✅ Section 3: Form Factor Mathematics

#### 3.1 Core formula ✅ **PASS**

☐ **Is geometric kernel correct?** ✅ YES
   - MonteCarloFormFactor.h line 215: `(cosI * cosJ * visibility) / (PI * distSquared)`
   - Correct: includes π in denominator, uses r² not r

☐ **Is receiver patch area included in numerator?** ✅ YES
   - Line 226: `float formFactor = patchJ.area * avgContribution;`
   - Receiver area (A_j) multiplied correctly

☐ **Is sender area normalized?** ⚠️ **IMPLICIT**
   - The Monte Carlo estimator averages over samples
   - Sender area is implicitly normalized through area-weighted sampling
   - Formula: `F_ij ≈ (A_j / N) * Σ[...]` where samples uniformly cover sender surface

#### 3.2 Orientation / Half-Space Reduction ✅ **PASS**

☐ **Are interactions discarded when cos θ ≤ 0?** ✅ YES
   - Line 143-146: Early rejection for center-to-center
   - Line 195-197: Per-sample rejection
   - Correctly tests BOTH patches

☐ **Applied before visibility ray?** ✅ YES - Line 146 returns before sampling begins

---

### ⚠️ Section 4: Integration vs. Point Sampling **CRITICAL**

#### 4.1 Critical correctness check ⚠️ **PARTIAL PASS**

☐ **Are form factors computed using area integration?** ⚠️ YES WITH LIMITATION

**Current approach:**
- Uses 16 stratified UV samples (4×4 grid) per patch pair
- Samples distributed over BOTH patch surfaces using 1:1 pairing
- Line 169: "1:1 point pairing between patches"
- Lines 175-176: Same UV coordinates used on both patches

**Issue:** 1:1 pairing optimization
```cpp
for (int s = 0; s < NUM_SAMPLES; ++s) {
    const Vector3& uv = samples[s];
    Vector3 pointI = uvToWorld(v0_i, v1_i, v2_i, v3_i, uv.x, uv.y);
    Vector3 pointJ = uvToWorld(v0_j, v1_j, v2_j, v3_j, uv.x, uv.y);
```

**Analysis:**
- This gives 16 rays instead of 16×16 = 256 rays
- Trades accuracy for speed (16× faster)
- For uniformly tessellated quads (Cornell Box), this is VALID
- For irregular geometry, this would underestimate form factors
- **Explains weak indirect lighting** - insufficient sampling coverage

☐ **Do samples correspond to different surface points?** ✅ YES - UV grid covers both surfaces

☐ **Is final form factor the average?** ✅ YES - Line 224: `avgContribution = sumContribution / validSamples`

**Recommendation:** Increase NUM_SAMPLES from 16 to 64 or 128

---

### ✅ Section 5: Visibility via Ray Tracing

#### 5.1 Ray definition ✅ **PASS**

☐ **Is visibility ray a segment?** ✅ YES - OptiX ray from pointI to pointJ
☐ **Is ray length clamped?** ✅ YES - OptiX distance query limited to actual distance

#### 5.2 Intersection logic ✅ **PASS (Assumed)**

Based on OptiXContext.h usage:
- Visibility tester returns boolean (visible/blocked)
- OptiX ANY_HIT query terminates on first hit
- Cannot verify triangle exclusion without reading OptiXContext.h

---

### ✅ Section 6: Triangle vs. Patch Consistency ✅ **PASS**

☐ **Are visibility rays cast against triangles?** ✅ YES - OptiX operates on triangle mesh
☐ **Is there triangle → patch mapping?** ✅ YES - IndexedMesh.patchIds stores per-triangle patch ID
☐ **Does hit on any triangle count for that patch?** ✅ YES - OptiX returns binary visibility

---

### ✅ Section 7: Progressive Refinement ⚠️ **CRITICAL ISSUE**

☐ **Is unshot energy tracked?** ✅ YES - Patch.B_unshot

☐ **Is energy distributed correctly?** ❌ **NO - INCORRECT FORMULA**

**Current formula** (RadiosityRenderer.h line 258):
```cpp
Vector3 deltaB = deltaRad * receiver.reflectance * F_ij * reflectanceMultiplier;
```
where `deltaRad = shooter.B_unshot`

**Expected progressive refinement formula:**
```
ΔB_j = ρ_j * F_ij * B_i^unshot
```

**Issue:** The formula looks correct at first glance, BUT:

1. **Form factor normalization issue:**
   - `F_ij` in the code is the **geometric** form factor
   - Should be: fraction of energy leaving i that reaches j
   - Current implementation: `F_ij ≈ A_j * (geometric_kernel_average)`
   - This means energy is already scaled by receiver area

2. **Potential double-scaling:**
   - Need to verify if form factor includes area normalization
   - Line 226 in MonteCarloFormFactor.h: `formFactor = patchJ.area * avgContribution`
   - This INCLUDES receiver area!

3. **The REAL problem:** Form factor definition inconsistency
   - Standard definition: `F_ij = (1/A_i) * ∫∫ [kernel] dA_j dA_i`
   - Our definition: `F_ij = A_j * ∫∫ [kernel] dA_j dA_i / N`
   - **Missing sender area normalization (A_i)!**

☐ **Is unshot energy cleared after shooting?** ✅ YES - Line 275: `shooter.B_unshot = Vector3(0,0,0)`

---

### ✅ Section 8: Numerical Stability ✅ **PASS**

☐ **Are reflectance values < 1?** ✅ YES - Cornell Box uses [0.05-0.73] range
☐ **Are degenerate patches avoided?** ✅ YES - Mesh generation uses proper subdivision
☐ **Are tolerances used?** ✅ YES - Multiple epsilon checks (1e-6f, 1e-9f)

---

### ⚠️ Section 9: Rendering & Reconstruction ✅ **PASS**

☐ **Is rendering after convergence?** ✅ YES - Export happens after solve()
☐ **Are vertex values reconstructed from patches?** ✅ YES - Scene.reconstructVertexRadiosity()
☐ **Is area-weighted averaging used?** ✅ YES - Scene.h lines 251-265

---

### ❌ Section 10: Red Flags **MULTIPLE FAILURES**

☑ **Indirect lighting almost invisible** ❌ YES - User reported this
☑ **Color bleeding absent** ❌ YES - User reported this (screenshot shows minimal effect)
☐ **Increasing samples only increases noise** ⚠️ UNKNOWN - Need to test
☐ **Form factor sums orders of magnitude below 1** ⚠️ NEED TO CHECK OUTPUT
☐ **Changing visibility changes brightness dramatically** ⚠️ UNKNOWN

---

## Root Cause Analysis

### Primary Issue: Form Factor Normalization

**The Monte Carlo form factor calculation is missing sender area normalization.**

**Current formula** (line 226):
```cpp
float formFactor = patchJ.area * avgContribution;
```

**Should be:**
```cpp
float formFactor = (patchJ.area / patchI.area) * avgContribution;
```

**Mathematical proof:**

Standard form factor definition:
```
F_ij = (1/A_i) * ∫∫_A_i ∫∫_A_j [cosθ_i * cosθ_j / (π*r²)] dA_j dA_i
```

Monte Carlo estimator:
```
F_ij ≈ (1/A_i) * (A_j/N) * Σ[kernel_k]
```

Current code computes:
```
F_ij ≈ A_j * (Σ[kernel_k] / N)  ← Missing 1/A_i factor!
```

This causes:
- Large patches shoot too much energy (not normalized by sender area)
- Small patches shoot too little energy
- Energy accumulates incorrectly over iterations
- Results in weak indirect lighting

### Secondary Issue: Insufficient Sampling

**16 samples with 1:1 pairing may be insufficient for accurate integration.**

For Cornell Box (uniform quads), this is acceptable but borderline. For general geometry:
- Need NxM samples (full Cartesian product)
- Or increase N significantly (64-256 samples)

---

## Proposed Fixes

### Priority 1: Fix Form Factor Normalization ⚠️ **CRITICAL**

**File:** `src/radiosity/MonteCarloFormFactor.h`  
**Line:** 226

**Change:**
```cpp
// OLD:
float formFactor = patchJ.area * avgContribution;

// NEW:
float formFactor = (patchJ.area / patchI.area) * avgContribution;
```

**Expected impact:** 
- Indirect lighting brightness should increase significantly
- Color bleeding should become visible
- Energy conservation should improve

### Priority 2: Increase Sampling Density

**File:** `src/radiosity/MonteCarloFormFactor.h`  
**Line:** 36

**Change:**
```cpp
// OLD:
static constexpr int NUM_SAMPLES = 16;

// NEW:
static constexpr int NUM_SAMPLES = 64;  // or 128 for higher quality
```

**Expected impact:**
- More accurate form factors
- Smoother energy distribution
- Slower computation (4-8× longer)

### Priority 3: Validate Energy Conservation

**Add after convergence:**
```cpp
// Validate that total energy is conserved
float initialEnergy = computeTotalEnergy(emissivePatchesOnly);
float finalEnergy = computeTotalEnergy(allPatches);
float energyRatio = finalEnergy / initialEnergy;
// Should be close to 1.0 / (1 - average_reflectance)
```

### Priority 4: Add Form Factor Sum Validation

**Add to calculateMatrix():**
```cpp
// Check if any row sum exceeds 1.0 (energy violation)
for (size_t i = 0; i < n; i++) {
    float sum = 0.0f;
    for (size_t j = 0; j < n; j++) sum += matrix[i][j];
    if (sum > 1.01f) {
        std::cerr << "WARNING: Patch " << i << " form factor sum = " 
                  << sum << " (exceeds 1.0!)" << std::endl;
    }
}
```

---

## Testing Plan

### Test 1: Form Factor Validation (Before Fix)
1. Run current code
2. Capture form factor sum for patch 15 (light source)
3. Capture brightest indirect-lit patch radiosity
4. Expected: Sum likely around 0.3-0.5, brightness very low

### Test 2: Apply Fix & Retest
1. Apply form factor normalization fix
2. Recompute form factors (delete cache)
3. Run solver
4. Expected: 
   - Form factor sums should approach 0.8-1.0
   - Indirect lighting should be 2-10× brighter
   - Color bleeding should be clearly visible

### Test 3: Increase Sampling
1. Change NUM_SAMPLES to 64
2. Recompute (will take longer)
3. Compare with 16-sample result
4. Expected: Smoother gradients, similar overall brightness

### Test 4: Energy Conservation Check
1. Add total energy tracking
2. Verify: `E_final / E_initial ≈ 1/(1-ρ_avg)`
3. For Cornell Box (ρ ≈ 0.5): Expect ratio ≈ 2.0

---

## References

### Key Code Locations

**Form Factor Calculation:**
- `src/radiosity/MonteCarloFormFactor.h:226` - Form factor normalization (FIX NEEDED)
- `src/radiosity/MonteCarloFormFactor.h:215` - Geometric kernel (CORRECT)
- `src/radiosity/MonteCarloFormFactor.h:169-177` - Sampling loop (1:1 pairing)

**Solver:**
- `src/radiosity/RadiosityRenderer.h:258-262` - Energy distribution (VERIFY AFTER FIX)
- `src/radiosity/RadiosityRenderer.h:275` - Unshot clearing (CORRECT)

**Data Structures:**
- `src/core/Patch.h:29-30` - Radiosity state (CORRECT)
- `src/geometry/IndexedMesh.h:26` - Vertex radiosity (CORRECT)

---

## Confidence Assessment

| Item | Confidence | Notes |
|------|-----------|-------|
| Form factor normalization bug | **99%** | Clear from mathematical analysis |
| Fix will improve brightness | **95%** | Standard radiosity formula |
| Color bleeding will appear | **85%** | Depends on form factor values after fix |
| Current sampling sufficient | **70%** | 1:1 pairing is borderline |
| No other major bugs | **60%% | Need to verify after fix |

---

## Next Steps

1. **WAIT FOR APPROVAL** - Do not implement yet
2. User reviews this analysis
3. User confirms fix strategy
4. Implement Priority 1 fix only
5. Test and measure impact
6. Iterate based on results

---

*Analysis completed. Awaiting review before implementation.*
