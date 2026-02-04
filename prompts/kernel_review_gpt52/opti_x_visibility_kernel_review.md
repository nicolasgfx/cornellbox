# OptiX Visibility Kernel – Detailed Review

This document is a **structured, Copilot‑friendly review** of your current `__raygen__visibility` kernel in `optix_kernels.cu`, evaluated **strictly against the previously defined OptiX kernel plan** (data structures, sampling, occlusion logic, robustness, and performance).

It is written so you can:
- drop it into Copilot / GitHub Copilot Chat
- reference it while refactoring
- turn sections directly into TODOs or PR checklist items

---

## 1. Critical correctness issue: LaunchParams access

### ❌ Current state
```cpp
const LaunchParams& params = *(const LaunchParams*)optixGetSbtDataPointer();
```

This is **incorrect**.
- `optixGetSbtDataPointer()` returns the **SBT record data**, not launch params.
- This leads to undefined behavior and invalid memory reads.

### ✅ Required fix (OptiX 7 standard)

Device side:
```cpp
extern "C" {
__constant__ LaunchParams params;
}
```

Usage:
```cpp
extern "C" __global__ void __raygen__visibility() {
    const uint3 idx = optixGetLaunchIndex();
    // use params.xxx directly
}
```

Host side:
- Upload `LaunchParams` to a device buffer
- Pass it as the `params` argument to `optixLaunch`

---

## 2. Triangle sampling is mathematically incorrect

### ❌ Current state
You treat `(u,v)` in `[0,1]²` directly as barycentrics:
```cpp
float w = 1.0f - u - v;
```

This samples **outside the triangle** whenever `u+v > 1`, causing:
- invalid ray origins
- negative barycentric weights
- biased visibility results

### ✅ Correct square → triangle mapping (uniform area)
```cpp
__device__ inline float3 sampleTriangle(
    const float3& v0,
    const float3& v1,
    const float3& v2,
    float2 xi)
{
    float su = sqrtf(xi.x);
    float b0 = 1.0f - su;
    float b1 = su * (1.0f - xi.y);
    float b2 = su * xi.y;
    return v0*b0 + v1*b1 + v2*b2;
}
```

Use this for both triangles A and B.

---

## 3. Sample tables are broken / biased

### ❌ Issues
- Duplicate entries in `SAMPLES_4`
- Repeated and correlated entries in `SAMPLES_16`
- No stratification or scrambling

This leads to:
- poor convergence
- structured artifacts
- wasted rays

### ✅ Recommended replacement

Use a **low‑discrepancy sequence** (Hammersley or Halton) with per‑pair scrambling:

```cpp
float2 xi = hammersley(sampleIdx, numSamples);
xi = frac(xi + hashToFloat2(pairId)); // Cranley–Patterson rotation
```

Benefits:
- deterministic
- no tables
- much better convergence per ray

---

## 4. Occlusion logic lacks explicit self/target rejection

### ❌ Current state
- Uses `tmin = 1e‑4`
- Uses `tmax = dist - 1e‑4`
- Relies on closest‑hit only

This is **not robust**:
- grazing rays still self‑intersect
- numerical precision varies with scene scale

### ✅ Correct OptiX pattern (recommended)

Use an **any‑hit program** for visibility rays:

- Enable:
  - `OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT`
  - `OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT`
- In `anyhit`:
  - If hit primitive == triA or triB → `optixIgnoreIntersection()`
  - Else → mark blocked and `optixTerminateRay()`

This guarantees:
- no self‑occlusion
- target triangle never blocks itself
- early exit on first occluder

---

## 5. Hemisphere checks are incomplete

### ❌ Current state
```cpp
if (dot(nA, dir) <= 0) continue;
```

You only test triangle A.

### ✅ Correct front‑to‑front visibility test
```cpp
if (dot(nA, dir) <= 0) continue;
if (dot(nB, -dir) <= 0) continue;
```

### Policy decision (must be explicit)
- **Skip invalid samples** → divide by `validSamples`
- **Count as occluded** → divide by `numSamples`

Both are valid, but must be consistent.

---

## 6. Epsilon handling is not scale‑aware

### ❌ Current state
```cpp
tmin = 1e-4f;
tmax = dist - 1e-4f;
origin = pA;
```

Problems:
- fails for large scenes
- fails for tiny geometry
- no origin offset

### ✅ Robust epsilon strategy
```cpp
float eps = max(baseEps, 1e-4f * dist);
float3 origin = pA + nA * normalOffset;
float tmin = eps;
float tmax = dist - eps;
```

If `dist <= 2*eps`:
- treat as fully visible
- or mark sample invalid

---

## 7. Visibility normalization is incorrect

### ❌ Current state
You `continue` on invalid samples but still divide by `numSamples`.

This biases results downward depending on geometry orientation.

### ✅ Correct normalization
```cpp
visibility = unoccluded / validSamples;
visibility = clamp(visibility, 0.0f, 1.0f);
```

Handle `validSamples == 0` explicitly.

---

## 8. Launch strategy is catastrophically inefficient

### ❌ Current state
- Launch grid: `numTriangles × numTriangles`
- Early‑return when `i >= j`

This still schedules **O(N²)** threads.

### ✅ Correct strategy (mandatory)

Precompute candidate triangle pairs:
```cpp
struct TriPair {
  uint32_t triA;
  uint32_t triB;
};
```

Then:
- Upload `TriPair[]` to GPU
- Launch 1D grid: `launchDim.x = numPairs`
- One thread = one pair

This is the **single biggest performance win**.

---

## 9. Global triangle IDs (multi‑mesh correctness)

Reminder:
- `optixGetPrimitiveIndex()` is **local to the GAS**

If using multiple meshes or IAS:
- Store `baseTriangleIndex` per instance in SBT hitgroup data
- Compute:
```cpp
globalTriId = base + optixGetPrimitiveIndex();
```

Required for correct self/target exclusion.

---

## 10. Minimal safe refactor order

1. Fix launch params access
2. Fix triangle sampling
3. Fix normalization (`validSamples`)
4. Add scale‑aware epsilon + normal offset
5. Switch to any‑hit occlusion logic
6. Replace NxN launch with pair list

---

## Final verdict

**The kernel is conceptually correct**, but:
- currently numerically fragile
- biased in sampling
- extremely inefficient at scale

After applying the changes above, the kernel will:
- return stable `[0..1]` visibility
- behave correctly under all orientations
- scale to large meshes
- be fully OptiX‑idiomatic

---

If needed, this document can be turned into:
- a Copilot refactor prompt
- a PR checklist
- a line‑by‑line annotated kernel rewrite

