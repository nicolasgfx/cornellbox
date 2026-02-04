# Coding Plan – Per-Vertex Radiosity Reconstruction

This document specifies **how to reconstruct per-vertex radiosity values from a per-patch radiosity solution**, suitable for **per-vertex interpolation during rendering**.

It is designed to be:
- handed directly to **Copilot / code-generation tools**
- consistent with **classical radiosity theory**
- simple, robust, and defensible

This step is **purely a reconstruction / visualization stage** and does **not** affect the radiosity solve itself.

---

## 1. Conceptual Background

### 1.1 What radiosity provides

Classical radiosity computes:

- **Radiosity per patch**
- Piecewise-constant energy per unit area
- View-independent lighting

Formally:

\[
B_i = \text{constant over patch } i
\]

Radiosity is **not defined at points** or vertices.

---

### 1.2 Why vertex radiosity is needed

Modern rasterization pipelines:
- interpolate **per-vertex attributes**
- do not support per-patch constants directly

Therefore, a **reconstruction step** is required:

> Reconstruct a smooth scalar/vector field from piecewise-constant patch samples.

This is a numerical approximation, not a physical re-simulation.

---

## 2. Design Principles

### Invariants

- Radiosity solve remains **unchanged**
- Reconstruction happens **after convergence**
- No energy feedback into solver

### Constraints

- Must avoid visible seams
- Must preserve global energy behavior
- Must be stable across mesh resolutions

---

## 3. Data Model Assumptions

### 3.1 Patch

```cpp
struct Patch {
    float area;
    Vec3  normal;
    Vec3  B;     // converged radiosity (RGB)
};
```

---

### 3.2 Vertex

```cpp
struct Vertex {
    Vec3 position;
    Vec3 normal;   // shading normal
    Vec3 Bv;       // reconstructed vertex radiosity
};
```

---

### 3.3 Connectivity

A precomputed mapping:

```cpp
std::vector<PatchID> patchesTouchingVertex[vertexCount];
```

Vertices may belong to multiple patches.

---

## 4. Core Reconstruction Algorithm

### 4.1 Area-Weighted Averaging (Baseline)

For each vertex \(v\):

\[
B_v = \frac{\sum_{i \in P(v)} A_i B_i}{\sum_{i \in P(v)} A_i}
\]

Where:
- \(P(v)\) = patches incident to vertex
- \(A_i\) = area of patch \(i\)
- \(B_i\) = radiosity of patch \(i\)

---

### 4.2 Implementation Sketch

```cpp
void computeVertexRadiosity(int v) {
    Vec3 sum = {0,0,0};
    float wsum = 0.0f;

    for (PatchID p : patchesTouchingVertex[v]) {
        float w = patches[p].area;
        sum  += patches[p].B * w;
        wsum += w;
    }

    vertices[v].Bv = sum / wsum;
}
```

This is executed **once**, after radiosity convergence.

---

## 5. Normal-Aware Refinement (Optional)

### 5.1 Motivation

At sharp corners or creases:
- patches with very different orientations meet
- naive averaging causes light bleeding

---

### 5.2 Orientation Weighting

Introduce an additional weight:

\[
w_i = \max(0, n_i \cdot n_v)
\]

Final reconstruction:

\[
B_v = \frac{\sum_i A_i w_i B_i}{\sum_i A_i w_i}
\]

---

### 5.3 Implementation Sketch

```cpp
float w = patches[p].area * max(0.0f, dot(patches[p].normal, vertices[v].normal));
```

This step is **optional** but recommended for scenes with hard edges.

---

## 6. Vertex Splitting at Sharp Edges (Strongly Recommended)

### 6.1 Rationale

Radiosity should **not bleed across geometric discontinuities**.

This is identical to normal splitting rules in rasterization.

---

### 6.2 Rule

- If angle between adjacent patch normals > threshold (e.g. 45°)
- Split the vertex
- Each surface owns its own vertex instance

This prevents invalid averaging entirely.

---

## 7. Rendering Integration

### 7.1 Attribute Binding

During rasterization:

- per-vertex attribute = `Bv`
- standard barycentric interpolation

No lighting calculation is done during rendering.

---

### 7.2 Debug Modes (Recommended)

Provide at least two rendering modes:

1. **Flat per-patch shading**
2. **Per-vertex interpolated shading**

This allows verification of reconstruction quality.

---

## 8. Numerical & Practical Notes

- Use floating point (Vec3) for radiosity
- Clamp only during final display
- Reconstruction does not affect convergence

---

## 9. Conceptual Summary

- Radiosity is **area-based physics**
- Vertices are **point-based visualization artifacts**
- Reconstruction is a **numerical smoothing step**
- Area-weighted averaging is the standard solution

---

## 10. Scope Boundaries

Explicitly out of scope:

- Re-solving radiosity at vertices
- Higher-order basis functions
- Hierarchical radiosity interpolation

---

*End of per-vertex reconstruction coding plan.*

