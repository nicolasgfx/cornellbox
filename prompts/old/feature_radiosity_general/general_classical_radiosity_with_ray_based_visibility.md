# Coding Plan: Classical Radiosity Renderer (Theory‑First, Ray‑Based Visibility)

This document is a **technical coding plan** that consolidates all architectural, mathematical, and implementation decisions discussed so far.

It is written to be **handed directly to Copilot or another code‑generation assistant**.  
The plan is explicit, modular, and intentionally conservative in scope.

The focus is **classical radiosity**, not path tracing.

---

## 0. Design Goals & Constraints

### Goals
- Implement **classical diffuse radiosity**
- Use **explicit form factors**
- Use **iteration / progressive refinement** for energy transfer
- Use **ray casting for visibility** (not hemicube)
- Support **acceleration via OptiX (optional backend)**
- Keep math and structure clear and inspectable

### Explicit Non‑Goals
- No specular reflection
- No glossy BRDFs
- No Monte Carlo path tracing
- No participating media
- No wavelength‑dependent transport (RGB only)

---

## 1. High‑Level Architecture

The system is split into **four orthogonal layers**:

```
Scene Geometry   →   Visibility Backend   →   Radiosity Solver   →   Visualization
```

Each layer is replaceable without affecting the others.

---

## 2. Core Data Model

### 2.1 Patch (Radiosity Domain)

A *patch* is the fundamental radiosity element.

```cpp
struct Patch {
    Vec3 center;
    Vec3 normal;
    float area;

    Vec3 emission;     // RGB
    Vec3 reflectance;  // RGB (0..1)

    Vec3 B;            // current radiosity
    Vec3 B_unshot;     // for progressive refinement
};
```

Notes:
- Radiosity is stored **per patch**, not per triangle
- RGB is treated as a 3‑sample spectral discretization

---

### 2.2 Triangle (Geometric / Visibility Domain)

All geometry is represented as **triangles**, even if logically grouped as quads.

```cpp
struct Triangle {
    Vec3 v0, v1, v2;
    int patch_id;      // back‑reference to owning patch
};
```

Notes:
- One quad → two triangles
- OptiX and rasterization both operate on triangles
- Patch ↔ triangle mapping is explicit

---

## 3. Scene Construction

### 3.1 Authoring Geometry

- Scene is defined in terms of **logical surfaces (quads)**
- Each quad:
  - becomes one Patch
  - is split into two Triangles

This preserves conceptual clarity while staying compatible with ray tracing.

---

### 3.2 Precomputation per Patch

For each patch:
- Compute area
- Compute center
- Compute normal

These values are immutable during radiosity.

---

## 4. Visibility Computation (Ray‑Based)

Visibility is treated as a **binary occlusion query**:

> Is the line segment between two surface points unobstructed?

### 4.1 Half‑Space Reduction (Mandatory)

Before any ray is cast:

```text
if dot(n_i, (p_j − p_i)) ≤ 0 → skip
```

This removes all physically impossible energy transfer.

This step is **exact**, not an approximation.

---

### 4.2 Visibility Backend Abstraction

Define a backend‑independent interface:

```cpp
bool visible(Vec3 origin, Vec3 target, int ignore_patch_id);
```

Two interchangeable implementations are supported.

---

### 4.3 Backend A: Uniform Grid (CPU, Reference)

#### Purpose
- Low code complexity
- Didactically clean
- Portable

#### Components
- Scene AABB
- Uniform 3D grid
- Per‑cell triangle lists

#### Ray Traversal
- 3D DDA stepping
- Early exit on first hit
- Stop traversal at target distance

#### Intersection
- Ray–triangle intersection
- Ignore triangles belonging to the target patch

---

### 4.4 Backend B: OptiX (GPU, Optional)

#### Purpose
- High performance
- No custom acceleration structure

#### Strategy
- Build OptiX BVH once (static scene)
- Use **shadow rays** only
- Any‑hit program exits on first intersection

#### Mapping
- Triangle → patch_id mapping stored in hit data
- Visibility ray ignores target patch

#### Role in System
- Used **only** during form‑factor computation
- Radiosity iteration remains on CPU

---

## 5. Form Factor Computation

### 5.1 Mathematical Basis

For two patches i and j:

\[
F_{ij} = \frac{1}{A_i}
\int_{A_i}\int_{A_j}
\frac{\cos\theta_i \cos\theta_j}{\pi r^2}
V(x,y)\, dA_j dA_i
\]

This is approximated numerically.

---

### 5.2 Numerical Strategy

- Sample one or more points on patch i
- Sample one or more points on patch j
- For each sample pair:
  - Apply half‑space test
  - Perform visibility ray test
  - Accumulate contribution

Form factors are stored as:

```cpp
F[i][j]
```

Sparse storage is recommended.

---

## 6. Radiosity Solver

### 6.1 Solver Choice

Use **progressive refinement radiosity**.

Reason:
- Faster convergence
- Natural energy interpretation
- Produces meaningful intermediate states

---

### 6.2 Algorithm Outline

1. Initialize:
   - B = E
   - B_unshot = E

2. Loop until convergence:
   - Select patch p with maximum |B_unshot|
   - For all patches j:
     - ΔB_j += reflectance_j * F[p][j] * B_unshot[p]
   - Accumulate ΔB into B
   - Set B_unshot[p] = 0

---

### 6.3 Convergence Criteria

Stop when:
- max(|B_unshot|) < ε

ε is chosen based on perceptual relevance, not machine precision.

---

## 7. Rendering / Visualization

### 7.1 Rendering Strategy

- Rendering is **separate** from radiosity
- Use any simple rasterizer or offline renderer

Per triangle color:
```text
color = patch.B
```

This preserves view independence.

---

### 7.2 Optional Enhancements

- False‑color visualization of convergence
- Time‑step visualization of energy propagation
- Patch graph visualization

---

## 8. Numerical & Practical Considerations

### Stability
- Reflectance must be < 1
- Avoid degenerate patches

### Precision
- Use float for geometry
- Use double for accumulated radiosity if needed

### Performance
- Half‑space reduction first
- Visibility backend second
- Radiosity iteration last

---

## 9. Extensibility (Deliberately Deferred)

The following are **explicitly out of scope**, but structurally supported:

- Spectral radiosity
- Hierarchical radiosity
- Monte Carlo form factors
- Path tracing comparison

---

## 10. Mental Model Summary

- **Radiosity = global energy conservation**
- **Form factors = geometric coupling**
- **Iteration = physical feedback**
- **Visibility = replaceable module**
- **Triangles = geometric currency**
- **Patches = physical entities**

---

*End of coding plan.*

