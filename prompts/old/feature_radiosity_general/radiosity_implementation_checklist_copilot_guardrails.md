# Radiosity Implementation Checklist

*A guardrail document for Copilot / code generation tools.*  
Use this as a **systematic checklist** to verify that a classical radiosity implementation is **mathematically correct, physically meaningful, and complete**.

This document is intentionally **prescriptive**. Each item is phrased so it can be answered with **YES / NO**.

---

## 0. Scope of This Checklist

This checklist assumes:
- Classical **diffuse radiosity**
- Explicit **form factor computation**
- **Ray tracing** used only for *visibility* (occlusion)
- Geometry represented as **triangles**, radiosity solved on **patches**
- No path tracing, no Monte Carlo light transport recursion

If your implementation goes beyond this, this checklist still applies to the radiosity core.

---

## 1. Radiosity Equation – Global Correctness

### 1.1 Fundamental equation

☐ Is the radiosity equation implemented as:

\[
B_i = E_i + \rho_i \sum_{j=1}^{N} F_{ji} B_j
\]

(not a local or per-light approximation)?

☐ Is radiosity **view-independent** (no camera terms anywhere)?

☐ Is radiosity stored **per patch**, not per vertex or per triangle?

---

### 1.2 Iteration / Solver

☐ Is the equation solved **iteratively** (Jacobi, Gauss–Seidel, or progressive refinement)?

☐ Are *all* other patches allowed to contribute to patch *i* (not only neighbors)?

☐ Is convergence determined by **remaining energy** or **change in B**, not by a fixed bounce count?

---

## 2. Patch Interaction Coverage

### 2.1 Completeness

☐ For each patch *i*, are form factors computed **against every other patch *j***?

☐ Is self-interaction (i == j) explicitly excluded?

☐ Is there **no artificial cutoff** based on distance alone?

---

### 2.2 Energy Conservation

☐ For each patch *i*, does the sum satisfy:

\[
\sum_j F_{ij} \le 1
\]

(for a closed scene)?

☐ Is this sum **measured and validated** during debugging?

---

## 3. Form Factor Mathematics

### 3.1 Core formula

☐ Is the geometric kernel implemented as:

\[
\frac{\cos\theta_i \cos\theta_j}{\pi r^2}
\]

(not missing π, not using r instead of r²)?

☐ Is the **receiver patch area** included in the numerator?

☐ Is the **sender patch area** correctly normalized (division by A_i)?

---

### 3.2 Orientation / Half-Space Reduction

☐ Are interactions discarded when:

\[
\cos\theta_i \le 0 \quad \text{or} \quad \cos\theta_j \le 0
\]

☐ Is this half-space test applied **before** any visibility ray is cast?

---

## 4. Integration vs. Point Sampling

### 4.1 Critical correctness check

☐ Are form factors computed using **area integration** (sampling points on patches)?

☐ Or is the implementation *only* using patch centers?

⚠️ If only centers are used:
- Expect very weak indirect light
- Expect almost no color bleeding

---

### 4.2 Sampling sanity

☐ Do multiple samples correspond to **different surface points**, not just repeated rays?

☐ Are samples distributed over the *area* of the patch (uniform or stratified)?

☐ Is the final form factor the **average over samples**, not the sum?

---

## 5. Visibility via Ray Tracing (Triangle-to-Triangle)

### 5.1 Ray definition

☐ Is each visibility ray defined as a **segment** from source point to target point?

☐ Is the ray length clamped to the distance between the two surface points?

---

### 5.2 Intersection logic

☐ Does the visibility test return **occluded** if *any* hit occurs *before* the target point?

☐ Is the target triangle (or patch) explicitly ignored during intersection testing?

☐ Are backfaces handled consistently (either culled or intersected intentionally)?

---

### 5.3 Acceleration

☐ Is an acceleration structure used (uniform grid, BVH, OptiX)?

☐ Is visibility testing treated as a **binary query** (visible / blocked)?

---

## 6. Triangle vs. Patch Consistency

☐ Are visibility rays cast against **triangles**, not abstract patches?

☐ Is there a clear mapping:

```text
triangle → patch
```

☐ Does a hit on *any* triangle belonging to a patch count as blocking that patch?

---

## 7. Progressive Refinement (if used)

☐ Is unshot energy tracked per patch?

☐ Is energy distributed according to:

\[
\Delta B_j = \rho_j F_{ij} B_i^{unshot}
\]

☐ Is unshot energy cleared after shooting?

---

## 8. Numerical Stability

☐ Are reflectance values strictly < 1?

☐ Are degenerate or extremely small patches avoided?

☐ Are floating-point tolerances used consistently (ε for distance, dot products)?

---

## 9. Rendering & Reconstruction (Post-Solve)

☐ Is rendering performed **after** radiosity convergence?

☐ Are per-vertex values reconstructed from per-patch radiosity (not recomputed)?

☐ Is area-weighted averaging used for vertex reconstruction?

---

## 10. Red Flags (Common Failure Modes)

If any of the following are true, the result is likely incorrect:

☐ Indirect lighting is almost invisible even after many iterations
☐ Color bleeding is absent in Cornell box scenes
☐ Increasing samples only increases noise, not brightness
☐ Form factor sums are orders of magnitude below 1
☐ Changing visibility backend changes brightness dramatically

---

## 11. Mental Model Check (Final)

Before trusting the output, confirm:

☐ Radiosity is treated as **global energy equilibrium**
☐ Visibility is treated as **binary occlusion**, not shading
☐ Sampling approximates **surface integrals**, not point values
☐ Rendering is only a visualization of the solved energy field

---

*End of radiosity correctness checklist.*

