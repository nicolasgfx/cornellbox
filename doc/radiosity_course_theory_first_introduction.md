# A Theory‑First Introduction to Classical Radiosity

*A compact but rigorous course, designed for sharing and self‑study.*

This document is structured as **three sessions**, each building on the previous one.  
The focus is **theory, math, and physical intuition**. Code and implementation details are intentionally secondary and will be introduced later.

---

## Session 1 – What Radiosity Solves (Physics & Mathematics)

### 1.1 What problem radiosity addresses

Radiosity is a method for computing **global illumination** in scenes where:

- All surfaces are **diffuse (Lambertian)**
- Light transport is **view‑independent**
- Energy is **conserved**

Unlike local illumination models, radiosity accounts for **indirect light** — light bouncing multiple times between surfaces.

---

### 1.2 Radiosity as an energy balance problem

For each surface patch *i*, radiosity describes a balance of energy:

- Energy **emitted** by the patch
- Energy **reflected** from all other patches

This leads to the fundamental radiosity equation:

\[
B_i = E_i + \rho_i \sum_{j=1}^{N} F_{ji} B_j
\]

Where:

- \(B_i\) — radiosity of patch *i* (total outgoing energy per unit area)
- \(E_i\) — emitted energy (light sources)
- \(\rho_i\) — diffuse reflectance (albedo)
- \(F_{ji}\) — **form factor** from patch *j* to *i*

This equation already encodes **infinite light bounces**.

---

### 1.3 The form factor (geometric coupling)

The form factor represents **how much energy leaving one patch reaches another**:

\[
F_{ij} = \frac{1}{A_i}
\int_{A_i}\int_{A_j}
\frac{\cos\theta_i \cos\theta_j}{\pi r^2} V(x,y)\, dA_j dA_i
\]

Key properties:

- Purely **geometric**
- Independent of materials and light color
- Encodes visibility, orientation, and distance

Important identities:

- **Reciprocity**: \(A_i F_{ij} = A_j F_{ji}\)
- **Energy conservation**: \(\sum_j F_{ij} \le 1\)

---

### 1.4 Why radiosity becomes a linear system

Writing the equation for all patches gives:

\[
\mathbf{B} = \mathbf{E} + \mathbf{R}\mathbf{F}\mathbf{B}
\]

or:

\[
(\mathbf{I} - \mathbf{R}\mathbf{F})\mathbf{B} = \mathbf{E}
\]

This is a **global, coupled linear system**.

> There is no closed‑form per‑patch solution.

This fact drives every algorithmic choice that follows.

---

## Session 2 – Solving the Radiosity Equation (Iteration & Convergence)

### 2.1 Why iteration is unavoidable

The radiosity matrix couples *every patch to every other patch*.

You have only three options:

1. Direct matrix solve (\(O(N^3)\), impractical)
2. Iterative solvers
3. Progressive refinement

All practical radiosity systems are **iterative**.

---

### 2.2 Jacobi and Gauss–Seidel iteration

A simple fixed‑point iteration:

\[
B_i^{(k+1)} = E_i + \rho_i \sum_j F_{ji} B_j^{(k)}
\]

Properties:

- Conceptually simple
- Slow convergence
- Requires full updates every iteration

Gauss–Seidel improves convergence by immediately reusing updated values.

---

### 2.3 Progressive refinement (energy shooting)

Instead of updating all patches every iteration:

1. Track **unshot energy** per patch
2. Select the patch with the most remaining energy
3. Distribute that energy to all other patches
4. Mark it as shot

Mathematically equivalent, but:

- Faster convergence
- Better numerical stability
- Intermediate visual results

This became the **dominant classical radiosity method**.

---

### 2.4 Convergence and stopping criteria

Radiosity converges when:

- Remaining unshot energy < \(\varepsilon\)
- Or maximum change in \(B\) falls below threshold

Important observations:

- High‑order bounces contribute less energy
- Visual convergence often precedes numeric convergence

---

### 2.5 Spectral vs RGB radiosity (theory view)

Radiosity is wavelength‑separable:

\[
B(\lambda) = E(\lambda) + \rho(\lambda) F B(\lambda)
\]

If materials are wavelength‑independent:

- Each wavelength solves the **same equation**
- RGB is already a 3‑sample spectral discretization

Spectral radiosity only adds value when transport depends on wavelength.

---

## Session 3 – Visibility, Geometry, and Practical Constraints

### 3.1 Why visibility dominates runtime

The hardest part of radiosity is computing:

\[
V(x,y) \in \{0,1\}
\]

Naive visibility (ray vs all patches) scales as:

\[
O(N^3)
\]

This is completely impractical.

---

### 3.2 Rasterization‑based visibility (hemicube)

Classical radiosity avoids ray tracing entirely.

For each patch:

- Place a hemicube aligned with the surface normal
- Rasterize the scene
- Use a z‑buffer to resolve visibility
- Accumulate form factors from visible pixels

Visibility is solved **implicitly** by rasterization.

---

### 3.3 Ray‑based visibility (modern reinterpretation)

If rays are used, acceleration is mandatory.

A simple and effective structure:

- **Uniform spatial grid**

Why it works well for radiosity:

- Rays are short
- Scene is static
- Geometry density is uniform (e.g. Cornell box)

This reduces intersection tests from \(O(N)\) to a small constant.

---

### 3.4 Why radiosity prefers rasterization historically

Radiosity emerged before fast ray tracing.

Rasterization offered:

- Hardware acceleration
- Z‑buffered visibility
- Efficient many‑to‑many coupling

Ray tracing only became competitive decades later.

---

### 3.5 Conceptual limits of classical radiosity

Radiosity cannot model:

- Specular reflection
- Glossy surfaces
- Caustics
- Participating media

These limitations are **theoretical**, not implementation flaws.

---

## Closing perspective

Radiosity is best understood as:

> **A global energy‑conservation problem solved numerically over geometry.**

Its importance lies not in modern production use, but in:

- Making light transport explicit
- Teaching global illumination fundamentals
- Providing a bridge to Monte Carlo methods

---

### Suggested next steps

- Introduce code only *after* the math is internalized
- Implement a minimal patch system
- Compare radiosity vs path tracing on the same scene

---

*End of theory session document.*

