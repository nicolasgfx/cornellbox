# Radiosity Debug Review – Black Room Except Light

This document reviews the failure case where **only the ceiling light is visible and the rest of the Cornell box remains black**.

This is a **diagnostic checklist**, not a redesign. The symptom is extremely specific and almost always indicates **zero energy propagation after emission**.

---

## 1. Symptom Summary

Observed output:
- Ceiling light triangles are bright and visible
- All other surfaces are completely black
- No gradients, no indirect illumination, no color bleeding

Interpretation:

> **Iteration 0 (emission) works. Iteration ≥ 1 transfers zero energy.**

This means the bug is **not** in:
- OBJ export
- vertex coloring
- emission assignment
- rendering/viewing

The bug **must** be in:
- form factor computation
- visibility usage
- radiosity iteration logic
- or surface normals

---

## 2. Most Likely Causes (Ranked)

### 2.1 Form factors are zero or near-zero (VERY LIKELY)

If:

```text
sum_j F[i][j] ≈ 0
```

for wall triangles, then **no energy can ever leave the light**.

Typical reasons:
- `cos(theta_i)` or `cos(theta_j)` always ≤ 0
- normals flipped
- incorrect distance term (`r²` too large or wrong units)
- dividing by area twice
- missing `1 / π`
- visibility factor always zero

**Required check:**
- Print max / mean form factor per triangle

---

### 2.2 Normals are flipped (EXTREMELY LIKELY)

For the Cornell box:
- wall normals must point **inwards**
- ceiling light normals must point **downwards**

If the light normal points upward:

```text
cos(theta_light) ≤ 0
```

→ zero contribution everywhere

**Minimal diagnostic:**
For one light triangle `L` and one wall triangle `W`, print:

```cpp
dot(n_L, normalize(c_W - c_L))
dot(n_W, normalize(c_L - c_W))
```

Both values must be **positive**.

---

### 2.3 Visibility is zero for all non-self pairs

If the visibility cache contains:

```text
V(light, wall) = 0
```

then no transfer occurs, even with correct form factors.

Possible causes:
- target triangle incorrectly ignored
- `tmax` too small
- hemisphere check rejecting valid rays
- sample points outside triangles

**Required check:**

```cpp
count j where V[light][j] > 0
```

If near zero → visibility kernel bug.

---

### 2.4 Radiosity iteration only performs emission step

Another classic failure mode:
- radiosity loop executes once
- or `B` is overwritten instead of accumulated
- or progressive refinement never shoots unshot energy

Correct conceptual logic:

```text
B = E
repeat:
  B += ρ · F · B
until convergence
```

If the loop exits early or resets buffers, the result matches the observed image.

---

### 2.5 Reflectance values are zero or too small

If wall reflectance is:

```text
ρ = (0,0,0)
```

or extremely small, no visible indirect light will appear.

**Check:**
- Print average wall reflectance
- Cornell box walls typically use ~0.7–0.8

---

### 2.6 Energy scale or clamping issues (LESS LIKELY)

If:
- emission is tiny
- form factors are small
- and colors are clamped early

You may see near-black results.

However, this usually still shows *something*. A completely black room usually indicates a harder logic error.

---

## 3. Two Killer Sanity Tests (Do These First)

### 3.1 Force visibility to 1

Temporarily override:

```cpp
V[i][j] = 1.0f;
```

for all `i ≠ j`.

- If the room **lights up** → visibility computation or usage is broken
- If the room is **still black** → radiosity solver or form factors are broken

This single test cleanly isolates the problem.

---

### 3.2 Skip form factors entirely

Temporarily apply:

```cpp
B[j] += 0.1f * B_light;
```

for all wall triangles.

- If walls light up → form factor / visibility bug
- If walls stay black → solver / accumulation / export bug

---

## 4. Expected Correct Early Output

Even with:
- very coarse geometry
- few iterations
- approximate form factors

A correct implementation should show:
- ceiling bright near the light
- walls faintly illuminated
- floor not completely black
- visible gradients

A fully black room indicates **zero energy propagation**, not just low quality.

---

## 5. Strongest Overall Diagnosis

Based on the symptom:

> **The most likely root cause is flipped normals (especially the ceiling light) or zero visibility between light and walls.**

Second most likely:
> **Form factor computation returns zero everywhere.**

This is almost never a high-level design problem.

---

## 6. What to Inspect Next (Minimal Set)

To resolve this in one iteration, inspect ONE of the following:

1. The code computing `cos(theta_i)` / `cos(theta_j)`
2. The radiosity iteration loop (accumulation vs overwrite)
3. A printout of `max(F_ij)` and `avg(F_ij)`
4. Result of the “force visibility = 1” test

---

*End of radiosity debug review.*

