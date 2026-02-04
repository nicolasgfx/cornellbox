#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace HemisphereSampling {

// Cosine-weighted hemisphere sampling using Hammersley low-discrepancy sequence
// PDF: p(ω) = cos(θ) / π
// This makes the Monte Carlo estimator weight = 1, so each ray contributes 1/N_rays

// Generate Hammersley 2D sample point
__host__ __device__ inline float radicalInverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

__host__ __device__ inline float2 hammersley(uint32_t i, uint32_t N) {
    return make_float2(float(i) / float(N), radicalInverse(i));
}

// Convert uniform 2D sample to cosine-weighted hemisphere direction
// Input: u ∈ [0,1]²
// Output: direction in local space (N = +Z axis)
__host__ __device__ inline float3 cosineWeightedHemisphere(float2 u) {
    // Map to disk using Shirley's concentric mapping
    float r = sqrtf(u.x);
    float theta = 2.0f * 3.14159265359f * u.y;
    
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u.x)); // cos(θ) = sqrt(1 - sin²(θ))
    
    return make_float3(x, y, z);
}

// Build orthonormal basis from normal (Duff et al. 2017 - Building an Orthonormal Basis, Revisited)
// Input: N (normalized)
// Output: T, B such that {T, B, N} forms right-handed orthonormal basis
__host__ __device__ inline void buildOrthonormalBasis(const float3& N, float3& T, float3& B) {
    float sign = copysignf(1.0f, N.z);
    const float a = -1.0f / (sign + N.z);
    const float b = N.x * N.y * a;
    
    T = make_float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = make_float3(b, sign + N.y * N.y * a, -N.y);
}

// Transform local direction to world space
__host__ __device__ inline float3 localToWorld(const float3& localDir, const float3& T, const float3& B, const float3& N) {
    return make_float3(
        localDir.x * T.x + localDir.y * B.x + localDir.z * N.x,
        localDir.x * T.y + localDir.y * B.y + localDir.z * N.y,
        localDir.x * T.z + localDir.y * B.z + localDir.z * N.z
    );
}

// Barycentric sampling for origin points on triangle
// Returns barycentric coordinates (u, v, w) where w = 1 - u - v
__host__ __device__ inline float3 barycentricSample(uint32_t sampleIdx, uint32_t numSamples) {
    if (numSamples == 1) {
        // Centroid only
        return make_float3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f);
    }
    
    // Stratified sampling over triangle
    // For small N (2-4), use simple patterns
    switch (sampleIdx % 4) {
        case 0: return make_float3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f); // centroid
        case 1: return make_float3(0.5f, 0.25f, 0.25f);
        case 2: return make_float3(0.25f, 0.5f, 0.25f);
        case 3: return make_float3(0.25f, 0.25f, 0.5f);
        default: return make_float3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f);
    }
}

// Get world-space origin from triangle vertices and barycentric coords
__host__ __device__ inline float3 barycentricToWorld(
    const float3& v0, const float3& v1, const float3& v2,
    const float3& bary) {
    return make_float3(
        bary.x * v0.x + bary.y * v1.x + bary.z * v2.x,
        bary.x * v0.y + bary.y * v1.y + bary.z * v2.y,
        bary.x * v0.z + bary.y * v1.z + bary.z * v2.z
    );
}

} // namespace HemisphereSampling
