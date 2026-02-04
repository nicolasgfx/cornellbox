//
// OptiX Hemisphere Form Factor Kernel - DIRECT REPLACEMENT
// Computes form factors via hemisphere ray casting (Monte Carlo)
//
// REPLACES OLD: Visibility + geometric kernel approach
// NEW APPROACH: F[i,j] = (hits from i to j) / N_rays
//

#include <optix.h>
#include <cuda_runtime.h>

// Launch parameters (uploaded to constant memory)
struct LaunchParams {
    // Geometry (World A - for OptiX traversal)
    float3* vertices;
    uint3* indices;
    
    // Vertex normals (computed from incident triangles)
    float3* vertexNormals;     // [numVertices]
    
    // Hemisphere sampling parameters
    uint32_t numVertices;
    uint32_t numTriangles;
    uint32_t dirSamples;       // Directions per vertex
    
    // Output: vertex-to-triangle form factors [numVertices * numTriangles]
    float* formFactors;
    
    // Scene parameters
    float sceneEpsilon;
    
    OptixTraversableHandle gasHandle;
};

extern "C" {
    __constant__ LaunchParams params;
}

// Hammersley low-discrepancy sequence
__device__ inline float radicalInverse(uint32_t bits) {
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10f;
}

__device__ inline float2 hammersley(uint32_t i, uint32_t N) {
    return make_float2(float(i) / float(N), radicalInverse(i));
}

// Cosine-weighted hemisphere sampling
__device__ inline float3 cosineWeightedHemisphere(float2 u) {
    float r = sqrtf(u.x);
    float theta = 2.0f * 3.14159265359f * u.y;
    
    float x = r * cosf(theta);
    float y = r * sinf(theta);
    float z = sqrtf(fmaxf(0.0f, 1.0f - u.x));
    
    return make_float3(x, y, z);
}

// Build orthonormal basis from normal
__device__ inline void buildOrthonormalBasis(const float3& N, float3& T, float3& B) {
    float sign = copysignf(1.0f, N.z);
    const float a = -1.0f / (sign + N.z);
    const float b = N.x * N.y * a;
    
    T = make_float3(1.0f + sign * N.x * N.x * a, sign * b, -sign * N.x);
    B = make_float3(b, sign + N.y * N.y * a, -N.y);
}

// Transform local direction to world space
__device__ inline float3 localToWorld(const float3& localDir, const float3& T, const float3& B, const float3& N) {
    return make_float3(
        localDir.x * T.x + localDir.y * B.x + localDir.z * N.x,
        localDir.x * T.y + localDir.y * B.y + localDir.z * N.y,
        localDir.x * T.z + localDir.y * B.z + localDir.z * N.z
    );
}

// Barycentric sampling for origin points
__device__ inline float3 barycentricSample(uint32_t sampleIdx) {
    if (sampleIdx == 0) return make_float3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f);
    switch (sampleIdx % 4) {
        case 1: return make_float3(0.5f, 0.25f, 0.25f);
        case 2: return make_float3(0.25f, 0.5f, 0.25f);
        case 3: return make_float3(0.25f, 0.25f, 0.5f);
        default: return make_float3(1.0f/3.0f, 1.0f/3.0f, 1.0f/3.0f);
    }
}

// Get world-space origin from triangle vertices and barycentric coords
__device__ inline float3 barycentricToWorld(const float3& v0, const float3& v1, const float3& v2, const float3& bary) {
    return make_float3(
        bary.x * v0.x + bary.y * v1.x + bary.z * v2.x,
        bary.x * v0.y + bary.y * v1.y + bary.z * v2.y,
        bary.x * v0.z + bary.y * v1.z + bary.z * v2.z
    );
}

//
// Ray generation program - PER-VERTEX HEMISPHERE FORM FACTORS
// Launch dims: [numVertices, dirSamples]
//
extern "C" __global__ void __raygen__hemisphere() {
    const uint3 idx = optixGetLaunchIndex();
    
    const uint32_t vertexId = idx.x;
    const uint32_t rayId = idx.y;
    
    if (vertexId >= params.numVertices) return;
    if (rayId >= params.dirSamples) return;
    
    // Load vertex position and normal
    const float3 origin = params.vertices[vertexId];
    const float3 normal = params.vertexNormals[vertexId];
    
    // Offset origin along normal to avoid self-intersection
    float3 rayOrigin = make_float3(
        origin.x + params.sceneEpsilon * normal.x,
        origin.y + params.sceneEpsilon * normal.y,
        origin.z + params.sceneEpsilon * normal.z
    );
    
    // Build orthonormal basis from vertex normal
    float3 tangent, bitangent;
    buildOrthonormalBasis(normal, tangent, bitangent);
    
    // Generate cosine-weighted direction using Hammersley sequence
    float2 u = hammersley(rayId, params.dirSamples);
    float3 localDir = cosineWeightedHemisphere(u);
    float3 worldDir = localToWorld(localDir, tangent, bitangent, normal);
    
    // Trace ray
    uint32_t p0 = UINT32_MAX;  // hitTriangleId
    uint32_t p1 = vertexId;    // sourceVertexId (for debugging)
    
    optixTrace(
        params.gasHandle,
        rayOrigin,
        worldDir,
        1e-4f,
        1e16f,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        p0, p1
    );
    
    // Accumulate to vertex-to-triangle form factor matrix
    uint32_t hitTriangleId = p0;
    if (hitTriangleId != UINT32_MAX) {
        float contribution = 1.0f / float(params.dirSamples);
        size_t ffIndex = size_t(vertexId) * params.numTriangles + hitTriangleId;
        atomicAdd(&params.formFactors[ffIndex], contribution);
    }
}

extern "C" __global__ void __closesthit__hemisphere() {
    optixSetPayload_0(optixGetPrimitiveIndex());
}

extern "C" __global__ void __anyhit__hemisphere() {
    // No self-hit filtering needed for per-vertex sampling
    // Vertices can see incident triangles
}

extern "C" __global__ void __miss__hemisphere() {
    // Leave payload as UINT32_MAX
}
