//
// Hemispherical Ray Casting for Radiosity Form Factors
// OptiX + CUDA Implementation
//
// This kernel computes form factors F[i,j] using Monte Carlo hemisphere sampling:
// - For each patch i, cast rays over the positive hemisphere
// - Use cosine-weighted sampling: PDF = cos(θ) / π
// - Each ray contributes 1/N_rays to the hit patch j
// - Result: F[i,j] = (# hits from i to j) / N_rays
//

#include <optix.h>
#include <cuda_runtime.h>
#include "../math/HemisphereSampling.h"

// Payload for hemisphere rays
struct HemispherePayload {
    uint32_t hitPatchId;      // Hit patch ID (UINT32_MAX if miss)
    uint32_t sourcePatchId;   // Source patch ID (for self-hit filtering)
};

// Launch parameters
struct HemisphereLaunchParams {
    // Geometry (World A - for OptiX traversal)
    float3* vertices;
    uint3* indices;
    
    // Patch data (World B - SoA)
    float* cx;     // centroid x [numTriangles]
    float* cy;     // centroid y [numTriangles]
    float* cz;     // centroid z [numTriangles]
    float* nx;     // normal x [numTriangles]
    float* ny;     // normal y [numTriangles]
    float* nz;     // normal z [numTriangles]
    
    // Sampling parameters
    uint32_t numTriangles;
    uint32_t originSamples;    // Origins per patch (typically 1 or 4)
    uint32_t dirSamples;       // Directions per origin (256-2048)
    
    // Output: form factor accumulation
    // Dense storage: formFactors[i * numTriangles + j] (only for LOW profile)
    // For MEDIUM/HIGH, use atomic adds to sparse structure
    float* formFactors;        // [numTriangles * numTriangles] or NULL if sparse
    
    // Scene parameters
    float sceneEpsilon;        // Offset epsilon (scaled to scene size)
    
    OptixTraversableHandle gasHandle;
};

extern "C" {
    __constant__ HemisphereLaunchParams params;
}

//
// Ray generation program
// Launch dimensions: (numTriangles, originSamples * dirSamples)
// Thread (i, r) computes one ray for patch i
//
extern "C" __global__ void __raygen__hemisphere_formfactor() {
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    
    const uint32_t patchId = idx.x;
    const uint32_t rayId = idx.y;
    
    if (patchId >= params.numTriangles) return;
    
    // Decompose rayId into origin and direction sample
    const uint32_t raysPerOrigin = params.dirSamples;
    const uint32_t originIdx = rayId / raysPerOrigin;
    const uint32_t dirIdx = rayId % raysPerOrigin;
    
    if (originIdx >= params.originSamples) return;
    
    // Load patch data from SoA
    const float3 centroid = make_float3(params.cx[patchId], 
                                       params.cy[patchId], 
                                       params.cz[patchId]);
    const float3 normal = make_float3(params.nx[patchId], 
                                     params.ny[patchId], 
                                     params.nz[patchId]);
    
    // Get triangle vertices for origin sampling
    const uint3 tri = params.indices[patchId];
    const float3 v0 = params.vertices[tri.x];
    const float3 v1 = params.vertices[tri.y];
    const float3 v2 = params.vertices[tri.z];
    
    // Sample origin on triangle surface
    float3 baryCoords = HemisphereSampling::barycentricSample(originIdx, params.originSamples);
    float3 origin = HemisphereSampling::barycentricToWorld(v0, v1, v2, baryCoords);
    
    // Add epsilon offset along normal to avoid self-intersection
    origin.x += params.sceneEpsilon * normal.x;
    origin.y += params.sceneEpsilon * normal.y;
    origin.z += params.sceneEpsilon * normal.z;
    
    // Build orthonormal basis from normal
    float3 tangent, bitangent;
    HemisphereSampling::buildOrthonormalBasis(normal, tangent, bitangent);
    
    // Generate cosine-weighted direction sample
    uint32_t totalDirSamples = params.originSamples * params.dirSamples;
    uint32_t globalDirIdx = originIdx * params.dirSamples + dirIdx;
    float2 u = HemisphereSampling::hammersley(globalDirIdx, totalDirSamples);
    
    float3 localDir = HemisphereSampling::cosineWeightedHemisphere(u);
    float3 worldDir = HemisphereSampling::localToWorld(localDir, tangent, bitangent, normal);
    
    // Trace ray
    // Pack payload: p0 = hitPatchId (UINT32_MAX initially), p1 = sourcePatchId
    uint32_t p0 = UINT32_MAX;
    uint32_t p1 = patchId;
    
    optixTrace(
        params.gasHandle,
        origin,
        worldDir,
        1e-4f,           // tmin
        1e16f,           // tmax
        0.0f,            // rayTime
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,               // SBT offset
        1,               // SBT stride
        0,               // missSBTIndex
        p0, p1
    );
    
    // Unpack payload
    uint32_t hitPatchId = p0;
    
    // Accumulate to form factor matrix
    if (hitPatchId != UINT32_MAX && hitPatchId != patchId) {
        uint32_t j = hitPatchId;
        
        // Atomic add contribution: 1 / (originSamples * dirSamples)
        float contribution = 1.0f / float(params.originSamples * params.dirSamples);
        
        if (params.formFactors != nullptr) {
            // Dense storage for LOW profile
            size_t idx = size_t(patchId) * params.numTriangles + j;
            atomicAdd(&params.formFactors[idx], contribution);
        }
    }
}

//
// Closest hit program
// Records the hit primitive index as the hit patch ID
//
extern "C" __global__ void __closesthit__hemisphere() {
    const uint32_t hitPrimIdx = optixGetPrimitiveIndex();
    
    // Set payload (p0 = hit patch ID)
    optixSetPayload_0(hitPrimIdx);
}

//
// Any-hit program
// Filters self-intersections
//
extern "C" __global__ void __anyhit__hemisphere() {
    const uint32_t hitPrimIdx = optixGetPrimitiveIndex();
    const uint32_t sourcePatchId = optixGetPayload_1();
    
    // Ignore self-hits
    if (hitPrimIdx == sourcePatchId) {
        optixIgnoreIntersection();
    }
}

//
// Miss program
// Leaves payload as UINT32_MAX (no hit)
//
extern "C" __global__ void __miss__hemisphere() {
    // Payload already initialized to UINT32_MAX in raygen
    // No action needed
}
