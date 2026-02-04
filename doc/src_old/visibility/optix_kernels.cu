#include <optix.h>
#include <optix_device.h>
#include "OptiXLaunchParams.h"

/**
 * OptiX Ray Tracing Kernels for Visibility Testing
 *
 * One-ray-per-launch implementation. The host is responsible for dispatching
 * multiple launches and aggregating the results to obtain fractional
 * visibility. Each launch receives one sampled point on the source triangle
 * and one on the target triangle and returns a binary visibility result.
 */

extern "C" {
__constant__ RayGenParams params;
}

__device__ inline float3 load_float3(const float v[3]) {
    return make_float3(v[0], v[1], v[2]);
}

__device__ inline float3 sub3(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ inline float3 cross3(const float3& a, const float3& b) {
    return make_float3(
        a.y * b.z - a.z * b.y,
        a.z * b.x - a.x * b.z,
        a.x * b.y - a.y * b.x
    );
}

__device__ inline float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ inline float3 normalize3(const float3& v) {
    float len2 = dot3(v, v);
    if (len2 <= 0.0f) {
        return make_float3(0.0f, 0.0f, 0.0f);
    }
    float invLen = rsqrtf(len2);
    return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
}

__device__ inline float3 triangleNormal(const TriangleData& tri) {
    float3 v0 = load_float3(tri.v0);
    float3 v1 = load_float3(tri.v1);
    float3 v2 = load_float3(tri.v2);
    return normalize3(cross3(sub3(v1, v0), sub3(v2, v0)));
}

__device__ inline float3 sampleTrianglePoint(const TriangleData& tri, float u, float v) {
    // Standard uniform barycentric sampling with reflection
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    float w = 1.0f - u - v;
    float3 v0 = load_float3(tri.v0);
    float3 v1 = load_float3(tri.v1);
    float3 v2 = load_float3(tri.v2);
    return make_float3(
        v0.x * w + v1.x * u + v2.x * v,
        v0.y * w + v1.y * u + v2.y * v,
        v0.z * w + v1.z * u + v2.z * v
    );
}

/**
 * Ray generation program – evaluates a single sampled visibility ray.
 */
extern "C" __global__ void __raygen__visibility() {
    const int resultIndex = params.result_offset;
    unsigned int* results = params.results;
    if (!results) {
        return;
    }

    // Sampled points on source and target triangles
    float3 sourcePoint = sampleTrianglePoint(params.source, params.source_uv[0], params.source_uv[1]);
    float3 targetPoint = sampleTrianglePoint(params.target, params.target_uv[0], params.target_uv[1]);

    float3 rayVec = sub3(targetPoint, sourcePoint);
    float distanceSquared = dot3(rayVec, rayVec);
    if (distanceSquared <= 1e-8f) {
        results[resultIndex] = 0u;
        return;
    }

    float distance = sqrtf(distanceSquared);
    float invDistance = 1.0f / distance;
    float3 rayDir = make_float3(rayVec.x * invDistance,
                                rayVec.y * invDistance,
                                rayVec.z * invDistance);

    float3 sourceNormal = triangleNormal(params.source);
    float3 targetNormal = triangleNormal(params.target);

    float cosSource = dot3(sourceNormal, rayDir);
    float3 negRayDir = make_float3(-rayDir.x, -rayDir.y, -rayDir.z);
    float cosTarget = dot3(targetNormal, negRayDir);
    if (cosSource <= 0.0f || cosTarget <= 0.0f) {
        results[resultIndex] = 0u;
        return;
    }

    unsigned int payloadVisible = 1u;
    float tmax = fmaxf(distance - 1e-4f, 1e-5f);

    optixTrace(
        params.traversable,
        sourcePoint,
        rayDir,
        1e-4f,                 // tmin – small epsilon to avoid self hits
        tmax,                   // tmax – stop just short of the target point
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
        0,
        1,
        0,
        payloadVisible
    );

    results[resultIndex] = payloadVisible;
}

/**
 * Any-hit program
 * Called for EVERY intersection - can filter/ignore specific hits
 * This is where we ignore the target patch to prevent self-occlusion
 */
extern "C" __global__ void __anyhit__occlusion() {
    // Get the hit patch ID
    int hitPatchId = optixGetPrimitiveIndex();
    
    // MANDATORY: Ignore the target patch - it should NOT block visibility to itself
    if (hitPatchId == params.target.patch_id) {
        optixIgnoreIntersection();  // Legal here - this is an any-hit program!
        return;
    }
    
    // Otherwise accept the hit - it's a real occluder
    // The ray will terminate due to OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
}

/**
 * Closest hit program
 * Called when ray hits geometry (after any-hit filtering)
 */
extern "C" __global__ void __closesthit__occlusion() {
    // Ray hit an actual occluder (target patch was already filtered out in any-hit)
    optixSetPayload_0(0u);
}

/**
 * Miss program
 * Called when ray misses all geometry -> visible
 */
extern "C" __global__ void __miss__visibility() {
    // Ray missed everything -> visible
    optixSetPayload_0(1u);
}
