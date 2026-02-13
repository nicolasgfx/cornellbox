// OptiX render kernel: pinhole camera ray tracer with per-vertex color interpolation.
//
// Traces one primary ray per pixel.  On hit, retrieves barycentric coordinates
// from the built-in triangle intersection and interpolates vertex colors stored
// in a device-side float3 array.  Writes the result to an RGBA float framebuffer.

#include <optix.h>
#include <cuda_runtime.h>

struct RenderParams {
    // Framebuffer (float4 RGBA, row-major, top-to-bottom).
    float4* framebuffer;
    uint32_t width;
    uint32_t height;

    // Camera (pinhole).
    float3 eye;
    float3 U;    // camera right  (scaled to half-image-plane width)
    float3 V;    // camera up     (scaled to half-image-plane height)
    float3 W;    // camera forward (eye → center, scaled to focal length)

    // Per-vertex colors (float3, indexed by mesh vertex id).
    float3* vertexColors;

    // Index buffer so we can look up which 3 vertices form the hit triangle.
    uint3* indices;

    OptixTraversableHandle gasHandle;
};

extern "C" {
    __constant__ RenderParams params;
}

extern "C" __global__ void __raygen__render() {
    const uint3 idx = optixGetLaunchIndex();
    const uint32_t px = idx.x;
    const uint32_t py = idx.y;

    // Normalised device coordinates in [-1, 1].
    const float u = 2.0f * (float(px) + 0.5f) / float(params.width)  - 1.0f;
    const float v = 2.0f * (float(py) + 0.5f) / float(params.height) - 1.0f;

    // Ray direction through the image plane.
    float3 dir;
    dir.x = params.W.x + u * params.U.x + v * params.V.x;
    dir.y = params.W.y + u * params.U.y + v * params.V.y;
    dir.z = params.W.z + u * params.U.z + v * params.V.z;

    // Normalise direction.
    const float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    dir.x /= len;
    dir.y /= len;
    dir.z /= len;

    // Payload: rgb packed into 3 uints via __float_as_uint.
    uint32_t p0 = __float_as_uint(0.0f);
    uint32_t p1 = __float_as_uint(0.0f);
    uint32_t p2 = __float_as_uint(0.0f);

    optixTrace(
        params.gasHandle,
        params.eye,
        dir,
        0.0f,          // tmin
        1e16f,         // tmax
        0.0f,          // ray time
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0, 1, 0,
        p0, p1, p2
    );

    const float r = __uint_as_float(p0);
    const float g = __uint_as_float(p1);
    const float b = __uint_as_float(p2);

    const uint32_t fbIdx = py * params.width + px;
    params.framebuffer[fbIdx] = make_float4(r, g, b, 1.0f);
}

extern "C" __global__ void __closesthit__render() {
    // Built-in triangle barycentrics (u, v).  w = 1 - u - v.
    const float2 bary = optixGetTriangleBarycentrics();
    const float bw = 1.0f - bary.x - bary.y;

    const uint32_t primIdx = optixGetPrimitiveIndex();
    const uint3 tri = params.indices[primIdx];

    const float3 c0 = params.vertexColors[tri.x];
    const float3 c1 = params.vertexColors[tri.y];
    const float3 c2 = params.vertexColors[tri.z];

    const float r = bw * c0.x + bary.x * c1.x + bary.y * c2.x;
    const float g = bw * c0.y + bary.x * c1.y + bary.y * c2.y;
    const float b = bw * c0.z + bary.x * c1.z + bary.y * c2.z;

    optixSetPayload_0(__float_as_uint(r));
    optixSetPayload_1(__float_as_uint(g));
    optixSetPayload_2(__float_as_uint(b));
}

extern "C" __global__ void __miss__render() {
    // Background: black.
    optixSetPayload_0(__float_as_uint(0.0f));
    optixSetPayload_1(__float_as_uint(0.0f));
    optixSetPayload_2(__float_as_uint(0.0f));
}
