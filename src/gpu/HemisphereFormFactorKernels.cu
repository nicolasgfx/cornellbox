#include <optix.h>
#include <cuda_runtime.h>

// OptiX kernel for computing combined form-factor × visibility rows.
//
// Uses Monte Carlo integration of the area-to-area form factor integral:
//   F_ij = (1/A_i) ∫∫ (cos θ_i · cos θ_j) / (π r²) · V(x,y) dA_j dA_i
//
// With uniform sampling on both triangles (1 sample pair per thread):
//   F_ij ≈ (A_j / N) Σ_k [ cos θ_i,k · cos θ_j,k / (π r_k²) ] · V_k
//
// Each thread:
//   1) Samples one random point on source triangle i
//   2) Samples one random point on target triangle j
//   3) Evaluates the geometric kernel: cos_i * cos_j * A_j / (π r²)
//   4) Traces a segment ray; if unoccluded, atomicAdds the sample to rowOutput[j]
//
// Launch dimensions: [numTriangles, samplesPerTarget]

__device__ inline uint32_t hash32(uint32_t x) {
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

__device__ inline float u01_from_u32(uint32_t x) {
    return float(x) * 2.3283064365386963e-10f;
}

__device__ inline float3 barycentricSample(uint32_t sampleIdx, uint32_t salt) {
    const uint32_t h0 = hash32(sampleIdx * 747796405u ^ salt);
    const uint32_t h1 = hash32(sampleIdx * 2891336453u ^ (salt ^ 0x9e3779b9u));
    const float u = u01_from_u32(h0);
    const float v = u01_from_u32(h1);

    const float su = sqrtf(u);
    const float b0 = 1.0f - su;
    const float b1 = su * (1.0f - v);
    const float b2 = su * v;
    return make_float3(b0, b1, b2);
}

__device__ inline float3 barycentricToWorld(const float3& a,
                                            const float3& b,
                                            const float3& c,
                                            const float3& bary) {
    return make_float3(
        bary.x * a.x + bary.y * b.x + bary.z * c.x,
        bary.x * a.y + bary.y * b.y + bary.z * c.y,
        bary.x * a.z + bary.y * b.z + bary.z * c.z
    );
}

struct LaunchParams {
    float3* vertices;
    uint3* indices;
    float* cx;
    float* cy;
    float* cz;
    float* nx;
    float* ny;
    float* nz;
    float* area;           // per-triangle area
    uint32_t numTriangles;
    uint32_t originSamples;
    uint32_t dirSamples;
    float* formFactors;
    float sceneEpsilon;
    float distanceSoftening;
    uint32_t basePatchId;
    float* rowOutput;
    OptixTraversableHandle gasHandle;
};

extern "C" {
    __constant__ LaunchParams params;
}

extern "C" __global__ void __raygen__formfactor_row() {
    const uint3 idx = optixGetLaunchIndex();
    const uint32_t targetId = idx.x;
    const uint32_t sampleId = idx.y;
    const uint32_t sourceId = params.basePatchId;
    const uint32_t N = params.dirSamples;

    if (targetId >= params.numTriangles || targetId == sourceId || sampleId >= N) return;

    // Load normals.
    const float3 nS = make_float3(params.nx[sourceId], params.ny[sourceId], params.nz[sourceId]);
    const float3 nT = make_float3(params.nx[targetId], params.ny[targetId], params.nz[targetId]);

    // Load triangle vertices.
    const uint3 triS = params.indices[sourceId];
    const uint3 triT = params.indices[targetId];
    const float3 s0 = params.vertices[triS.x];
    const float3 s1 = params.vertices[triS.y];
    const float3 s2 = params.vertices[triS.z];
    const float3 t0 = params.vertices[triT.x];
    const float3 t1 = params.vertices[triT.y];
    const float3 t2 = params.vertices[triT.z];

    // Sample random points on source and target triangles.
    const uint32_t sourceSalt = sourceId * 73856093u ^ targetId * 19349663u;
    const uint32_t targetSalt = targetId * 83492791u ^ sourceId * 2654435761u;
    float3 origin = barycentricToWorld(s0, s1, s2, barycentricSample(sampleId, sourceSalt));
    float3 target = barycentricToWorld(t0, t1, t2, barycentricSample(sampleId, targetSalt));

    // Direction vector from sample point to sample point.
    float3 dir = make_float3(target.x - origin.x, target.y - origin.y, target.z - origin.z);
    const float actualDist2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    if (actualDist2 <= 1e-10f) return;
    const float actualDist = sqrtf(actualDist2);
    const float invDist = 1.0f / actualDist;
    dir.x *= invDist;
    dir.y *= invDist;
    dir.z *= invDist;

    // Evaluate geometric kernel: cos_i * cos_j / (π r²).
    const float cosI = nS.x * dir.x + nS.y * dir.y + nS.z * dir.z;
    const float cosJ = -(nT.x * dir.x + nT.y * dir.y + nT.z * dir.z);
    if (cosI <= 0.0f || cosJ <= 0.0f) return;

    const float Aj = params.area[targetId];
    // Use clamped dist² in the denominator to bound form factors for
    // very close sample points, but direction is always from actual geometry.
    const float minDist2 = Aj * 0.1f;
    const float clampedDist2 = (actualDist2 < minDist2) ? minDist2 : actualDist2;
    // Monte Carlo sample of the form factor: (cosI * cosJ * Aj) / (π (r² + k) N)
    // k = distanceSoftening; 0 = standard physics, >0 compresses near/far ratio.
    float sample = (cosI * cosJ * Aj) / (3.14159265358979323846f * (clampedDist2 + params.distanceSoftening) * float(N));
    // Clamp: a single sample contribution must not exceed 1/N.
    if (sample > 1.0f / float(N)) sample = 1.0f / float(N);
    if (sample < 1e-12f) return;

    // Offset origin/target along normals to avoid self-intersection.
    origin.x += params.sceneEpsilon * nS.x;
    origin.y += params.sceneEpsilon * nS.y;
    origin.z += params.sceneEpsilon * nS.z;

    // Recompute direction after offset.
    dir.x = target.x - origin.x;
    dir.y = target.y - origin.y;
    dir.z = target.z - origin.z;
    const float tmax2 = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
    if (tmax2 <= 1e-8f) return;
    const float tmaxDist = sqrtf(tmax2);
    dir.x /= tmaxDist;
    dir.y /= tmaxDist;
    dir.z /= tmaxDist;

    const float tmax = tmaxDist - params.sceneEpsilon;
    if (tmax <= 1e-4f) return;

    // Trace segment ray; p0 receives hit primitive index, p1 carries source id for anyhit.
    uint32_t p0 = UINT32_MAX;
    uint32_t p1 = sourceId;

    optixTrace(
        params.gasHandle,
        origin,
        dir,
        1e-4f,
        tmax,
        0.0f,
        OptixVisibilityMask(255),
        OPTIX_RAY_FLAG_NONE,
        0,
        1,
        0,
        p0,
        p1
    );

    // If nothing blocks the segment, the path is unoccluded.
    // p0 == UINT32_MAX means miss (ray ended before hitting anything).
    // p0 == targetId means the target itself was the first hit.
    if (p0 == UINT32_MAX || p0 == targetId) {
        atomicAdd(&params.rowOutput[targetId], sample);
    }
}

extern "C" __global__ void __closesthit__hemisphere() {
    optixSetPayload_0(optixGetPrimitiveIndex());
}

extern "C" __global__ void __anyhit__hemisphere() {
    if (optixGetPrimitiveIndex() == optixGetPayload_1()) optixIgnoreIntersection();
}

extern "C" __global__ void __miss__hemisphere() {
}
