#pragma once
// ---------------------------------------------------------------------------
// CPU form-factor computation — extracted from main.cpp for testability.
//
// Centroid-to-centroid geometric form factor:
//   F_ij ≈ cos θ_i · cos θ_j · A_j / (π r²)
//
// Parameterised via Options so tests can isolate individual clamps/corrections.
// ---------------------------------------------------------------------------

#include "../mesh/MeshData.h"
#include "../app/Config.h"
#include <vector>
#include <cmath>
#include <algorithm>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace FormFactor {

inline constexpr float kSenderHalfspaceEps = 1e-6f;

// Tunables exposed for test isolation.
struct Options {
    float distanceSoftening       = kDistanceSoftening;  // added to r² in denominator
    bool  clampEnergyConservation = true;                // row sum ≤ 1
    bool  clampIndividualFF       = true;                // single F_ij ≤ 1
};

// Compute one full row of form factors from source patch `sourceId` to all
// other triangles in `mesh`.  Results are written to `row[j]`.
inline void computeRowCPU(const Mesh& mesh,
                           uint32_t sourceId,
                           std::vector<float>& row,
                           const Options& opts = {}) {
    const uint32_t N = static_cast<uint32_t>(mesh.numTriangles());
    row.assign(N, 0.0f);
    const Vec3& ni = mesh.triangle_normal[sourceId];

    for (uint32_t j = 0; j < N; ++j) {
        if (j == sourceId) continue;

        Vec3 r = mesh.triangle_centroid[j] - mesh.triangle_centroid[sourceId];
        float dist2 = r.dot(r);
        if (dist2 < 1e-10f) continue;
        if (ni.dot(r) <= kSenderHalfspaceEps) continue;   // behind sender

        float dist = std::sqrt(dist2);
        Vec3 rn = r / dist;
        float cosI = ni.dot(rn);
        float cosJ = mesh.triangle_normal[j].dot(rn * (-1.0f));
        if (cosI <= 0.0f || cosJ <= 0.0f) continue;

        float Aj = std::max(mesh.triangle_area[j], 1e-12f);
        // No area-dependent distance floor — distanceSoftening regularises
        // the denominator without introducing subdivision-variant bias.
        float ff = (cosI * cosJ * Aj)
                 / (float(M_PI) * (dist2 + opts.distanceSoftening));

        if (opts.clampIndividualFF && ff > 1.0f) ff = 1.0f;
        if (ff < 1e-8f) continue;
        row[j] = ff;
    }

    // Energy conservation: row sum must be ≤ 1 for closed geometry.
    if (opts.clampEnergyConservation) {
        float sum = 0.0f;
        for (float v : row) sum += v;
        if (sum > 1.0f) {
            float inv = 1.0f / sum;
            for (float& v : row) v *= inv;
        }
    }
}

} // namespace FormFactor
