#pragma once
// ---------------------------------------------------------------------------
// Form-factor-driven adaptive subdivision.
//
// Key idea: the centroid-to-centroid form factor approximation breaks down
// when a patch's characteristic size (sqrt(area)) is comparable to or larger
// than the distance to an interacting patch.  This module iteratively
// identifies such pairs and subdivides the larger patch (or both) until the
// geometric accuracy criterion is met.
//
// This naturally handles:
//   - Near-contact regions (couch legs on floor)
//   - Non-coplanar adjacent geometry (wall corners, folds)
//   - Any pair that actually exchanges energy (form factor > 0)
//
// A spatial hash grid avoids O(N²) pair checks.
// ---------------------------------------------------------------------------

#include "../mesh/MeshData.h"
#include "../mesh/AdaptiveRefinement.h"   // EdgeSplitMap, applyTemplates, getOrCreateMidpoint
#include "../math/Vec3.h"
#include "../math/MathUtils.h"

#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <iostream>
#include <iomanip>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace FormFactorRefinement {

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------
struct Options {
    // Core accuracy criterion: subdivide when
    //   max(sqrt(area_i), sqrt(area_j)) / dist(i,j) > accuracyRatio
    // for a pair with non-trivial form factor.
    // Lower = more aggressive refinement = higher quality.
    float accuracyRatio     = 0.5f;

    // Minimum geometric form factor to consider a pair "interacting".
    float minFormFactor     = 1e-5f;

    // Don't split edges shorter than this.
    float minEdgeLength     = 0.01f;

    // Don't subdivide triangles below this area.
    float minArea           = 1e-8f;

    // Hard limits.
    uint32_t maxTriangles   = 2000000;
    uint32_t maxPasses      = 10;

    // Use 1-to-4 split (all 3 edges) vs longest-edge bisection.
    bool splitAll3Edges     = false;   // longest-edge by default

    // When both patches in a pair violate the accuracy criterion,
    // should we refine both or only the larger one?
    bool refineBoth         = true;

    // Search radius multiplier: only check triangles within
    // this factor of the current patch's characteristic size.
    // Larger = more pairs checked, slower but more thorough.
    float searchRadiusMultiplier = 4.0f;
};

// ---------------------------------------------------------------------------
// Simple spatial hash grid for fast neighbor lookups.
// ---------------------------------------------------------------------------
struct SpatialGrid {
    float cellSize;
    float invCellSize;
    Vec3  origin;

    struct CellKey {
        int32_t x, y, z;
        bool operator==(const CellKey& o) const { return x == o.x && y == o.y && z == o.z; }
    };
    struct CellHash {
        size_t operator()(const CellKey& k) const noexcept {
            size_t h = std::hash<int32_t>{}(k.x);
            h ^= std::hash<int32_t>{}(k.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>{}(k.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<CellKey, std::vector<uint32_t>, CellHash> cells;

    void build(const std::vector<Vec3>& centroids, float cell) {
        cellSize = cell;
        invCellSize = 1.0f / cell;
        cells.clear();
        cells.reserve(centroids.size());
        for (uint32_t i = 0; i < static_cast<uint32_t>(centroids.size()); ++i) {
            CellKey k = toCell(centroids[i]);
            cells[k].push_back(i);
        }
    }

    CellKey toCell(const Vec3& p) const {
        return {
            static_cast<int32_t>(std::floor(p.x * invCellSize)),
            static_cast<int32_t>(std::floor(p.y * invCellSize)),
            static_cast<int32_t>(std::floor(p.z * invCellSize))
        };
    }

    // Call `func(uint32_t j)` for every triangle in cells overlapping a sphere
    // of radius `radius` centered at `center`.
    template <typename Func>
    void queryRadius(const Vec3& center, float radius, Func func) const {
        int32_t rCells = static_cast<int32_t>(std::ceil(radius * invCellSize));
        CellKey c0 = toCell(center);
        for (int32_t dz = -rCells; dz <= rCells; ++dz)
        for (int32_t dy = -rCells; dy <= rCells; ++dy)
        for (int32_t dx = -rCells; dx <= rCells; ++dx) {
            CellKey k{ c0.x + dx, c0.y + dy, c0.z + dz };
            auto it = cells.find(k);
            if (it != cells.end()) {
                for (uint32_t idx : it->second) {
                    func(idx);
                }
            }
        }
    }
};

// ---------------------------------------------------------------------------
// Check if two triangles share a vertex (topologically adjacent).
// ---------------------------------------------------------------------------
inline bool sharesVertex(const TriIdx& a, const TriIdx& b) {
    uint32_t va[3] = { a.i0, a.i1, a.i2 };
    uint32_t vb[3] = { b.i0, b.i1, b.i2 };
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (va[i] == vb[j]) return true;
    return false;
}

// ---------------------------------------------------------------------------
// Check if two triangles share an edge (strongly adjacent).
// ---------------------------------------------------------------------------
inline int sharedVertexCount(const TriIdx& a, const TriIdx& b) {
    uint32_t va[3] = { a.i0, a.i1, a.i2 };
    uint32_t vb[3] = { b.i0, b.i1, b.i2 };
    int count = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            if (va[i] == vb[j]) count++;
    return count;
}

// ---------------------------------------------------------------------------
// Fast geometric form factor estimate (centroid-to-centroid, no visibility).
// Returns the approximate F_ij = cos_i * cos_j * A_j / (pi * r²).
// Also returns the distance (via out parameter).
// ---------------------------------------------------------------------------
inline float estimateFormFactor(const Vec3& ci, const Vec3& ni, float /*ai*/,
                                 const Vec3& cj, const Vec3& nj, float aj,
                                 float& dist_out) {
    Vec3 r = cj - ci;
    float dist2 = r.dot(r);
    if (dist2 < 1e-12f) { dist_out = 0.0f; return 0.0f; }

    float dist = std::sqrt(dist2);
    dist_out = dist;
    Vec3 rn = r / dist;

    float cosI = ni.dot(rn);
    float cosJ = nj.dot(rn * (-1.0f));

    if (cosI <= 0.0f || cosJ <= 0.0f) return 0.0f;

    float ff = cosI * cosJ * aj / (static_cast<float>(M_PI) * dist2);
    return ff;
}

// ---------------------------------------------------------------------------
// Core: form-factor-driven adaptive refinement.
//
// For each triangle, find nearby non-adjacent triangles with non-trivial
// form factor.  If the patch-size-to-distance ratio exceeds the accuracy
// threshold, mark the larger patch (or both) for subdivision.
//
// Uses a spatial hash grid for O(N)-amortised pair finding.
// ---------------------------------------------------------------------------
inline Mesh refineByFormFactor(const Mesh& input, const Options& opts) {
    Mesh mesh = input;

    for (uint32_t pass = 0; pass < opts.maxPasses; ++pass) {
        if (mesh.numTriangles() >= opts.maxTriangles) break;

        const size_t N = mesh.numTriangles();

        // 1) Recompute triangle geometry.
        mesh.triangle_area.resize(N);
        mesh.triangle_centroid.resize(N);
        mesh.triangle_normal.resize(N);
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            const auto& tri = mesh.indices[i];
            Vec3 v0 = mesh.vertices[tri.i0].toVec3();
            Vec3 v1 = mesh.vertices[tri.i1].toVec3();
            Vec3 v2 = mesh.vertices[tri.i2].toVec3();
            mesh.triangle_area[i]     = MathUtils::triangleArea(v0, v1, v2);
            mesh.triangle_centroid[i] = MathUtils::triangleCentroid(v0, v1, v2);
            mesh.triangle_normal[i]   = MathUtils::triangleNormal(v0, v1, v2).normalized();
        }

        // 2) Compute per-triangle characteristic size.
        std::vector<float> patchSize(N);
        float maxPatchSize = 0.0f;
        for (size_t i = 0; i < N; ++i) {
            patchSize[i] = std::sqrt(std::max(mesh.triangle_area[i], 0.0f));
            maxPatchSize = std::max(maxPatchSize, patchSize[i]);
        }

        // 3) Build spatial hash grid.
        // Cell size: use a fraction of scene extent to keep cells manageable.
        float gridCellSize = maxPatchSize * opts.searchRadiusMultiplier;
        if (gridCellSize < 1e-6f) break;   // degenerate scene

        SpatialGrid grid;
        grid.build(mesh.triangle_centroid, gridCellSize);

        // 4) Identify triangles needing refinement.
        //    needsRefine[i] is only ever set to true (idempotent), so
        //    concurrent writes from different threads are safe.
        std::vector<uint8_t> needsRefine(N, 0);
        int64_t pairsChecked = 0;
        int64_t pairsTriggered = 0;

#ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, 64) reduction(+:pairsChecked,pairsTriggered)
#endif
        for (int64_t i = 0; i < static_cast<int64_t>(N); ++i) {
            float hi = patchSize[i];
            if (hi < opts.minEdgeLength) continue;
            if (mesh.triangle_area[i] < opts.minArea) continue;

            const Vec3& ci = mesh.triangle_centroid[i];
            const Vec3& ni = mesh.triangle_normal[i];
            float ai = mesh.triangle_area[i];

            // Search radius: we only care about nearby triangles.
            float searchRadius = hi * opts.searchRadiusMultiplier;

            grid.queryRadius(ci, searchRadius, [&](uint32_t j) {
                if (static_cast<int64_t>(j) <= i) return;  // avoid double-counting pairs
                if (mesh.triangle_area[j] < opts.minArea) return;

                const TriIdx& triI = mesh.indices[i];
                const TriIdx& triJ = mesh.indices[j];

                // For edge-adjacent triangles that are nearly coplanar, skip.
                // For non-coplanar adjacent triangles, check form factor accuracy.
                int shared = sharedVertexCount(triI, triJ);
                if (shared >= 2) {
                    // Edge-adjacent: only refine if normals differ significantly
                    // (fold/corner: normals form a meaningful angle).
                    float normalDot = mesh.triangle_normal[i].dot(mesh.triangle_normal[j]);
                    if (normalDot > 0.95f) return;  // nearly coplanar, skip
                }

                float hj = patchSize[j];
                const Vec3& cj = mesh.triangle_centroid[j];
                const Vec3& nj = mesh.triangle_normal[j];
                float aj = mesh.triangle_area[j];

                // Estimate form factor i->j.
                float dist = 0.0f;
                float ff_ij = estimateFormFactor(ci, ni, ai, cj, nj, aj, dist);

                // Also check j->i (asymmetric because areas differ).
                float dist2 = 0.0f;
                float ff_ji = estimateFormFactor(cj, nj, aj, ci, ni, ai, dist2);

                float maxFF = std::max(ff_ij, ff_ji);
                if (maxFF < opts.minFormFactor) return;

                pairsChecked++;

                if (dist < 1e-10f) return;

                // Accuracy criterion: patch size vs distance.
                float maxH = std::max(hi, hj);
                float ratio = maxH / dist;

                if (ratio > opts.accuracyRatio) {
                    pairsTriggered++;

                    if (opts.refineBoth) {
                        // Refine both triangles in the pair.
                        if (hi >= opts.minEdgeLength * 2.0f &&
                            mesh.triangle_area[i] > opts.minArea * 4.0f)
                            needsRefine[i] = 1;
                        if (hj >= opts.minEdgeLength * 2.0f &&
                            mesh.triangle_area[j] > opts.minArea * 4.0f)
                            needsRefine[j] = 1;
                    } else {
                        // Refine only the larger patch.
                        size_t target = (hi >= hj) ? static_cast<size_t>(i) : static_cast<size_t>(j);
                        float ht = (target == static_cast<size_t>(i)) ? hi : hj;
                        float at = (target == static_cast<size_t>(i)) ? ai : aj;
                        if (ht >= opts.minEdgeLength * 2.0f &&
                            at > opts.minArea * 4.0f)
                            needsRefine[target] = 1;
                    }
                }
            });
        }

        uint32_t refineCount = 0;
        for (size_t i = 0; i < N; ++i) if (needsRefine[i] != 0) refineCount++;

        if (refineCount == 0) {
            std::cout << "  FF-refine pass " << (pass + 1)
                      << ": converged (no triangles need refinement)\n";
            break;
        }

        // 5) Build EdgeSplitMap for marked triangles.
        AdaptiveRefinement::EdgeSplitMap splitMap;

        for (size_t i = 0; i < N; ++i) {
            if (needsRefine[i] == 0) continue;

            const auto& tri = mesh.indices[i];
            Vec3 v0 = mesh.vertices[tri.i0].toVec3();
            Vec3 v1 = mesh.vertices[tri.i1].toVec3();
            Vec3 v2 = mesh.vertices[tri.i2].toVec3();

            float e01 = (v1 - v0).length();
            float e12 = (v2 - v1).length();
            float e20 = (v0 - v2).length();

            if (opts.splitAll3Edges) {
                // 1-to-4 split: all 3 edges.
                if (e01 > opts.minEdgeLength)
                    AdaptiveRefinement::getOrCreateMidpoint(splitMap, mesh.vertices, tri.i0, tri.i1);
                if (e12 > opts.minEdgeLength)
                    AdaptiveRefinement::getOrCreateMidpoint(splitMap, mesh.vertices, tri.i1, tri.i2);
                if (e20 > opts.minEdgeLength)
                    AdaptiveRefinement::getOrCreateMidpoint(splitMap, mesh.vertices, tri.i2, tri.i0);
            } else {
                // Longest-edge bisection.
                float maxEdge = std::max({e01, e12, e20});
                if (maxEdge > opts.minEdgeLength) {
                    if (e01 >= e12 && e01 >= e20)
                        AdaptiveRefinement::getOrCreateMidpoint(splitMap, mesh.vertices, tri.i0, tri.i1);
                    else if (e12 >= e01 && e12 >= e20)
                        AdaptiveRefinement::getOrCreateMidpoint(splitMap, mesh.vertices, tri.i1, tri.i2);
                    else
                        AdaptiveRefinement::getOrCreateMidpoint(splitMap, mesh.vertices, tri.i2, tri.i0);
                }
            }
        }

        if (splitMap.empty()) {
            std::cout << "  FF-refine pass " << (pass + 1)
                      << ": no edges to split (triangles too small)\n";
            break;
        }

        // 6) Apply conforming templates (reuses AdaptiveRefinement infrastructure).
        std::vector<TriIdx> newTris;
        std::vector<uint32_t> newMats;
        std::vector<uint8_t> newSubdivFlags;
        AdaptiveRefinement::applyTemplates(
            mesh.indices, mesh.triangle_material_id,
            mesh.triangle_is_subdivided,
            splitMap, mesh.vertices, newTris, newMats, newSubdivFlags);

        uint32_t oldCount = static_cast<uint32_t>(N);
        mesh.indices = std::move(newTris);
        mesh.triangle_material_id = std::move(newMats);
        mesh.triangle_is_subdivided = std::move(newSubdivFlags);

        std::cout << "  FF-refine pass " << (pass + 1) << ": "
                  << oldCount << " -> " << mesh.numTriangles() << " tris ("
                  << refineCount << " refined, "
                  << static_cast<uint32_t>(pairsTriggered) << "/" << static_cast<uint32_t>(pairsChecked) << " pairs triggered)\n";

        if (mesh.numTriangles() >= opts.maxTriangles) {
            std::cout << "  Triangle budget reached (" << opts.maxTriangles << ")\n";
            break;
        }
    }

    size_t subdivCount = static_cast<size_t>(std::count(
        mesh.triangle_is_subdivided.begin(),
        mesh.triangle_is_subdivided.end(), static_cast<uint8_t>(1)));
    std::cout << "  FF-refine total: " << input.numTriangles() << " -> "
              << mesh.numTriangles() << " tris ("
              << subdivCount << " subdivided, "
              << (mesh.numTriangles() > 0
                  ? static_cast<int>(100.0 * subdivCount / mesh.numTriangles()) : 0)
              << "%)\n";

    return mesh;
}

} // namespace FormFactorRefinement
