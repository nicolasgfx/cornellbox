#pragma once
#include "../mesh/MeshData.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <iostream>

namespace AdaptiveRefinement {

// --------------------------------------------------------------------------
// Configuration
// --------------------------------------------------------------------------
struct Options {
    float targetArea      = 0.5f;   // max triangle area (scene-scale dependent)
    float minEdgeLength   = 0.01f;  // don't split edges shorter than this
    uint32_t maxTriangles = 2000000; // hard cap on total triangles
    uint32_t maxDepth     = 12;     // max subdivision depth
    float proximityFactor = 1.0f;   // refine when gap < proximityFactor * sqrt(area)
    float gradientThresh  = 0.0f;   // future: radiosity gradient threshold
    uint32_t perPassCap   = 0;      // 0 = no cap, else max refines per batch
    float maxEdgeRatio    = 0.0f;   // max longest/shortest edge ratio (0=disabled)
};

// --------------------------------------------------------------------------
// Edge key for EdgeSplitMap: undirected edge (min, max)
// --------------------------------------------------------------------------
struct EdgeKey {
    uint32_t a, b;

    EdgeKey(uint32_t v0, uint32_t v1)
        : a(std::min(v0, v1)), b(std::max(v0, v1)) {}

    bool operator==(const EdgeKey& o) const { return a == o.a && b == o.b; }
};

struct EdgeHash {
    size_t operator()(const EdgeKey& k) const noexcept {
        size_t h = std::hash<uint32_t>{}(k.a);
        h ^= std::hash<uint32_t>{}(k.b) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

using EdgeSplitMap = std::unordered_map<EdgeKey, uint32_t, EdgeHash>;

// --------------------------------------------------------------------------
// Create midpoint vertex for an edge, or return existing from map.
// --------------------------------------------------------------------------
inline uint32_t getOrCreateMidpoint(EdgeSplitMap& splitMap,
                                     std::vector<Vertex>& vertices,
                                     uint32_t v0, uint32_t v1) {
    EdgeKey key(v0, v1);
    auto it = splitMap.find(key);
    if (it != splitMap.end()) return it->second;

    Vec3 p0 = vertices[v0].toVec3();
    Vec3 p1 = vertices[v1].toVec3();
    Vec3 mid = (p0 + p1) * 0.5f;
    uint32_t idx = static_cast<uint32_t>(vertices.size());
    vertices.push_back(Vertex(mid));
    splitMap[key] = idx;
    return idx;
}

// --------------------------------------------------------------------------
// Conforming subdivision: apply templates based on which edges are split.
//
// Triangle (v0, v1, v2), edges e01, e12, e20.
// For each triangle, check which of its edges appear in the EdgeSplitMap.
// Apply the corresponding template (0/1/2/3 split edges).
// --------------------------------------------------------------------------
inline void applyTemplates(const std::vector<TriIdx>& oldTris,
                           const std::vector<uint32_t>& oldMats,
                           const std::vector<uint8_t>& oldSubdivFlags,
                           EdgeSplitMap& splitMap,
                           std::vector<Vertex>& vertices,
                           std::vector<TriIdx>& newTris,
                           std::vector<uint32_t>& newMats,
                           std::vector<uint8_t>& newSubdivFlags) {
    newTris.reserve(oldTris.size() * 2);
    newMats.reserve(oldTris.size() * 2);
    newSubdivFlags.reserve(oldTris.size() * 2);

    for (size_t i = 0; i < oldTris.size(); ++i) {
        uint32_t v0 = oldTris[i].i0;
        uint32_t v1 = oldTris[i].i1;
        uint32_t v2 = oldTris[i].i2;
        uint32_t mat = (i < oldMats.size()) ? oldMats[i] : 0u;
        uint8_t wasSub = (i < oldSubdivFlags.size()) ? oldSubdivFlags[i] : 0u;

        // Check which edges are split.
        bool s01 = splitMap.count(EdgeKey(v0, v1)) > 0;
        bool s12 = splitMap.count(EdgeKey(v1, v2)) > 0;
        bool s20 = splitMap.count(EdgeKey(v2, v0)) > 0;
        int splitCount = (int)s01 + (int)s12 + (int)s20;

        if (splitCount == 0) {
            // No splits: keep triangle.
            newTris.push_back(TriIdx(v0, v1, v2));
            newMats.push_back(mat);
            newSubdivFlags.push_back(wasSub);
        } else if (splitCount == 3) {
            // All 3 edges split: 1-to-4 subdivision.
            uint32_t m01 = splitMap[EdgeKey(v0, v1)];
            uint32_t m12 = splitMap[EdgeKey(v1, v2)];
            uint32_t m20 = splitMap[EdgeKey(v2, v0)];
            newTris.push_back(TriIdx(v0,  m01, m20));
            newTris.push_back(TriIdx(m01, v1,  m12));
            newTris.push_back(TriIdx(m20, m12, v2));
            newTris.push_back(TriIdx(m01, m12, m20));
            newMats.push_back(mat); newMats.push_back(mat);
            newMats.push_back(mat); newMats.push_back(mat);
            newSubdivFlags.push_back(1); newSubdivFlags.push_back(1);
            newSubdivFlags.push_back(1); newSubdivFlags.push_back(1);
        } else if (splitCount == 1) {
            // 1 edge split: 1-to-2
            if (s01) {
                uint32_t m = splitMap[EdgeKey(v0, v1)];
                newTris.push_back(TriIdx(v0, m,  v2));
                newTris.push_back(TriIdx(m,  v1, v2));
            } else if (s12) {
                uint32_t m = splitMap[EdgeKey(v1, v2)];
                newTris.push_back(TriIdx(v1, m,  v0));
                newTris.push_back(TriIdx(m,  v2, v0));
            } else { // s20
                uint32_t m = splitMap[EdgeKey(v2, v0)];
                newTris.push_back(TriIdx(v2, m,  v1));
                newTris.push_back(TriIdx(m,  v0, v1));
            }
            newMats.push_back(mat);
            newMats.push_back(mat);
            newSubdivFlags.push_back(1);
            newSubdivFlags.push_back(1);
        } else { // splitCount == 2
            // 2 edges split: 1-to-3
            // Rotate so the un-split edge is e12 (between v1 and v2).
            uint32_t a = v0, b = v1, c = v2;
            if (!s01 && s12 && s20) {
                // e01 unsplit -> rotate: a=v1, b=v2, c=v0
                a = v1; b = v2; c = v0;
                // split edges: e_bc (v2,v0) and e_ca (v0,v1)... wait
                // Actually we need the two split edges to be e_ab and e_ca.
                // Let me re-derive: we want the vertex OPPOSITE the unsplit edge.
                // If e01 is unsplit: split are e12 and e20.
                // Opposite vertex to e01 is v2. Put v2 as 'a'.
                a = v2; b = v0; c = v1;
            } else if (s01 && !s12 && s20) {
                // e12 unsplit: split are e01 and e20. Opposite vertex is v0.
                a = v0; b = v1; c = v2;
            } else { // s01 && s12 && !s20
                // e20 unsplit: split are e01 and e12. Opposite vertex is v1.
                a = v1; b = v2; c = v0;
            }
            // Now: edges ab and ca are split, edge bc is unsplit.
            // Vertex a is opposite the unsplit edge.
            uint32_t mab = splitMap[EdgeKey(a, b)];
            uint32_t mca = splitMap[EdgeKey(c, a)];
            newTris.push_back(TriIdx(a,   mab, mca));
            newTris.push_back(TriIdx(mab, b,   c));
            newTris.push_back(TriIdx(mca, mab, c));
            newMats.push_back(mat);
            newMats.push_back(mat);
            newMats.push_back(mat);
            newSubdivFlags.push_back(1);
            newSubdivFlags.push_back(1);
            newSubdivFlags.push_back(1);
        }
    }
}

// --------------------------------------------------------------------------
// Uniform area subdivision: split all triangles above targetArea.
// Uses conforming EdgeSplitMap to avoid T-junctions.
// Iterates in batches until all triangles are below threshold or limits hit.
// --------------------------------------------------------------------------
inline Mesh subdivideUniform(const Mesh& input, const Options& opts) {
    Mesh mesh = input;

    for (uint32_t depth = 0; depth < opts.maxDepth; ++depth) {
        if (mesh.numTriangles() >= opts.maxTriangles) break;

        // Identify triangles that need refinement.
        EdgeSplitMap splitMap;
        uint32_t refineCount = 0;
        uint32_t areaRefineCount = 0;
        uint32_t shapeRefineCount = 0;

        for (size_t i = 0; i < mesh.numTriangles(); ++i) {
            const auto& tri = mesh.indices[i];
            Vec3 v0 = mesh.vertices[tri.i0].toVec3();
            Vec3 v1 = mesh.vertices[tri.i1].toVec3();
            Vec3 v2 = mesh.vertices[tri.i2].toVec3();
            float area = MathUtils::triangleArea(v0, v1, v2);

            float e01 = (v1 - v0).length();
            float e12 = (v2 - v1).length();
            float e20 = (v0 - v2).length();

            // Check aspect ratio: longest / shortest edge.
            float longest  = std::max({e01, e12, e20});
            float shortest = std::min({e01, e12, e20});
            bool needsShapeRefine = (opts.maxEdgeRatio > 0.0f
                                     && shortest > 1e-10f
                                     && longest / shortest > opts.maxEdgeRatio
                                     && longest > opts.minEdgeLength);

            if (area > opts.targetArea) {
                // Mark all 3 edges for splitting (1-to-4).
                if (e01 > opts.minEdgeLength)
                    getOrCreateMidpoint(splitMap, mesh.vertices, tri.i0, tri.i1);
                if (e12 > opts.minEdgeLength)
                    getOrCreateMidpoint(splitMap, mesh.vertices, tri.i1, tri.i2);
                if (e20 > opts.minEdgeLength)
                    getOrCreateMidpoint(splitMap, mesh.vertices, tri.i2, tri.i0);
                refineCount++;
                areaRefineCount++;
            } else if (needsShapeRefine) {
                // Split only the longest edge to improve triangle shape.
                if (e01 >= e12 && e01 >= e20)
                    getOrCreateMidpoint(splitMap, mesh.vertices, tri.i0, tri.i1);
                else if (e12 >= e01 && e12 >= e20)
                    getOrCreateMidpoint(splitMap, mesh.vertices, tri.i1, tri.i2);
                else
                    getOrCreateMidpoint(splitMap, mesh.vertices, tri.i2, tri.i0);
                refineCount++;
                shapeRefineCount++;
            }
        }

        if (refineCount == 0) break; // All triangles below threshold.

        // Apply conforming templates.
        std::vector<TriIdx> newTris;
        std::vector<uint32_t> newMats;
        std::vector<uint8_t> newSubdivFlags;
        applyTemplates(mesh.indices, mesh.triangle_material_id,
                       mesh.triangle_is_subdivided,
                       splitMap, mesh.vertices, newTris, newMats, newSubdivFlags);

        mesh.indices = std::move(newTris);
        mesh.triangle_material_id = std::move(newMats);
        mesh.triangle_is_subdivided = std::move(newSubdivFlags);

        std::cout << "  Refine pass " << (depth + 1) << ": "
                  << mesh.numTriangles() << " tris ("
                  << refineCount << " refined: "
                  << areaRefineCount << " area, "
                  << shapeRefineCount << " shape)\n";

        if (mesh.numTriangles() >= opts.maxTriangles) {
            std::cout << "  Triangle budget reached (" << opts.maxTriangles << ")\n";
            break;
        }
    }

    size_t subdivCount = (size_t)std::count(
        mesh.triangle_is_subdivided.begin(),
        mesh.triangle_is_subdivided.end(), (uint8_t)1);
    std::cout << "  Uniform: " << input.numTriangles() << " -> "
              << mesh.numTriangles() << " tris ("
              << subdivCount << " new, "
              << (mesh.numTriangles() > 0
                  ? (int)(100.0 * subdivCount / mesh.numTriangles()) : 0)
              << "% refined)\n";

    return mesh;
}

// --------------------------------------------------------------------------
// Compute per-triangle proximity score: minimum distance to a non-adjacent
// triangle. Uses a simple O(N²) centroid-distance check (practical for
// scenes up to ~100k triangles before refinement).
//
// Returns distance to nearest non-adjacent triangle centroid.
// --------------------------------------------------------------------------
inline float computeProximityScore(const Mesh& mesh, size_t triIdx,
                                    float searchRadius) {
    const Vec3& c = mesh.triangle_centroid[triIdx];
    float minDist = searchRadius;

    // Build adjacency set: triangles sharing a vertex with triIdx.
    const auto& tri = mesh.indices[triIdx];
    uint32_t verts[3] = { tri.i0, tri.i1, tri.i2 };

    for (size_t j = 0; j < mesh.numTriangles(); ++j) {
        if (j == triIdx) continue;

        // Skip if sharing a vertex (adjacent).
        const auto& other = mesh.indices[j];
        bool adjacent = false;
        for (int k = 0; k < 3 && !adjacent; ++k) {
            if (verts[k] == other.i0 || verts[k] == other.i1 || verts[k] == other.i2)
                adjacent = true;
        }
        if (adjacent) continue;

        float d = (c - mesh.triangle_centroid[j]).length();
        if (d < minDist) minDist = d;
    }
    return minDist;
}

// --------------------------------------------------------------------------
// Adaptive proximity-based refinement: refine triangles that are close
// to non-adjacent geometry (shadow boundaries, near-contact regions).
//
// This does a pre-pass computing triangle data, then iteratively refines.
// --------------------------------------------------------------------------
inline Mesh refineByProximity(const Mesh& input, const Options& opts) {
    Mesh mesh = input;

    for (uint32_t depth = 0; depth < opts.maxDepth; ++depth) {
        if (mesh.numTriangles() >= opts.maxTriangles) break;

        // Recompute triangle geometry for scoring.
        size_t N = mesh.numTriangles();
        mesh.triangle_area.resize(N);
        mesh.triangle_centroid.resize(N);
        mesh.triangle_normal.resize(N);
        for (size_t i = 0; i < N; ++i) {
            const auto& tri = mesh.indices[i];
            Vec3 v0 = mesh.vertices[tri.i0].toVec3();
            Vec3 v1 = mesh.vertices[tri.i1].toVec3();
            Vec3 v2 = mesh.vertices[tri.i2].toVec3();
            mesh.triangle_area[i]     = MathUtils::triangleArea(v0, v1, v2);
            mesh.triangle_centroid[i] = MathUtils::triangleCentroid(v0, v1, v2);
            mesh.triangle_normal[i]   = MathUtils::triangleNormal(v0, v1, v2).normalized();
        }

        // Score triangles for proximity refinement.
        Vec3 extent = Vec3(0);
        for (size_t i = 0; i < mesh.vertices.size(); ++i) {
            Vec3 p = mesh.vertices[i].toVec3();
            extent.x = std::max(extent.x, std::abs(p.x));
            extent.y = std::max(extent.y, std::abs(p.y));
            extent.z = std::max(extent.z, std::abs(p.z));
        }
        float sceneScale = std::max({extent.x, extent.y, extent.z});
        float searchRadius = sceneScale * 0.1f; // only check nearby geometry

        EdgeSplitMap splitMap;
        uint32_t refineCount = 0;

        for (size_t i = 0; i < N; ++i) {
            float area = mesh.triangle_area[i];
            if (area <= opts.minEdgeLength * opts.minEdgeLength) continue;

            float patchSize = std::sqrt(area);
            float proximity = computeProximityScore(mesh, i, searchRadius);

            // Refine if non-adjacent geometry is closer than
            // proximityFactor * patchSize.
            if (proximity < opts.proximityFactor * patchSize &&
                area > opts.targetArea * 0.1f) { // don't go below 10% of base target
                const auto& tri = mesh.indices[i];
                Vec3 v0 = mesh.vertices[tri.i0].toVec3();
                Vec3 v1 = mesh.vertices[tri.i1].toVec3();
                Vec3 v2 = mesh.vertices[tri.i2].toVec3();

                float e01 = (v1 - v0).length();
                float e12 = (v2 - v1).length();
                float e20 = (v0 - v2).length();

                // Longest-edge bisection.
                float maxEdge = std::max({e01, e12, e20});
                if (maxEdge > opts.minEdgeLength) {
                    if (e01 == maxEdge)
                        getOrCreateMidpoint(splitMap, mesh.vertices, tri.i0, tri.i1);
                    else if (e12 == maxEdge)
                        getOrCreateMidpoint(splitMap, mesh.vertices, tri.i1, tri.i2);
                    else
                        getOrCreateMidpoint(splitMap, mesh.vertices, tri.i2, tri.i0);
                    refineCount++;
                }
            }
        }

        if (refineCount == 0) break;
        if (opts.perPassCap > 0 && refineCount > opts.perPassCap) {
            // Would need priority queue — skip for now, just limit passes.
        }

        std::vector<TriIdx> newTris;
        std::vector<uint32_t> newMats;
        std::vector<uint8_t> newSubdivFlags;
        applyTemplates(mesh.indices, mesh.triangle_material_id,
                       mesh.triangle_is_subdivided,
                       splitMap, mesh.vertices, newTris, newMats, newSubdivFlags);

        mesh.indices = std::move(newTris);
        mesh.triangle_material_id = std::move(newMats);
        mesh.triangle_is_subdivided = std::move(newSubdivFlags);

        std::cout << "  Proximity pass " << (depth + 1) << ": "
                  << mesh.numTriangles() << " tris ("
                  << refineCount << " refined)\n";
    }

    size_t subdivCount = (size_t)std::count(
        mesh.triangle_is_subdivided.begin(),
        mesh.triangle_is_subdivided.end(), (uint8_t)1);
    std::cout << "  Proximity: " << input.numTriangles() << " -> "
              << mesh.numTriangles() << " tris ("
              << subdivCount << " new, "
              << (mesh.numTriangles() > 0
                  ? (int)(100.0 * subdivCount / mesh.numTriangles()) : 0)
              << "% refined)\n";

    return mesh;
}

} // namespace AdaptiveRefinement
