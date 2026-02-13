#pragma once
#include "../mesh/MeshData.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace OBJExporter {

// Export OBJ with smooth per-vertex colors from per-triangle radiosity.
// Welds coincident vertices on the same surface (position + face-normal
// direction + material) and computes area-weighted average colors.
// Uses the 'v x y z r g b' vertex-color extension (MeshLab, Blender, etc.).
inline bool exportSmoothedOBJ(const std::string& filepath,
                               const Mesh& mesh,
                               const std::vector<Vec3>& triangleColors,
                               float positionEps = 1e-5f)
{
    auto clampF = [](float v) -> float { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
    auto quantPos  = [&](float v) -> int64_t  { return static_cast<int64_t>(std::llround(double(v) / double(positionEps))); };
    auto quantNorm = [](float v) -> int32_t   { return static_cast<int32_t>(std::llround(double(v) * 10.0)); };

    struct Key {
        int64_t px, py, pz;
        int32_t nx, ny, nz;
        uint32_t mat;
        bool operator==(const Key& o) const {
            return px == o.px && py == o.py && pz == o.pz &&
                   nx == o.nx && ny == o.ny && nz == o.nz && mat == o.mat;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const noexcept {
            size_t h = std::hash<int64_t>{}(k.px);
            h ^= std::hash<int64_t>{}(k.py) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int64_t>{}(k.pz) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>{}(k.nx) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>{}(k.ny) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>{}(k.nz) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>{}(k.mat) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };

    std::unordered_map<Key, uint32_t, KeyHash> vertexMap;
    std::vector<Vec3>  positions;
    std::vector<Vec3>  colorSums;
    std::vector<float> weightSums;
    std::vector<TriIdx> outIndices;
    outIndices.reserve(mesh.numTriangles());

    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 vp[3] = { mesh.vertices[tri.i0].toVec3(),
                       mesh.vertices[tri.i1].toVec3(),
                       mesh.vertices[tri.i2].toVec3() };
        float area = MathUtils::triangleArea(vp[0], vp[1], vp[2]);
        const Vec3& color = (i < triangleColors.size()) ? triangleColors[i] : Vec3(0.8f);
        const Vec3& n = mesh.triangle_normal[i];
        uint32_t matID = (i < mesh.triangle_material_id.size())
                         ? mesh.triangle_material_id[i] : 0u;

        uint32_t newIdx[3];
        for (int k = 0; k < 3; ++k) {
            Key key{ quantPos(vp[k].x), quantPos(vp[k].y), quantPos(vp[k].z),
                     quantNorm(n.x),    quantNorm(n.y),    quantNorm(n.z), matID };
            auto it = vertexMap.find(key);
            if (it == vertexMap.end()) {
                uint32_t cid = static_cast<uint32_t>(positions.size());
                vertexMap[key] = cid;
                positions.push_back(vp[k]);
                colorSums.push_back(color * area);
                weightSums.push_back(area);
                newIdx[k] = cid;
            } else {
                uint32_t cid = it->second;
                colorSums[cid] += color * area;
                weightSums[cid] += area;
                newIdx[k] = cid;
            }
        }
        outIndices.push_back(TriIdx(newIdx[0], newIdx[1], newIdx[2]));
    }

    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Failed to create OBJ: " << filepath << "\n";
        return false;
    }

    f << "# Radiosity Cornell Box (smooth per-vertex colors)\n";
    for (size_t i = 0; i < positions.size(); ++i) {
        const Vec3& p = positions[i];
        Vec3 c = (weightSums[i] > 1e-8f) ? colorSums[i] / weightSums[i] : Vec3(0.8f);
        f << "v " << p.x << " " << p.y << " " << p.z
          << " " << clampF(c.x) << " " << clampF(c.y) << " " << clampF(c.z) << "\n";
    }
    f << "\n";
    for (const auto& tri : outIndices)
        f << "f " << (tri.i0 + 1) << " " << (tri.i1 + 1) << " " << (tri.i2 + 1) << "\n";
    f.close();

    std::cout << "Exported OBJ: " << filepath
              << " (" << positions.size() << " vertices, "
              << outIndices.size() << " triangles)\n";
    return true;
}

} // namespace OBJExporter
