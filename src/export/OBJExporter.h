#pragma once
#include "../mesh/MeshData.h"
#include "../scene/CornellBox.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <map>
#include <unordered_map>
#include <vector>

namespace OBJExporter {

// Material name for each material ID.
inline const char* materialName(uint32_t matID) {
    switch (matID) {
        case CornellBox::MAT_WHITE:     return "White";
        case CornellBox::MAT_RED_WALL:  return "Red";
        case CornellBox::MAT_GREEN_WALL:return "Green";
        case CornellBox::MAT_LIGHT:     return "Light";
        case CornellBox::MAT_SHORT_BOX: return "ShortBox";
        case CornellBox::MAT_TALL_BOX:  return "TallBox";
        default:                        return "White";
    }
}

// Object/group name for each material ID.
inline const char* objectName(uint32_t matID) {
    switch (matID) {
        case CornellBox::MAT_WHITE:     return "Room";
        case CornellBox::MAT_RED_WALL:  return "LeftWall";
        case CornellBox::MAT_GREEN_WALL:return "RightWall";
        case CornellBox::MAT_LIGHT:     return "CeilingLight";
        case CornellBox::MAT_SHORT_BOX: return "ShortBox";
        case CornellBox::MAT_TALL_BOX:  return "TallBox";
        default:                        return "Room";
    }
}

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

// ---------------------------------------------------------------------------
// Export Wavefront MTL file with material definitions from the mesh.
// ---------------------------------------------------------------------------
inline bool exportMTL(const std::string& filepath, const Mesh& mesh) {
    // Collect all unique material IDs present in the mesh.
    std::map<uint32_t, CornellBox::Material> materials;
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        uint32_t matID = (i < mesh.triangle_material_id.size())
                         ? mesh.triangle_material_id[i] : 0u;
        if (materials.find(matID) == materials.end()) {
            materials[matID] = CornellBox::getMaterialForID(matID);
        }
    }

    std::ofstream f(filepath);
    if (!f.is_open()) {
        std::cerr << "Failed to create MTL: " << filepath << "\n";
        return false;
    }

    f << "# Cornell Box materials (generated from subdivision mesh)\n";
    f << "# " << mesh.numTriangles() << " triangles, "
      << materials.size() << " materials\n\n";

    for (const auto& [matID, mat] : materials) {
        const char* name = materialName(matID);
        f << "newmtl " << name << "\n";
        f << "Kd " << mat.diffuse.x << " " << mat.diffuse.y
          << " " << mat.diffuse.z << "\n";
        f << "Ka 0.0000 0.0000 0.0000\n";
        f << "Ks 0.0000 0.0000 0.0000\n";
        if (mat.emission.x > 0.0f || mat.emission.y > 0.0f || mat.emission.z > 0.0f) {
            f << "Ke " << mat.emission.x << " " << mat.emission.y
              << " " << mat.emission.z << "\n";
        }
        f << "illum 1\n\n";
    }

    f.close();
    std::cout << "Exported MTL: " << filepath
              << " (" << materials.size() << " materials)\n";
    return true;
}

// ---------------------------------------------------------------------------
// Export OBJ with material references (mtllib + usemtl), grouped by object.
// Vertex positions only (no baked colors); materials come from the .mtl file.
// ---------------------------------------------------------------------------
inline bool exportOBJWithMaterials(const std::string& objPath,
                                    const std::string& mtlFilename,
                                    const Mesh& mesh,
                                    float positionEps = 1e-5f)
{
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

    // Weld coincident vertices (same position + face normal + material).
    std::unordered_map<Key, uint32_t, KeyHash> vertexMap;
    std::vector<Vec3> positions;

    // Group triangles by material ID.
    std::map<uint32_t, std::vector<TriIdx>> groups;

    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 vp[3] = { mesh.vertices[tri.i0].toVec3(),
                       mesh.vertices[tri.i1].toVec3(),
                       mesh.vertices[tri.i2].toVec3() };
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
                newIdx[k] = cid;
            } else {
                newIdx[k] = it->second;
            }
        }
        groups[matID].push_back(TriIdx(newIdx[0], newIdx[1], newIdx[2]));
    }

    std::ofstream f(objPath);
    if (!f.is_open()) {
        std::cerr << "Failed to create OBJ: " << objPath << "\n";
        return false;
    }

    f << "# Cornell Box (subdivision mesh with materials)\n";
    f << "mtllib " << mtlFilename << "\n\n";

    // Write all vertices.
    for (const auto& p : positions) {
        f << "v " << p.x << " " << p.y << " " << p.z << "\n";
    }
    f << "\n";

    // Write faces grouped by material/object.
    size_t totalFaces = 0;
    for (const auto& [matID, tris] : groups) {
        f << "o " << objectName(matID) << "\n";
        f << "usemtl " << materialName(matID) << "\n";
        for (const auto& tri : tris) {
            f << "f " << (tri.i0 + 1) << " " << (tri.i1 + 1) << " " << (tri.i2 + 1) << "\n";
        }
        f << "\n";
        totalFaces += tris.size();
    }

    f.close();
    std::cout << "Exported OBJ+MTL: " << objPath
              << " (" << positions.size() << " vertices, "
              << totalFaces << " triangles, "
              << groups.size() << " objects)\n";
    return true;
}

} // namespace OBJExporter
