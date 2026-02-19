#pragma once
#include "../mesh/MeshData.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <filesystem>

namespace OBJLoader {

// --------------------------------------------------------------------------
// MTL material definition (subset relevant to radiosity)
// --------------------------------------------------------------------------
struct MTLMaterial {
    std::string name;
    Vec3 Kd{0.8f, 0.8f, 0.8f};  // diffuse reflectance
    Vec3 Ke{0.0f, 0.0f, 0.0f};  // emission
    Vec3 Ka{0.0f, 0.0f, 0.0f};  // ambient (used as fallback diffuse)
    float d  = 1.0f;             // dissolve (opacity)
    float Tr = 0.0f;             // transparency
};

// --------------------------------------------------------------------------
// Parse a .mtl file.  Returns map: material_name -> MTLMaterial.
// --------------------------------------------------------------------------
inline std::unordered_map<std::string, MTLMaterial>
loadMTL(const std::string& mtlPath) {
    std::unordered_map<std::string, MTLMaterial> materials;
    std::ifstream file(mtlPath);
    if (!file.is_open()) {
        std::cerr << "Warning: cannot open MTL file: " << mtlPath << "\n";
        return materials;
    }

    MTLMaterial* current = nullptr;
    std::string line;
    while (std::getline(file, line)) {
        // Strip leading whitespace and skip comments/blanks.
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        line = line.substr(start);

        std::istringstream iss(line);
        std::string token;
        iss >> token;

        if (token == "newmtl") {
            std::string name;
            iss >> name;
            materials[name] = MTLMaterial{};
            materials[name].name = name;
            current = &materials[name];
        } else if (!current) {
            continue;
        } else if (token == "Kd") {
            iss >> current->Kd.x >> current->Kd.y >> current->Kd.z;
        } else if (token == "Ke") {
            iss >> current->Ke.x >> current->Ke.y >> current->Ke.z;
        } else if (token == "Ka") {
            iss >> current->Ka.x >> current->Ka.y >> current->Ka.z;
        } else if (token == "d") {
            iss >> current->d;
        } else if (token == "Tr") {
            iss >> current->Tr;
        }
        // Ignore map_Kd, Ns, Ni, illum, etc. — not needed for radiosity.
    }
    return materials;
}

// --------------------------------------------------------------------------
// Load result
// --------------------------------------------------------------------------
struct LoadResult {
    Mesh mesh;
    std::vector<MTLMaterial> materialTable;  // indexed by mesh material ID
    Vec3 bboxMin, bboxMax;
    Vec3 centerOffset;                      // translation applied to center model
    uint32_t rawVertices   = 0;
    uint32_t rawFaces      = 0;
    uint32_t quadsTriangulated = 0;
    uint32_t ngonsTriangulated = 0;
    uint32_t degeneratesRemoved = 0;
};

// --------------------------------------------------------------------------
// Load an OBJ file with its associated MTL.
//
// Face formats handled: v, v/vt, v/vt/vn, v//vn
// Polygons: tris kept as-is, quads split by shortest diagonal,
//           n-gons fan-triangulated.
// Degenerates removed. Model centered at origin.
// --------------------------------------------------------------------------
inline LoadResult loadOBJ(const std::string& objPath, float degenerateEps = 1e-8f) {
    LoadResult result;

    std::ifstream file(objPath);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open OBJ file: " << objPath << "\n";
        return result;
    }

    std::string objDir = std::filesystem::path(objPath).parent_path().string();

    // ---- First pass: parse raw data ----
    std::vector<Vec3> positions;
    // Per-face data: list of vertex-position indices + material ID
    struct RawFace {
        std::vector<uint32_t> vIndices;
        uint32_t materialId;
    };
    std::vector<RawFace> rawFaces;

    // Material name -> index mapping
    std::unordered_map<std::string, MTLMaterial> mtlMap;
    std::unordered_map<std::string, uint32_t> matNameToId;
    uint32_t currentMatId = 0;

    // Always have a default material at index 0.
    result.materialTable.push_back(MTLMaterial{"_default"});
    matNameToId["_default"] = 0;

    std::string line;
    while (std::getline(file, line)) {
        size_t start = line.find_first_not_of(" \t\r\n");
        if (start == std::string::npos || line[start] == '#') continue;
        line = line.substr(start);

        if (line.size() < 2) continue;

        if (line[0] == 'v' && line[1] == ' ') {
            // Vertex position
            float x, y, z;
            if (std::sscanf(line.c_str(), "v %f %f %f", &x, &y, &z) == 3) {
                positions.push_back(Vec3(x, y, z));
            }
        } else if (line[0] == 'f' && line[1] == ' ') {
            // Face
            RawFace face;
            face.materialId = currentMatId;

            std::istringstream iss(line.substr(2));
            std::string vertStr;
            while (iss >> vertStr) {
                // Parse v, v/vt, v/vt/vn, v//vn
                int vi = 0;
                std::sscanf(vertStr.c_str(), "%d", &vi);
                if (vi < 0) vi = static_cast<int>(positions.size()) + vi + 1;
                if (vi > 0) face.vIndices.push_back(static_cast<uint32_t>(vi - 1));
            }
            if (face.vIndices.size() >= 3) {
                rawFaces.push_back(std::move(face));
            }
        } else if (line.substr(0, 7) == "mtllib ") {
            std::string mtlFile = line.substr(7);
            // Trim whitespace
            while (!mtlFile.empty() && (mtlFile.back() == ' ' || mtlFile.back() == '\r'))
                mtlFile.pop_back();
            std::string mtlPath = objDir + "/" + mtlFile;
            mtlMap = loadMTL(mtlPath);
        } else if (line.substr(0, 7) == "usemtl ") {
            std::string matName = line.substr(7);
            while (!matName.empty() && (matName.back() == ' ' || matName.back() == '\r'))
                matName.pop_back();

            auto it = matNameToId.find(matName);
            if (it != matNameToId.end()) {
                currentMatId = it->second;
            } else {
                uint32_t id = static_cast<uint32_t>(result.materialTable.size());
                matNameToId[matName] = id;
                currentMatId = id;

                // Look up in loaded MTL data.
                auto mit = mtlMap.find(matName);
                if (mit != mtlMap.end()) {
                    result.materialTable.push_back(mit->second);
                } else {
                    MTLMaterial fallback;
                    fallback.name = matName;
                    result.materialTable.push_back(fallback);
                }
            }
        }
    }

    result.rawVertices = static_cast<uint32_t>(positions.size());
    result.rawFaces    = static_cast<uint32_t>(rawFaces.size());

    if (positions.empty() || rawFaces.empty()) {
        std::cerr << "Error: OBJ file has no geometry\n";
        return result;
    }

    // ---- Compute AABB and center model at origin ----
    Vec3 bMin = positions[0], bMax = positions[0];
    for (const auto& p : positions) {
        bMin.x = std::min(bMin.x, p.x);
        bMin.y = std::min(bMin.y, p.y);
        bMin.z = std::min(bMin.z, p.z);
        bMax.x = std::max(bMax.x, p.x);
        bMax.y = std::max(bMax.y, p.y);
        bMax.z = std::max(bMax.z, p.z);
    }
    Vec3 center = (bMin + bMax) * 0.5f;
    result.bboxMin = bMin - center;
    result.bboxMax = bMax - center;
    result.centerOffset = center;

    // Translate all vertices to center at origin.
    for (auto& p : positions) {
        p = p - center;
    }

    // ---- Copy vertices into mesh ----
    result.mesh.vertices.reserve(positions.size());
    for (const auto& p : positions) {
        result.mesh.vertices.push_back(Vertex(p));
    }

    // ---- Triangulate faces ----
    // Compute median triangle area for degenerate threshold.
    std::vector<float> tempAreas;
    tempAreas.reserve(rawFaces.size());

    for (const auto& face : rawFaces) {
        const auto& vi = face.vIndices;
        uint32_t n = static_cast<uint32_t>(vi.size());

        if (n == 3) {
            // Triangle: keep as-is.
            result.mesh.indices.push_back(TriIdx(vi[0], vi[1], vi[2]));
            result.mesh.triangle_material_id.push_back(face.materialId);
            tempAreas.push_back(MathUtils::triangleArea(
                positions[vi[0]], positions[vi[1]], positions[vi[2]]));
        } else if (n == 4) {
            // Quad: split along shortest diagonal for best triangle quality.
            float d02 = (positions[vi[0]] - positions[vi[2]]).lengthSq();
            float d13 = (positions[vi[1]] - positions[vi[3]]).lengthSq();
            if (d02 <= d13) {
                result.mesh.indices.push_back(TriIdx(vi[0], vi[1], vi[2]));
                result.mesh.indices.push_back(TriIdx(vi[0], vi[2], vi[3]));
            } else {
                result.mesh.indices.push_back(TriIdx(vi[0], vi[1], vi[3]));
                result.mesh.indices.push_back(TriIdx(vi[1], vi[2], vi[3]));
            }
            result.mesh.triangle_material_id.push_back(face.materialId);
            result.mesh.triangle_material_id.push_back(face.materialId);
            result.quadsTriangulated++;
        } else {
            // N-gon: fan triangulation from vertex 0.
            for (uint32_t k = 1; k + 1 < n; ++k) {
                result.mesh.indices.push_back(TriIdx(vi[0], vi[k], vi[k + 1]));
                result.mesh.triangle_material_id.push_back(face.materialId);
            }
            result.ngonsTriangulated++;
        }
    }

    // ---- Remove degenerate triangles ----
    // Compute adaptive epsilon based on scene scale.
    Vec3 extent = result.bboxMax - result.bboxMin;
    float maxExtent = std::max({extent.x, extent.y, extent.z});
    float epsArea = degenerateEps * maxExtent * maxExtent;

    std::vector<TriIdx> cleanIndices;
    std::vector<uint32_t> cleanMats;
    cleanIndices.reserve(result.mesh.indices.size());
    cleanMats.reserve(result.mesh.triangle_material_id.size());

    for (size_t i = 0; i < result.mesh.indices.size(); ++i) {
        const auto& tri = result.mesh.indices[i];
        // Reject duplicate vertex indices.
        if (tri.i0 == tri.i1 || tri.i1 == tri.i2 || tri.i2 == tri.i0) {
            result.degeneratesRemoved++;
            continue;
        }
        Vec3 v0 = positions[tri.i0], v1 = positions[tri.i1], v2 = positions[tri.i2];
        float area = MathUtils::triangleArea(v0, v1, v2);
        if (area < epsArea) {
            result.degeneratesRemoved++;
            continue;
        }
        cleanIndices.push_back(tri);
        cleanMats.push_back(result.mesh.triangle_material_id[i]);
    }
    result.mesh.indices = std::move(cleanIndices);
    result.mesh.triangle_material_id = std::move(cleanMats);

    return result;
}

// --------------------------------------------------------------------------
// Print load summary
// --------------------------------------------------------------------------
inline void printLoadSummary(const LoadResult& r) {
    Vec3 extent = r.bboxMax - r.bboxMin;
    std::cout << "  Vertices     : " << r.rawVertices << "\n";
    std::cout << "  Raw faces    : " << r.rawFaces << "\n";
    std::cout << "  Triangles    : " << r.mesh.numTriangles() << "\n";
    std::cout << "  Materials    : " << r.materialTable.size() << "\n";
    if (r.quadsTriangulated)
        std::cout << "  Quads split  : " << r.quadsTriangulated << "\n";
    if (r.ngonsTriangulated)
        std::cout << "  N-gons split : " << r.ngonsTriangulated << "\n";
    if (r.degeneratesRemoved)
        std::cout << "  Degenerates  : " << r.degeneratesRemoved << " removed\n";
    std::cout << "  AABB extent  : ("
              << extent.x << ", " << extent.y << ", " << extent.z << ")\n";
    std::cout << "  Centered at  : ("
              << r.centerOffset.x << ", " << r.centerOffset.y << ", "
              << r.centerOffset.z << ")\n";
}

} // namespace OBJLoader
