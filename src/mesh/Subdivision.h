#pragma once
#include "../mesh/MeshData.h"
#include "../math/MathUtils.h"
#include <algorithm>

namespace Subdivision {

// Subdivide a single triangle into 4 triangles, preserving material ID
inline void subdivideTri1to4(const Vertex& v0, const Vertex& v1, const Vertex& v2,
                              uint32_t materialID,
                              std::vector<Vertex>& outVertices,
                              std::vector<TriIdx>& outIndices,
                              std::vector<uint32_t>& outMaterialIDs) {
    // Compute midpoints
    Vec3 p0 = v0.toVec3();
    Vec3 p1 = v1.toVec3();
    Vec3 p2 = v2.toVec3();
    
    Vec3 m01 = (p0 + p1) * 0.5f;
    Vec3 m12 = (p1 + p2) * 0.5f;
    Vec3 m20 = (p2 + p0) * 0.5f;
    
    uint32_t baseIdx = static_cast<uint32_t>(outVertices.size());
    
    // Add 6 vertices (3 original + 3 midpoints)
    outVertices.push_back(v0);
    outVertices.push_back(v1);
    outVertices.push_back(v2);
    outVertices.push_back(Vertex(m01));
    outVertices.push_back(Vertex(m12));
    outVertices.push_back(Vertex(m20));
    
    // Create 4 triangles
    // Corner triangles
    outIndices.push_back(TriIdx(baseIdx + 0, baseIdx + 3, baseIdx + 5));
    outIndices.push_back(TriIdx(baseIdx + 3, baseIdx + 1, baseIdx + 4));
    outIndices.push_back(TriIdx(baseIdx + 5, baseIdx + 4, baseIdx + 2));
    // Center triangle
    outIndices.push_back(TriIdx(baseIdx + 3, baseIdx + 4, baseIdx + 5));
    
    // All 4 sub-triangles inherit the parent material ID
    outMaterialIDs.push_back(materialID);
    outMaterialIDs.push_back(materialID);
    outMaterialIDs.push_back(materialID);
    outMaterialIDs.push_back(materialID);
}

// Subdivide entire mesh (each triangle becomes 4)
inline Mesh subdivideMesh(const Mesh& input) {
    Mesh output;
    output.vertices.reserve(input.numTriangles() * 6); // Rough estimate
    output.indices.reserve(input.numTriangles() * 4);
    output.triangle_material_id.reserve(input.numTriangles() * 4);
    
    for (size_t i = 0; i < input.numTriangles(); ++i) {
        const auto& tri = input.indices[i];
        uint32_t matID = (i < input.triangle_material_id.size()) ? input.triangle_material_id[i] : 0;
        
        subdivideTri1to4(
            input.vertices[tri.i0],
            input.vertices[tri.i1],
            input.vertices[tri.i2],
            matID,
            output.vertices,
            output.indices,
            output.triangle_material_id
        );
    }
    
    return output;
}

// Apply subdivision N times
inline Mesh subdivideN(const Mesh& input, int levels) {
    if (levels <= 0) return input;
    
    Mesh result = input;
    for (int i = 0; i < levels; ++i) {
        result = subdivideMesh(result);
    }
    return result;
}

// Recursively subdivide a single triangle until target area is reached
inline void subdivideTriangleAdaptive(
    const Vertex& v0, const Vertex& v1, const Vertex& v2,
    uint32_t materialID,
    float targetArea,
    std::vector<Vertex>& outVertices,
    std::vector<TriIdx>& outIndices,
    std::vector<uint32_t>& outMaterialIDs)
{
    // Compute triangle area
    Vec3 p0 = v0.toVec3();
    Vec3 p1 = v1.toVec3();
    Vec3 p2 = v2.toVec3();
    float area = MathUtils::triangleArea(p0, p1, p2);
    
    // If small enough, add to output
    if (area <= targetArea) {
        uint32_t baseIdx = static_cast<uint32_t>(outVertices.size());
        outVertices.push_back(v0);
        outVertices.push_back(v1);
        outVertices.push_back(v2);
        outIndices.push_back(TriIdx(baseIdx + 0, baseIdx + 1, baseIdx + 2));
        outMaterialIDs.push_back(materialID);
        return;
    }
    
    // Otherwise, subdivide into 4 and recurse
    Vec3 m01 = (p0 + p1) * 0.5f;
    Vec3 m12 = (p1 + p2) * 0.5f;
    Vec3 m20 = (p2 + p0) * 0.5f;
    
    Vertex vm01(m01);
    Vertex vm12(m12);
    Vertex vm20(m20);
    
    // Recursively subdivide the 4 sub-triangles
    subdivideTriangleAdaptive(v0, vm01, vm20, materialID, targetArea, outVertices, outIndices, outMaterialIDs);
    subdivideTriangleAdaptive(vm01, v1, vm12, materialID, targetArea, outVertices, outIndices, outMaterialIDs);
    subdivideTriangleAdaptive(vm20, vm12, v2, materialID, targetArea, outVertices, outIndices, outMaterialIDs);
    subdivideTriangleAdaptive(vm01, vm12, vm20, materialID, targetArea, outVertices, outIndices, outMaterialIDs);
}

// Adaptive subdivision based on target area
inline Mesh subdivideByArea(const Mesh& input, float targetArea) {
    Mesh output;
    output.vertices.reserve(input.numTriangles() * 6); // Rough estimate
    output.indices.reserve(input.numTriangles() * 4);
    output.triangle_material_id.reserve(input.numTriangles() * 4);
    
    size_t initialTriCount = input.numTriangles();
    
    for (size_t i = 0; i < initialTriCount; ++i) {
        const auto& tri = input.indices[i];
        uint32_t matID = (i < input.triangle_material_id.size()) ? input.triangle_material_id[i] : 0;
        
        subdivideTriangleAdaptive(
            input.vertices[tri.i0],
            input.vertices[tri.i1],
            input.vertices[tri.i2],
            matID,
            targetArea,
            output.vertices,
            output.indices,
            output.triangle_material_id
        );
    }
    
    return output;
}

} // namespace Subdivision
