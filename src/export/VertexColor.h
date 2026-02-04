#pragma once
#include "../mesh/MeshData.h"
#include "../math/MathUtils.h"
#include <vector>

namespace VertexColor {

// Bake per-triangle colors to vertex colors using area-weighted averaging
inline std::vector<Vec3> bakeTriangleColorsToVertices(
    const Mesh& mesh,
    const std::vector<Vec3>& triangleColors) 
{
    size_t numVerts = mesh.numVertices();
    std::vector<Vec3> vertexColors(numVerts, Vec3(0, 0, 0));
    std::vector<float> vertexWeights(numVerts, 0.0f);
    
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();
        
        float area = MathUtils::triangleArea(v0, v1, v2);
        const Vec3& color = triangleColors[i];
        
        // Accumulate area-weighted colors
        vertexColors[tri.i0] += color * area;
        vertexColors[tri.i1] += color * area;
        vertexColors[tri.i2] += color * area;
        
        vertexWeights[tri.i0] += area;
        vertexWeights[tri.i1] += area;
        vertexWeights[tri.i2] += area;
    }
    
    // Normalize by total weight
    for (size_t i = 0; i < numVerts; ++i) {
        if (vertexWeights[i] > 1e-8f) {
            vertexColors[i] = vertexColors[i] / vertexWeights[i];
        }
    }
    
    return vertexColors;
}

} // namespace VertexColor
