#pragma once
#include "../mesh/MeshData.h"
#include "../scene/CornellBox.h"
#include <iostream>

namespace PatchBuilder {

// Build vertex adjacency information
inline void buildVertexAdjacency(Mesh& mesh) {
    mesh.vertex_to_tris.clear();
    mesh.vertex_to_tris.resize(mesh.numVertices());
    
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        mesh.vertex_to_tris[tri.i0].push_back(static_cast<uint32_t>(i));
        mesh.vertex_to_tris[tri.i1].push_back(static_cast<uint32_t>(i));
        mesh.vertex_to_tris[tri.i2].push_back(static_cast<uint32_t>(i));
    }
}

// Compute per-triangle geometry and radiosity data
inline void buildTriangleData(Mesh& mesh) {
    size_t N = mesh.numTriangles();
    mesh.triangle_area.resize(N);
    mesh.triangle_normal.resize(N);
    mesh.triangle_centroid.resize(N);
    mesh.triangle_reflectance.resize(N);
    mesh.triangle_emission.resize(N);
    mesh.triangle_radiosity.resize(N);
    mesh.triangle_unshot.resize(N);
    
    for (size_t i = 0; i < N; ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();
        
        // Compute geometry
        mesh.triangle_area[i] = MathUtils::triangleArea(v0, v1, v2);
        mesh.triangle_centroid[i] = MathUtils::triangleCentroid(v0, v1, v2);
        mesh.triangle_normal[i] = MathUtils::triangleNormal(v0, v1, v2).normalized();
        
        // Get material ID and material
        uint32_t matID = (i < mesh.triangle_material_id.size()) ? mesh.triangle_material_id[i] : 0;
        CornellBox::Material mat = CornellBox::getMaterialForID(matID);
        
        // Store material properties
        mesh.triangle_reflectance[i] = mat.diffuse;
        mesh.triangle_emission[i] = mat.emission;
        
        // Initialize radiosity to emission
        mesh.triangle_radiosity[i] = mat.emission;
        mesh.triangle_unshot[i] = mat.emission;
    }
}

// Validate mesh geometry
inline bool validateMesh(const Mesh& mesh) {
    size_t N = mesh.numTriangles();
    if (N == 0) return false;
    
    size_t badNormals = 0, badAreas = 0;
    for (size_t i = 0; i < N; ++i) {
        const Vec3& n = mesh.triangle_normal[i];
        if (n.isZero() || n.hasNaN()) badNormals++;
        if (mesh.triangle_area[i] <= 0.0f) badAreas++;
    }
    
    return badNormals == 0 && badAreas == 0;
}

} // namespace PatchBuilder
