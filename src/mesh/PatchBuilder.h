#pragma once
#include "../mesh/MeshData.h"
#include "../scene/CornellBox.h"
#include <iostream>

namespace PatchBuilder {

// Compute per-triangle geometry and material data from the mesh.
inline void buildTriangleData(Mesh& mesh) {
    size_t N = mesh.numTriangles();
    mesh.triangle_area.resize(N);
    mesh.triangle_normal.resize(N);
    mesh.triangle_centroid.resize(N);
    mesh.triangle_reflectance.resize(N);
    mesh.triangle_emission.resize(N);

    for (size_t i = 0; i < N; ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();

        mesh.triangle_area[i]     = MathUtils::triangleArea(v0, v1, v2);
        mesh.triangle_centroid[i] = MathUtils::triangleCentroid(v0, v1, v2);
        mesh.triangle_normal[i]   = MathUtils::triangleNormal(v0, v1, v2).normalized();

        uint32_t matID = (i < mesh.triangle_material_id.size())
                         ? mesh.triangle_material_id[i] : 0u;
        CornellBox::Material mat = CornellBox::getMaterialForID(matID);
        mesh.triangle_reflectance[i] = mat.diffuse;
        mesh.triangle_emission[i]    = mat.emission;
    }
}

// Basic mesh sanity check.
inline bool validateMesh(const Mesh& mesh) {
    size_t badN = 0, badA = 0;
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const Vec3& n = mesh.triangle_normal[i];
        if (n.isZero() || n.hasNaN()) badN++;
        if (mesh.triangle_area[i] <= 0.0f) badA++;
    }
    if (badN || badA)
        std::cerr << "Mesh validation: " << badN << " bad normals, " << badA << " bad areas\n";
    return badN == 0 && badA == 0;
}

} // namespace PatchBuilder
