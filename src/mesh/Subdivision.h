#pragma once
#include "../mesh/MeshData.h"
#include "../math/MathUtils.h"

namespace Subdivision {

// Recursively subdivide a triangle until each piece has area ≤ targetArea.
inline void subdivideTriangleAdaptive(
    const Vertex& v0, const Vertex& v1, const Vertex& v2,
    uint32_t materialID, float targetArea,
    std::vector<Vertex>&   outVerts,
    std::vector<TriIdx>&   outTris,
    std::vector<uint32_t>& outMats)
{
    Vec3 p0 = v0.toVec3(), p1 = v1.toVec3(), p2 = v2.toVec3();

    if (MathUtils::triangleArea(p0, p1, p2) <= targetArea) {
        uint32_t base = static_cast<uint32_t>(outVerts.size());
        outVerts.push_back(v0);
        outVerts.push_back(v1);
        outVerts.push_back(v2);
        outTris.push_back(TriIdx(base, base + 1, base + 2));
        outMats.push_back(materialID);
        return;
    }

    // Split into 4 sub-triangles at edge midpoints.
    Vertex m01((p0 + p1) * 0.5f);
    Vertex m12((p1 + p2) * 0.5f);
    Vertex m20((p2 + p0) * 0.5f);

    subdivideTriangleAdaptive(v0,  m01, m20, materialID, targetArea, outVerts, outTris, outMats);
    subdivideTriangleAdaptive(m01, v1,  m12, materialID, targetArea, outVerts, outTris, outMats);
    subdivideTriangleAdaptive(m20, m12, v2,  materialID, targetArea, outVerts, outTris, outMats);
    subdivideTriangleAdaptive(m01, m12, m20, materialID, targetArea, outVerts, outTris, outMats);
}

// Subdivide every triangle until all have area ≤ targetArea.
inline Mesh subdivideToUniformArea(const Mesh& input, float targetArea) {
    Mesh out;
    out.vertices.reserve(input.numTriangles() * 6);
    out.indices.reserve(input.numTriangles() * 4);
    out.triangle_material_id.reserve(input.numTriangles() * 4);

    for (size_t i = 0; i < input.numTriangles(); ++i) {
        const auto& tri = input.indices[i];
        uint32_t mat = (i < input.triangle_material_id.size())
                       ? input.triangle_material_id[i] : 0u;
        subdivideTriangleAdaptive(
            input.vertices[tri.i0], input.vertices[tri.i1], input.vertices[tri.i2],
            mat, targetArea, out.vertices, out.indices, out.triangle_material_id);
    }
    return out;
}

} // namespace Subdivision
