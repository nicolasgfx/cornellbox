#pragma once
#include "../math/Vec3.h"
#include <vector>
#include <cstdint>
#include <cstddef>

// OptiX-compatible vertex (3 floats, no padding).
struct Vertex {
    float x, y, z;

    Vertex() : x(0), y(0), z(0) {}
    Vertex(float x, float y, float z) : x(x), y(y), z(z) {}
    Vertex(const Vec3& v) : x(v.x), y(v.y), z(v.z) {}

    Vec3 toVec3() const { return Vec3(x, y, z); }
};

// Triangle index triplet.
struct TriIdx {
    uint32_t i0, i1, i2;

    TriIdx() : i0(0), i1(0), i2(0) {}
    TriIdx(uint32_t i0, uint32_t i1, uint32_t i2) : i0(i0), i1(i1), i2(i2) {}
};

// Main mesh container.
struct Mesh {
    std::vector<Vertex>   vertices;
    std::vector<TriIdx>   indices;

    // Per-triangle material identifier.
    std::vector<uint32_t> triangle_material_id;

    // Per-triangle geometry (computed by PatchBuilder).
    std::vector<float>    triangle_area;
    std::vector<Vec3>     triangle_normal;
    std::vector<Vec3>     triangle_centroid;

    // Per-triangle material properties (computed by PatchBuilder).
    std::vector<Vec3>     triangle_reflectance;   // diffuse reflectance  (ρ)
    std::vector<Vec3>     triangle_emission;      // emission for lights

    size_t numVertices()  const { return vertices.size(); }
    size_t numTriangles() const { return indices.size(); }
};
