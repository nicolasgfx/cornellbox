#pragma once
#include "../math/Vec3.h"
#include <vector>
#include <cstdint>

// OptiX-compatible vertex format
struct Vertex {
    float x, y, z;
    
    Vertex() : x(0), y(0), z(0) {}
    Vertex(float x, float y, float z) : x(x), y(y), z(z) {}
    Vertex(const Vec3& v) : x(v.x), y(v.y), z(v.z) {}
    
    Vec3 toVec3() const { return Vec3(x, y, z); }
};

// Triangle indices
struct TriIdx {
    uint32_t i0, i1, i2;
    
    TriIdx() : i0(0), i1(0), i2(0) {}
    TriIdx(uint32_t i0, uint32_t i1, uint32_t i2) : i0(i0), i1(i1), i2(i2) {}
};



// Main mesh container
struct Mesh {
    std::vector<Vertex> vertices;
    std::vector<TriIdx> indices;
    
    // Material tracking: per-triangle material identifier
    std::vector<uint32_t> triangle_material_id;
    
    // Vertex adjacency: for each vertex, list of incident triangle indices
    std::vector<std::vector<uint32_t>> vertex_to_tris;
    
    // Per-triangle geometry data
    std::vector<float> triangle_area;
    std::vector<Vec3> triangle_normal;
    std::vector<Vec3> triangle_centroid;
    
    // Per-triangle material/radiosity data
    std::vector<Vec3> triangle_reflectance;    // diffuse reflectance (rho)
    std::vector<Vec3> triangle_emission;       // emission for light sources
    std::vector<Vec3> triangle_radiosity;      // radiosity (B)
    std::vector<Vec3> triangle_unshot;         // unshot radiosity (Bu)
    
    size_t numVertices() const { return vertices.size(); }
    size_t numTriangles() const { return indices.size(); }
    
    void clear() {
        vertices.clear();
        indices.clear();
        triangle_material_id.clear();
        vertex_to_tris.clear();
        triangle_area.clear();
        triangle_normal.clear();
        triangle_centroid.clear();
        triangle_reflectance.clear();
        triangle_emission.clear();
        triangle_radiosity.clear();
        triangle_unshot.clear();
    }
};
