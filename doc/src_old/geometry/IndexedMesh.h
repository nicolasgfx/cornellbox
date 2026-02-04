#pragma once

#include "math/Vector3.h"
#include <vector>
#include <cstdint>
#include <iostream>

namespace radiosity {
namespace geometry {

using math::Vector3;

/**
 * OptiX-compatible indexed triangle mesh
 * 
 * Structure matches OptiX requirements:
 * - Shared vertex buffer (no duplication)
 * - Index buffer (triangle indices)
 * - Per-triangle patch mapping
 * 
 * This allows direct upload to GPU without transformation
 */
class IndexedMesh {
public:
    // Vertex data (shared between triangles)
    std::vector<Vector3> vertices;
    
    // Per-vertex radiosity (reconstructed from patches for smooth interpolation)
    std::vector<Vector3> vertexRadiosity;
    
    // Triangle indices (each group of 3 is one triangle)
    // OptiX expects uint32_t indices
    std::vector<uint32_t> indices;
    
    // Per-triangle patch ID (for radiosity mapping)
    // indices[i*3+0,1,2] form triangle i, which belongs to patchIds[i]
    std::vector<uint32_t> patchIds;
    
    std::string name;
    
    // Constructor
    IndexedMesh() : name("unnamed") {}
    explicit IndexedMesh(const std::string& name) : name(name) {}
    
    // Add a vertex, return its index
    uint32_t addVertex(const Vector3& v) {
        vertices.push_back(v);
        return static_cast<uint32_t>(vertices.size() - 1);
    }
    
    // Add a triangle by vertex indices
    void addTriangle(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t patchId) {
        indices.push_back(i0);
        indices.push_back(i1);
        indices.push_back(i2);
        patchIds.push_back(patchId);
    }
    
    // Add a quad (as two triangles) by vertex indices
    void addQuad(uint32_t i0, uint32_t i1, uint32_t i2, uint32_t i3, uint32_t patchId) {
        // Triangle 1: i0, i1, i2
        addTriangle(i0, i1, i2, patchId);
        // Triangle 2: i0, i2, i3
        addTriangle(i0, i2, i3, patchId);
    }
    
    // Get number of triangles
    size_t triangleCount() const {
        return indices.size() / 3;
    }
    
    // Get number of vertices
    size_t vertexCount() const {
        return vertices.size();
    }
    
    // Get triangle vertices by triangle index
    void getTriangleVertices(size_t triIndex, Vector3& v0, Vector3& v1, Vector3& v2) const {
        uint32_t i0 = indices[triIndex * 3 + 0];
        uint32_t i1 = indices[triIndex * 3 + 1];
        uint32_t i2 = indices[triIndex * 3 + 2];
        v0 = vertices[i0];
        v1 = vertices[i1];
        v2 = vertices[i2];
    }
    
    // Get triangle normal
    Vector3 getTriangleNormal(size_t triIndex) const {
        Vector3 v0, v1, v2;
        getTriangleVertices(triIndex, v0, v1, v2);
        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        return edge1.cross(edge2).normalized();
    }
    
    // Get triangle area
    float getTriangleArea(size_t triIndex) const {
        Vector3 v0, v1, v2;
        getTriangleVertices(triIndex, v0, v1, v2);
        Vector3 edge1 = v1 - v0;
        Vector3 edge2 = v2 - v0;
        return 0.5f * edge1.cross(edge2).length();
    }
    
    // Get triangle centroid
    Vector3 getTriangleCentroid(size_t triIndex) const {
        Vector3 v0, v1, v2;
        getTriangleVertices(triIndex, v0, v1, v2);
        return (v0 + v1 + v2) / 3.0f;
    }
    
    // Get patch ID for a triangle
    uint32_t getPatchId(size_t triIndex) const {
        return patchIds[triIndex];
    }
    
    // Get bounding box
    void getBounds(Vector3& minBound, Vector3& maxBound) const {
        if (vertices.empty()) {
            minBound = Vector3::zero();
            maxBound = Vector3::zero();
            return;
        }
        
        minBound = vertices[0];
        maxBound = vertices[0];
        
        for (size_t i = 1; i < vertices.size(); i++) {
            minBound = math::min(minBound, vertices[i]);
            maxBound = math::max(maxBound, vertices[i]);
        }
    }
    
    // Clear all data
    void clear() {
        vertices.clear();
        indices.clear();
        patchIds.clear();
    }
    
    // Reserve space (optimization)
    void reserve(size_t vertexCount, size_t triangleCount) {
        vertices.reserve(vertexCount);
        indices.reserve(triangleCount * 3);
        patchIds.reserve(triangleCount);
    }
    
    // OptiX-specific: Get raw pointer to vertices (for device upload)
    const float* getVertexDataPtr() const {
        return reinterpret_cast<const float*>(vertices.data());
    }
    
    // OptiX-specific: Get raw pointer to indices (for device upload)
    const uint32_t* getIndexDataPtr() const {
        return indices.data();
    }
    
    // OptiX-specific: Get vertex data size in bytes
    size_t getVertexDataSize() const {
        return vertices.size() * sizeof(Vector3);
    }
    
    // OptiX-specific: Get index data size in bytes
    size_t getIndexDataSize() const {
        return indices.size() * sizeof(uint32_t);
    }
    
    // DEBUG: Print mesh statistics
    void printStats() const {
        std::cout << "IndexedMesh: " << name << "\n";
        std::cout << "  Vertices: " << vertexCount() << "\n";
        std::cout << "  Triangles: " << triangleCount() << "\n";
        std::cout << "  Indices: " << indices.size() << "\n";
        
        Vector3 minB, maxB;
        getBounds(minB, maxB);
        std::cout << "  Bounds: " << minB << " to " << maxB << "\n";
        
        // Memory usage
        size_t vertexMem = vertices.size() * sizeof(Vector3);
        size_t indexMem = indices.size() * sizeof(uint32_t);
        size_t patchMem = patchIds.size() * sizeof(uint32_t);
        std::cout << "  Memory: " << (vertexMem + indexMem + patchMem) / 1024.0f << " KB\n";
    }
};

/**
 * Helper: Build a subdivided quad as indexed mesh
 */
class MeshBuilder {
public:
    static void addSubdividedQuad(
        IndexedMesh& mesh,
        const Vector3& corner0, const Vector3& corner1,
        const Vector3& corner2, const Vector3& corner3,
        int subdivisionsU, int subdivisionsV,
        uint32_t patchId)
    {
        // Generate grid of vertices
        uint32_t firstVertex = static_cast<uint32_t>(mesh.vertices.size());
        
        for (int j = 0; j <= subdivisionsV; j++) {
            for (int i = 0; i <= subdivisionsU; i++) {
                float u = float(i) / float(subdivisionsU);
                float v = float(j) / float(subdivisionsV);
                
                // Bilinear interpolation
                Vector3 bottom = math::lerp(corner0, corner1, u);
                Vector3 top = math::lerp(corner3, corner2, u);
                Vector3 point = math::lerp(bottom, top, v);
                
                mesh.addVertex(point);
            }
        }
        
        // Generate quad indices (as two triangles each)
        int vertsPerRow = subdivisionsU + 1;
        for (int j = 0; j < subdivisionsV; j++) {
            for (int i = 0; i < subdivisionsU; i++) {
                uint32_t i00 = firstVertex + j * vertsPerRow + i;
                uint32_t i10 = i00 + 1;
                uint32_t i01 = i00 + vertsPerRow;
                uint32_t i11 = i01 + 1;
                
                mesh.addQuad(i00, i10, i11, i01, patchId);
            }
        }
    }
};

} // namespace geometry
} // namespace radiosity
