#pragma once

#include <vector>
#include "../mesh/MeshData.h"

namespace AOExport {

// Bake per-triangle scalar values to vertex colors using area weighting
inline void bakeScalarToVertexColors(const Mesh& mesh,
                                      const PatchSoA& patches,
                                      const std::vector<float>& triScalars,
                                      std::vector<float>& outR,
                                      std::vector<float>& outG,
                                      std::vector<float>& outB) {
    uint32_t numVertices = static_cast<uint32_t>(mesh.vertices.size());
    uint32_t numTriangles = static_cast<uint32_t>(mesh.indices.size());
    
    outR.assign(numVertices, 0.0f);
    outG.assign(numVertices, 0.0f);
    outB.assign(numVertices, 0.0f);
    
    std::vector<float> vertexWeights(numVertices, 0.0f);
    
    // Area-weighted accumulation
    for (uint32_t t = 0; t < numTriangles; ++t) {
        const TriIdx& tri = mesh.indices[t];
        float area = patches.area[t];
        float value = triScalars[t];
        
        // Grayscale: use same value for RGB
        outR[tri.i0] += value * area;
        outG[tri.i0] += value * area;
        outB[tri.i0] += value * area;
        vertexWeights[tri.i0] += area;
        
        outR[tri.i1] += value * area;
        outG[tri.i1] += value * area;
        outB[tri.i1] += value * area;
        vertexWeights[tri.i1] += area;
        
        outR[tri.i2] += value * area;
        outG[tri.i2] += value * area;
        outB[tri.i2] += value * area;
        vertexWeights[tri.i2] += area;
    }
    
    // Normalize by total area
    for (uint32_t v = 0; v < numVertices; ++v) {
        if (vertexWeights[v] > 0.0f) {
            float invWeight = 1.0f / vertexWeights[v];
            outR[v] *= invWeight;
            outG[v] *= invWeight;
            outB[v] *= invWeight;
        }
    }
}

// Visualize visibility scores as grayscale AO (flat shaded)
inline void exportVisibilityAsAO(const Mesh& mesh,
                                  const PatchSoA& patches,
                                  const std::vector<float>& visScores,
                                  const std::string& outputPath) {
    // Export OBJ with vertex duplication for flat shading
    std::ofstream obj(outputPath);
    if (!obj) {
        std::cerr << "Failed to open output file: " << outputPath << "\n";
        return;
    }
    
    const uint32_t triangleCount = static_cast<uint32_t>(mesh.indices.size());
    
    obj << "# Phase 2 - Visibility/AO Visualization (Flat Shaded)\n";
    obj << "# " << (triangleCount * 3) << " vertices (duplicated), " 
        << triangleCount << " triangles\n\n";
    
    // Write duplicated vertices - each triangle gets its own 3 vertices
    for (uint32_t t = 0; t < triangleCount; ++t) {
        const TriIdx& tri = mesh.indices[t];
        const Vertex& v0 = mesh.vertices[tri.i0];
        const Vertex& v1 = mesh.vertices[tri.i1];
        const Vertex& v2 = mesh.vertices[tri.i2];
        
        // All 3 vertices of this triangle get the same color (grayscale AO)
        float aoValue = visScores[t];
        
        obj << "v " << v0.x << " " << v0.y << " " << v0.z
            << " " << aoValue << " " << aoValue << " " << aoValue << "\n";
        obj << "v " << v1.x << " " << v1.y << " " << v1.z
            << " " << aoValue << " " << aoValue << " " << aoValue << "\n";
        obj << "v " << v2.x << " " << v2.y << " " << v2.z
            << " " << aoValue << " " << aoValue << " " << aoValue << "\n";
    }
    
    obj << "\n";
    
    // Write faces with new indices (3 consecutive vertices per triangle)
    for (uint32_t t = 0; t < triangleCount; ++t) {
        uint32_t baseIdx = t * 3 + 1; // OBJ indices are 1-based
        obj << "f " << baseIdx << " " << (baseIdx + 1) << " " << (baseIdx + 2) << "\n";
    }
    
    std::cout << "Exported visibility/AO: " << outputPath 
              << " (" << (triangleCount * 3) << " vertices, flat shaded)\n";
}

// Bake per-triangle normal vectors to vertex colors using area weighting
inline void bakeNormalsToVertexColors(const Mesh& mesh,
                                       const PatchSoA& patches,
                                       std::vector<float>& outR,
                                       std::vector<float>& outG,
                                       std::vector<float>& outB) {
    uint32_t numVertices = static_cast<uint32_t>(mesh.vertices.size());
    uint32_t numTriangles = static_cast<uint32_t>(mesh.indices.size());
    
    outR.assign(numVertices, 0.0f);
    outG.assign(numVertices, 0.0f);
    outB.assign(numVertices, 0.0f);
    
    std::vector<float> vertexWeights(numVertices, 0.0f);
    
    // Area-weighted accumulation of normals
    for (uint32_t t = 0; t < numTriangles; ++t) {
        const TriIdx& tri = mesh.indices[t];
        float area = patches.area[t];
        
        // Get triangle normal from patches (already normalized)
        float nx = patches.nx[t];
        float ny = patches.ny[t];
        float nz = patches.nz[t];
        
        // Accumulate weighted normals
        outR[tri.i0] += nx * area;
        outG[tri.i0] += ny * area;
        outB[tri.i0] += nz * area;
        vertexWeights[tri.i0] += area;
        
        outR[tri.i1] += nx * area;
        outG[tri.i1] += ny * area;
        outB[tri.i1] += nz * area;
        vertexWeights[tri.i1] += area;
        
        outR[tri.i2] += nx * area;
        outG[tri.i2] += ny * area;
        outB[tri.i2] += nz * area;
        vertexWeights[tri.i2] += area;
    }
    
    // Normalize by total area
    for (uint32_t v = 0; v < numVertices; ++v) {
        if (vertexWeights[v] > 0.0f) {
            float invWeight = 1.0f / vertexWeights[v];
            outR[v] *= invWeight;
            outG[v] *= invWeight;
            outB[v] *= invWeight;
            
            // Normalize the resulting vector
            float len = std::sqrt(outR[v] * outR[v] + outG[v] * outG[v] + outB[v] * outB[v]);
            if (len > 1e-8f) {
                outR[v] /= len;
                outG[v] /= len;
                outB[v] /= len;
            }
            
            // Map from [-1, 1] to [0, 1] for color display
            outR[v] = outR[v] * 0.5f + 0.5f;
            outG[v] = outG[v] * 0.5f + 0.5f;
            outB[v] = outB[v] * 0.5f + 0.5f;
        }
    }
}

// Export normals as RGB vertex colors
inline void exportNormalsAsColors(const Mesh& mesh,
                                   const PatchSoA& patches,
                                   const std::string& outputPath) {
    std::vector<float> colorR, colorG, colorB;
    bakeNormalsToVertexColors(mesh, patches, colorR, colorG, colorB);
    
    // Export OBJ with vertex colors
    std::ofstream obj(outputPath);
    if (!obj) {
        std::cerr << "Failed to open output file: " << outputPath << "\n";
        return;
    }
    
    // Write header
    obj << "# Phase 2 - Normal Visualization\n";
    obj << "# " << mesh.vertices.size() << " vertices, " 
        << mesh.indices.size() << " triangles\n\n";
    
    // Write vertices with colors
    for (size_t i = 0; i < mesh.vertices.size(); ++i) {
        const Vertex& v = mesh.vertices[i];
        obj << "v " << v.x << " " << v.y << " " << v.z
            << " " << colorR[i] << " " << colorG[i] << " " << colorB[i] << "\n";
    }
    
    obj << "\n";
    
    // Write faces (1-indexed)
    for (const auto& tri : mesh.indices) {
        obj << "f " << (tri.i0 + 1) << " " 
            << (tri.i1 + 1) << " " 
            << (tri.i2 + 1) << "\n";
    }
    
    std::cout << "Exported normals visualization: " << outputPath 
              << " (" << mesh.vertices.size() << " vertices, "
              << mesh.indices.size() << " triangles)\n";
}

} // namespace AOExport
