#pragma once

#include "core/Patch.h"
#include "geometry/IndexedMesh.h"
#include "scene/Material.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>
#include <iomanip>

namespace radiosity {
namespace core {

using geometry::IndexedMesh;
using scene::Material;
using math::Vector3;

/**
 * RadiosityScene - central data structure for radiosity computation
 * 
 * Design principles:
 * 1. OptiX-compatible mesh representation (indexed triangles)
 * 2. Clear patch-to-triangle mapping
 * 3. Single source of truth for geometry
 * 4. Minimal data transformation for GPU upload
 * 
 * Data flow:
 * - Geometry authoring → patches created → triangles indexed → OptiX upload
 * - OptiX ray hit → triangle index → patch ID → radiosity update
 */
class RadiosityScene {
public:
    // Radiosity domain
    std::vector<Patch> patches;
    
    // Geometry domain (OptiX-compatible)
    IndexedMesh mesh;
    
    // Constructor
    RadiosityScene() : mesh("scene_mesh") {}
    
    /**
     * Add a patch with its geometry
     * Returns the patch ID
     */
    uint32_t addPatch(const Patch& patch) {
        uint32_t patchId = static_cast<uint32_t>(patches.size());
        
        Patch p = patch;
        p.firstTriangleIndex = static_cast<int>(mesh.triangleCount());
        p.triangleCount = 0;  // Will be incremented as triangles are added
        
        patches.push_back(p);
        return patchId;
    }
    
    /**
     * Add a quad surface as one patch
     * Automatically subdivides and creates indexed triangles
     */
    uint32_t addQuadPatch(
        const Vector3& corner0, const Vector3& corner1,
        const Vector3& corner2, const Vector3& corner3,
        const Material& material,
        int subdivisionU, int subdivisionV)
    {
        // Create patch
        Patch patch;
        
        // Compute patch properties from quad corners
        Vector3 center = (corner0 + corner1 + corner2 + corner3) * 0.25f;
        Vector3 edge1 = corner1 - corner0;
        Vector3 edge2 = corner3 - corner0;
        Vector3 normal = edge1.cross(edge2).normalized();
        
        // Approximate area (treat as two triangles)
        float area1 = 0.5f * edge1.cross(edge2).length();
        Vector3 edge3 = corner2 - corner1;
        Vector3 edge4 = corner2 - corner3;
        float area2 = 0.5f * edge3.cross(edge4).length();
        float area = area1 + area2;
        
        patch.center = center;
        patch.normal = normal;
        patch.area = area;
        patch.emission = material.emission;
        patch.reflectance = material.reflectance;
        patch.initializeRadiosity();
        
        uint32_t patchId = addPatch(patch);
        
        // Add subdivided geometry
        size_t trianglesBefore = mesh.triangleCount();
        geometry::MeshBuilder::addSubdividedQuad(
            mesh, corner0, corner1, corner2, corner3,
            subdivisionU, subdivisionV, patchId
        );
        size_t trianglesAfter = mesh.triangleCount();
        
        // Update triangle count
        patches[patchId].triangleCount = static_cast<int>(trianglesAfter - trianglesBefore);
        
        return patchId;
    }
    
    /**
     * Add a quad surface subdivided into MULTIPLE patches (for smooth gradients)
     * Each sub-quad becomes its own patch
     */
    void addQuadPatches(
        const Vector3& corner0, const Vector3& corner1,
        const Vector3& corner2, const Vector3& corner3,
        const Material& material,
        int subdivisionU, int subdivisionV)
    {
        // Subdivide the quad into a grid of smaller quads, each is one patch
        for (int v = 0; v < subdivisionV; v++) {
            for (int u = 0; u < subdivisionU; u++) {
                float u0 = float(u) / subdivisionU;
                float u1 = float(u + 1) / subdivisionU;
                float v0 = float(v) / subdivisionV;
                float v1 = float(v + 1) / subdivisionV;
                
                // Bilinear interpolation for sub-quad corners
                auto lerp = [](const Vector3& a, const Vector3& b, float t) {
                    return a * (1.0f - t) + b * t;
                };
                
                Vector3 c00 = lerp(lerp(corner0, corner1, u0), lerp(corner3, corner2, u0), v0);
                Vector3 c10 = lerp(lerp(corner0, corner1, u1), lerp(corner3, corner2, u1), v0);
                Vector3 c11 = lerp(lerp(corner0, corner1, u1), lerp(corner3, corner2, u1), v1);
                Vector3 c01 = lerp(lerp(corner0, corner1, u0), lerp(corner3, corner2, u0), v1);
                
                // Create one patch for this sub-quad (with mesh subdivision=1 for clean quads)
                addQuadPatch(c00, c10, c11, c01, material, 1, 1);
            }
        }
    }
    
    /**
     * Get patch from triangle index (OptiX ray hit → patch lookup)
     */
    const Patch& getPatchForTriangle(uint32_t triangleIndex) const {
        uint32_t patchId = mesh.getPatchId(triangleIndex);
        return patches[patchId];
    }
    
    Patch& getPatchForTriangle(uint32_t triangleIndex) {
        uint32_t patchId = mesh.getPatchId(triangleIndex);
        return patches[patchId];
    }
    
    /**
     * Get number of patches
     */
    size_t patchCount() const {
        return patches.size();
    }
    
    /**
     * Get number of triangles
     */
    size_t triangleCount() const {
        return mesh.triangleCount();
    }
    
    /**
     * Initialize all patch radiosity values
     */
    void initializeRadiosity() {
        for (auto& patch : patches) {
            patch.initializeRadiosity();
        }
    }
    
    /**
     * Get scene bounding box
     */
    void getBounds(Vector3& minBound, Vector3& maxBound) const {
        mesh.getBounds(minBound, maxBound);
    }
    
    /**
     * Find patch with maximum unshot radiosity (for progressive refinement)
     */
    uint32_t findBrightestPatch() const {
        if (patches.empty()) return 0;
        
        uint32_t brightestId = 0;
        float maxMagnitude = patches[0].unshotMagnitude();
        
        for (size_t i = 1; i < patches.size(); i++) {
            float magnitude = patches[i].unshotMagnitude();
            if (magnitude > maxMagnitude) {
                maxMagnitude = magnitude;
                brightestId = static_cast<uint32_t>(i);
            }
        }
        
        return brightestId;
    }
    
    /**
     * Check if scene has converged
     */
    bool hasConverged(float epsilon = 0.001f) const {
        for (const auto& patch : patches) {
            if (patch.unshotMagnitude() > epsilon) {
                return false;
            }
        }
        return true;
    }
    
    /**
     * Get total unshot energy in scene
     */
    float getTotalUnshotEnergy() const {
        float total = 0.0f;
        for (const auto& patch : patches) {
            total += patch.unshotMagnitude() * patch.area;
        }
        return total;
    }
    
    /**
     * Compute accumulated visibility for each patch (debugging/AO visualization)
     * Simple nested loop: sum visibility values from all other patches
     */
    void computeAccumulatedVisibility(
        const std::vector<std::vector<float>>& visibilityMatrix,
        bool useVertexSmoothing = true)
    {
        size_t patchCount = patches.size();
        std::vector<float> accumulatedVis(patchCount, 0.0f);
        
        // CRITICAL FIX: Sum visibility FROM this patch TO all others
        // visibilityMatrix[i][j] = how much patch i can see patch j
        // We want: "How much can THIS patch see?" (like ambient occlusion)
        // NOT "How much can others see this patch?" (which makes emitters bright)
        for (size_t i = 0; i < patchCount; ++i) {
            float visSum = 0.0f;
            for (size_t j = 0; j < patchCount; ++j) {
                if (i == j) continue;
                visSum += visibilityMatrix[i][j];  // FIXED: was [j][i] - WRONG!
            }
            accumulatedVis[i] = visSum;
        }
        
        // Find min/max for normalization
        float minVis = accumulatedVis[0];
        float maxVis = accumulatedVis[0];
        for (float v : accumulatedVis) {
            if (v < minVis) minVis = v;
            if (v > maxVis) maxVis = v;
        }
        
        std::cout << "  Accumulated visibility: min=" << minVis << " max=" << maxVis << "\n";
        
        // Normalize to [0,1] and assign to patch.B
        for (size_t i = 0; i < patchCount; ++i) {
            float normalized = (maxVis > 1e-6f) ? (accumulatedVis[i] / maxVis) : 0.0f;
            patches[i].B = Vector3(normalized, normalized, normalized);
        }
        
        // Apply vertex smoothing
        if (useVertexSmoothing) {
            reconstructVertexRadiosity();
        }
    }
    
    /**
     * Reconstruct per-vertex radiosity from per-patch radiosity
     * Uses area-weighted averaging as per coding plan
     * 
     * Algorithm:
     * For each vertex v:
     *   B_v = Σ(A_i * B_i) / Σ(A_i)
     * where i are patches incident to vertex v
     */
    void reconstructVertexRadiosity() {
        size_t vertexCount = mesh.vertexCount();
        
        // Initialize vertex radiosity storage
        mesh.vertexRadiosity.resize(vertexCount, Vector3(0, 0, 0));
        
        // Build vertex-to-patches mapping
        std::vector<std::vector<uint32_t>> vertexPatches(vertexCount);
        
        // Iterate through all triangles to find which patches touch each vertex
        for (size_t triIdx = 0; triIdx < mesh.triangleCount(); ++triIdx) {
            uint32_t patchId = mesh.patchIds[triIdx];
            uint32_t i0 = mesh.indices[triIdx * 3 + 0];
            uint32_t i1 = mesh.indices[triIdx * 3 + 1];
            uint32_t i2 = mesh.indices[triIdx * 3 + 2];
            
            // Add this patch to each vertex's patch list (avoid duplicates)
            auto addPatchToVertex = [&](uint32_t vertIdx) {
                auto& patchList = vertexPatches[vertIdx];
                if (std::find(patchList.begin(), patchList.end(), patchId) == patchList.end()) {
                    patchList.push_back(patchId);
                }
            };
            
            addPatchToVertex(i0);
            addPatchToVertex(i1);
            addPatchToVertex(i2);
        }
        
        // Compute area-weighted average radiosity for each vertex
        for (size_t v = 0; v < vertexCount; ++v) {
            const auto& patchList = vertexPatches[v];
            
            if (patchList.empty()) {
                // Isolated vertex (shouldn't happen in valid mesh)
                mesh.vertexRadiosity[v] = Vector3(0, 0, 0);
                continue;
            }
            
            Vector3 sum(0, 0, 0);
            float weightSum = 0.0f;
            
            // Area-weighted averaging
            for (uint32_t patchId : patchList) {
                const Patch& patch = patches[patchId];
                float weight = patch.area;
                sum = sum + (patch.B * weight);
                weightSum += weight;
            }
            
            // Compute weighted average
            if (weightSum > 1e-6f) {
                mesh.vertexRadiosity[v] = sum / weightSum;
            } else {
                mesh.vertexRadiosity[v] = Vector3(0, 0, 0);
            }
        }
    }
    
    // DEBUG: Print scene statistics
    void printStats() const {
        std::cout << "\n=== RADIOSITY SCENE STATISTICS ===\n";
        std::cout << "Patches: " << patchCount() << "\n";
        std::cout << "Triangles: " << triangleCount() << "\n";
        std::cout << "Vertices: " << mesh.vertexCount() << "\n";
        
        Vector3 minB, maxB;
        getBounds(minB, maxB);
        std::cout << "Bounds: " << minB << " to " << maxB << "\n";
        
        // Count emissive patches
        int emissiveCount = 0;
        for (const auto& patch : patches) {
            if (patch.isEmissive()) emissiveCount++;
        }
        std::cout << "Light sources: " << emissiveCount << "\n";
        
        std::cout << "\nMemory usage:\n";
        size_t patchMem = patches.size() * sizeof(Patch);
        size_t meshMem = mesh.getVertexDataSize() + mesh.getIndexDataSize();
        std::cout << "  Patches: " << patchMem / 1024.0f << " KB\n";
        std::cout << "  Mesh: " << meshMem / 1024.0f << " KB\n";
        std::cout << "  Total: " << (patchMem + meshMem) / 1024.0f << " KB\n";
    }
};

} // namespace core
} // namespace radiosity
