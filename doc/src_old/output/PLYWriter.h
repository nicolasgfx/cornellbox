#pragma once

#include "geometry/IndexedMesh.h"
#include "core/Patch.h"
#include "scene/CornellBox.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include <vector>
#include <cmath>

namespace radiosity {
namespace output {

using geometry::IndexedMesh;
using core::Patch;
using math::Vector3;

/**
 * PLY file writer with vertex colors
 * Exports radiosity solution with per-vertex color interpolation
 */
class PLYWriter {
public:
    // Debug flag: set to true to disable tone mapping
    inline static bool DEBUG_NO_TONE_MAPPING = false;
    
    /**
     * Simple Reinhard tone mapping
     */
    static Vector3 toneMap(const Vector3& color, float exposure = 1.0f) {
        Vector3 mapped = color * exposure;
        return Vector3(
            mapped.x / (1.0f + mapped.x),
            mapped.y / (1.0f + mapped.y),
            mapped.z / (1.0f + mapped.z)
        );
    }
    
    /**
     * Convert float [0,1] to byte [0,255]
     */
    static int floatToByte(float f) {
        return static_cast<int>(std::max(0.0f, std::min(1.0f, f)) * 255.0f);
    }
    
    /**
     * Compute vertex color by interpolating from patches that use this vertex
     */
    static Vector3 computeVertexColor(
        size_t vertexIndex,
        const IndexedMesh& mesh,
        const std::vector<Patch>& patches,
        float exposure)
    {
        Vector3 vertexPos = mesh.vertices[vertexIndex];
        Vector3 sumColor(0, 0, 0);
        float sumWeight = 0.0f;
        
        // Find all triangles using this vertex
        for (size_t triIdx = 0; triIdx < mesh.triangleCount(); triIdx++) {
            uint32_t i0 = mesh.indices[triIdx * 3 + 0];
            uint32_t i1 = mesh.indices[triIdx * 3 + 1];
            uint32_t i2 = mesh.indices[triIdx * 3 + 2];
            
            if (i0 == vertexIndex || i1 == vertexIndex || i2 == vertexIndex) {
                // This triangle uses our vertex
                uint32_t patchId = mesh.getPatchId(triIdx);
                const Patch& patch = patches[patchId];
                
                // Weight by inverse distance from vertex to patch center
                float dist = (vertexPos - patch.center).length();
                float weight = 1.0f / (1.0f + dist);
                
                sumColor = sumColor + patch.B * weight;
                sumWeight += weight;
            }
        }
        
        if (sumWeight > 0.0f) {
            sumColor = sumColor * (1.0f / sumWeight);
        }
        
        // Apply tone mapping unless disabled
        if (DEBUG_NO_TONE_MAPPING) {
            // Return raw radiosity * exposure (may exceed 1.0)
            return sumColor * exposure;
        } else {
            return toneMap(sumColor, exposure);
        }
    }
    
    /**
     * Write Cornell Box radiosity solution to PLY with vertex colors
     */
    static bool writeCornellBoxRadiosity(
        const std::string& filename,
        const scene::CornellBox& box,
        float exposure = 1.0f)
    {
        const IndexedMesh& mesh = box.scene.mesh;
        const auto& patches = box.scene.patches;
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for writing: " << filename << "\n";
            return false;
        }
        
        // Compute color for each vertex
        std::cout << "  Computing vertex colors from " << patches.size() << " patches...\n";
        std::cout << "  Tone mapping: " << (DEBUG_NO_TONE_MAPPING ? "DISABLED" : "ENABLED") << "\n";
        
        std::vector<Vector3> vertexColors(mesh.vertexCount());
        std::vector<bool> isEmittingVertex(mesh.vertexCount(), false);
        Vector3 minColor(1e6, 1e6, 1e6);
        Vector3 maxColor(-1e6, -1e6, -1e6);
        
        // Mark vertices that belong to emitting patches
        for (uint32_t triIdx = 0; triIdx < mesh.triangleCount(); triIdx++) {
            uint32_t patchId = mesh.getPatchId(triIdx);
            if (patches[patchId].emission.length() > 0.1f) {
                uint32_t i0 = mesh.indices[triIdx * 3 + 0];
                uint32_t i1 = mesh.indices[triIdx * 3 + 1];
                uint32_t i2 = mesh.indices[triIdx * 3 + 2];
                isEmittingVertex[i0] = true;
                isEmittingVertex[i1] = true;
                isEmittingVertex[i2] = true;
            }
        }
        
        // Compute colors: emitting vertices = white (1,1,1), others = computed radiosity
        for (size_t i = 0; i < mesh.vertexCount(); i++) {
            if (isEmittingVertex[i]) {
                vertexColors[i] = Vector3(1.0f, 1.0f, 1.0f);  // Light sources are pure white
            } else {
                vertexColors[i] = computeVertexColor(i, mesh, patches, exposure);
            }
            
            // Track min/max for diagnostics
            if (!isEmittingVertex[i]) {  // Only track indirectly lit surfaces
                minColor.x = std::min(minColor.x, vertexColors[i].x);
                minColor.y = std::min(minColor.y, vertexColors[i].y);
                minColor.z = std::min(minColor.z, vertexColors[i].z);
                maxColor.x = std::max(maxColor.x, vertexColors[i].x);
                maxColor.y = std::max(maxColor.y, vertexColors[i].y);
                maxColor.z = std::max(maxColor.z, vertexColors[i].z);
            }
        }
        
        std::cout << "  Color range (indirect lighting only):\n";
        std::cout << "    R: [" << minColor.x << ", " << maxColor.x << "]\n";
        std::cout << "    G: [" << minColor.y << ", " << maxColor.y << "]\n";
        std::cout << "    B: [" << minColor.z << ", " << maxColor.z << "]\n";
        
        // Auto-normalize based on brightest indirectly lit surface
        // Light sources are already pure white (1.0) and will stay white
        float maxIndirect = std::max(std::max(maxColor.x, maxColor.y), maxColor.z);
        
        float normalizationFactor = 1.0f;
        if (maxIndirect > 0.0f) {
            normalizationFactor = 1.0f / maxIndirect;  // Scale so max indirect becomes 1.0
            std::cout << "  Normalization factor: " << normalizationFactor << " (max indirect=" << maxIndirect << ")\n";
        }
        
        // Apply normalization only to indirectly lit surfaces (emitters stay white)
        if (normalizationFactor != 1.0f) {
            for (size_t i = 0; i < vertexColors.size(); i++) {
                if (!isEmittingVertex[i]) {
                    vertexColors[i] = vertexColors[i] * normalizationFactor;
                }
            }
            std::cout << "  ✓ Normalized indirect lighting (light sources = white)\n";
        }
        
        // Write PLY header
        file << "ply\n";
        file << "format ascii 1.0\n";
        file << "comment Radiosity Renderer - Cornell Box Solution\n";
        file << "comment Patches: " << patches.size() << "\n";
        file << "comment Exposure: " << exposure << "\n";
        file << "element vertex " << mesh.vertexCount() << "\n";
        file << "property float x\n";
        file << "property float y\n";
        file << "property float z\n";
        file << "property uchar red\n";
        file << "property uchar green\n";
        file << "property uchar blue\n";
        file << "element face " << mesh.triangleCount() << "\n";
        file << "property list uchar int vertex_indices\n";
        file << "end_header\n";
        
        // Write vertices with colors
        for (size_t i = 0; i < mesh.vertexCount(); i++) {
            const Vector3& v = mesh.vertices[i];
            const Vector3& color = vertexColors[i];
            
            file << v.x << " " << v.y << " " << v.z << " "
                 << floatToByte(color.x) << " "
                 << floatToByte(color.y) << " "
                 << floatToByte(color.z) << "\n";
        }
        
        // Write faces (triangles)
        for (size_t i = 0; i < mesh.triangleCount(); i++) {
            uint32_t i0 = mesh.indices[i * 3 + 0];
            uint32_t i1 = mesh.indices[i * 3 + 1];
            uint32_t i2 = mesh.indices[i * 3 + 2];
            
            file << "3 " << i0 << " " << i1 << " " << i2 << "\n";
        }
        
        file.close();
        
        std::cout << "✓ Exported Cornell Box Radiosity to " << filename << "\n";
        std::cout << "  Vertices: " << mesh.vertexCount() << " (with interpolated vertex colors)\n";
        std::cout << "  Triangles: " << mesh.triangleCount() << "\n";
        std::cout << "  Patches: " << patches.size() << "\n";
        std::cout << "  Exposure: " << exposure << "\n";
        
        return true;
    }
};

} // namespace output
} // namespace radiosity
