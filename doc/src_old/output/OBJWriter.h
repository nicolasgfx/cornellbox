#pragma once

#include "geometry/IndexedMesh.h"
#include "core/Patch.h"
#include "scene/CornellBox.h"
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>

namespace radiosity {
namespace output {

using geometry::IndexedMesh;
using core::Patch;
using math::Vector3;

/**
 * OBJ file writer for geometry visualization
 * Exports indexed meshes to Wavefront OBJ format
 */
class OBJWriter {
public:
    // Debug flag: set to true to disable tone mapping
    inline static bool DEBUG_NO_TONE_MAPPING = true;  // FORCE DISABLED for color debugging
    
    /**
     * Simple Reinhard tone mapping for HDR radiosity values
     * Maps [0, infinity) to [0, 1]
     */
    static Vector3 toneMap(const Vector3& color, float exposure = 1.0f) {
        Vector3 mapped = color * exposure;
        // Reinhard: L_out = L_in / (1 + L_in)
        return Vector3(
            mapped.x / (1.0f + mapped.x),
            mapped.y / (1.0f + mapped.y),
            mapped.z / (1.0f + mapped.z)
        );
    }
    
    /**
     * Write Cornell Box to OBJ with computed radiosity values
     * Uses vertex colors to show the lighting solution
     */
    static bool writeCornellBoxRadiosity(const std::string& filename, const scene::CornellBox& box, float exposure = 1.0f) {
        const IndexedMesh& mesh = box.scene.mesh;
        const auto& patches = box.scene.patches;
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for writing: " << filename << "\n";
            return false;
        }
        
        std::string mtlFilename = filename.substr(0, filename.find_last_of('.')) + ".mtl";
        std::string mtlBasename = mtlFilename.substr(mtlFilename.find_last_of('/') + 1);
        
        // Header
        file << "# Radiosity Renderer - Cornell Box with Radiosity Solution\n";
        file << "# Total triangles: " << mesh.triangleCount() << "\n";
        file << "# Total patches: " << patches.size() << "\n";
        file << "# Exposure: " << exposure << "\n";
        file << "mtllib " << mtlBasename << "\n\n";
        
        // Write all vertices
        for (const auto& v : mesh.vertices) {
            file << "v " << v.x << " " << v.y << " " << v.z << "\n";
        }
        file << "\n";
        
        // Write normals per triangle
        for (size_t i = 0; i < mesh.triangleCount(); i++) {
            Vector3 normal = mesh.getTriangleNormal(i);
            file << "vn " << normal.x << " " << normal.y << " " << normal.z << "\n";
        }
        file << "\n";
        
        // Group triangles by patch and create materials from radiosity values
        for (size_t triIdx = 0; triIdx < mesh.triangleCount(); triIdx++) {
            uint32_t patchId = mesh.getPatchId(triIdx);
            const Patch& patch = patches[patchId];
            
            // Start new group for each patch
            if (triIdx == 0 || mesh.getPatchId(triIdx - 1) != patchId) {
                file << "\no patch_" << patchId << "\n";
                file << "usemtl radiosity_" << patchId << "\n";
            }
            
            // Write face (indices are 1-based in OBJ)
            uint32_t i0 = mesh.indices[triIdx * 3 + 0] + 1;
            uint32_t i1 = mesh.indices[triIdx * 3 + 1] + 1;
            uint32_t i2 = mesh.indices[triIdx * 3 + 2] + 1;
            uint32_t n = static_cast<uint32_t>(triIdx + 1);
            
            file << "f " << i0 << "//" << n 
                 << " " << i1 << "//" << n
                 << " " << i2 << "//" << n << "\n";
        }
        
        file.close();
        
        // Write MTL file with radiosity values
        writeMTLRadiosity(mtlFilename, patches, exposure);
        
        std::cout << "✓ Exported Cornell Box Radiosity Solution to " << filename << "\n";
        std::cout << "  Patches: " << patches.size() << "\n";
        std::cout << "  Triangles: " << mesh.triangleCount() << "\n";
        std::cout << "  Vertices: " << mesh.vertexCount() << "\n";
        std::cout << "  Material file: " << mtlBasename << "\n";
        std::cout << "  Exposure: " << exposure << "\n";
        return true;
    }
    
    /**
     * Write Cornell Box to OBJ with material colors
     */
    static bool writeCornellBox(const std::string& filename, const scene::CornellBox& box) {
        const IndexedMesh& mesh = box.scene.mesh;
        const auto& patches = box.scene.patches;
        
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open file for writing: " << filename << "\n";
            return false;
        }
        
        std::string mtlFilename = filename.substr(0, filename.find_last_of('.')) + ".mtl";
        std::string mtlBasename = mtlFilename.substr(mtlFilename.find_last_of('/') + 1);
        
        // Header
        file << "# Radiosity Renderer - Cornell Box Scene\n";
        file << "# Total triangles: " << mesh.triangleCount() << "\n";
        file << "# Total patches: " << patches.size() << "\n";
        file << "mtllib " << mtlBasename << "\n\n";
        
        // Write all vertices
        for (const auto& v : mesh.vertices) {
            file << "v " << v.x << " " << v.y << " " << v.z << "\n";
        }
        file << "\n";
        
        // Write normals per triangle
        for (size_t i = 0; i < mesh.triangleCount(); i++) {
            Vector3 normal = mesh.getTriangleNormal(i);
            file << "vn " << normal.x << " " << normal.y << " " << normal.z << "\n";
        }
        file << "\n";
        
        // Group triangles by patch and write faces
        uint32_t currentPatchId = 0;
        std::string currentMaterial = "";
        
        for (size_t triIdx = 0; triIdx < mesh.triangleCount(); triIdx++) {
            uint32_t patchId = mesh.getPatchId(triIdx);
            
            // Start new object/material group when patch changes
            if (patchId != currentPatchId || triIdx == 0) {
                currentPatchId = patchId;
                const Patch& patch = patches[patchId];
                
                // Determine material name from reflectance
                std::string matName;
                if (patch.isEmissive()) {
                    matName = "light";
                } else if (patch.reflectance.x > 0.6f && patch.reflectance.y < 0.1f) {
                    matName = "red";
                } else if (patch.reflectance.y > 0.4f && patch.reflectance.x < 0.2f) {
                    matName = "green";
                } else {
                    matName = "white";
                }
                
                if (matName != currentMaterial) {
                    file << "\no patch_" << patchId << "\n";
                    file << "usemtl " << matName << "\n";
                    currentMaterial = matName;
                }
            }
            
            // Write face (indices are 1-based in OBJ)
            uint32_t i0 = mesh.indices[triIdx * 3 + 0] + 1;
            uint32_t i1 = mesh.indices[triIdx * 3 + 1] + 1;
            uint32_t i2 = mesh.indices[triIdx * 3 + 2] + 1;
            uint32_t n = static_cast<uint32_t>(triIdx + 1);
            
            file << "f " << i0 << "//" << n 
                 << " " << i1 << "//" << n
                 << " " << i2 << "//" << n << "\n";
        }
        
        file.close();
        
        // Write MTL file
        writeMTL(mtlFilename, patches);
        
        std::cout << "✓ Exported Cornell Box to " << filename << "\n";
        std::cout << "  Patches: " << patches.size() << "\n";
        std::cout << "  Triangles: " << mesh.triangleCount() << "\n";
        std::cout << "  Vertices: " << mesh.vertexCount() << "\n";
        std::cout << "  Material file: " << mtlBasename << "\n";
        return true;
    }
    
private:
    /**
     * Write MTL file with radiosity values as diffuse colors
     */
    static bool writeMTLRadiosity(const std::string& filename, const std::vector<Patch>& patches, float exposure) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open MTL file for writing: " << filename << "\n";
            return false;
        }
        
        file << "# Radiosity Renderer - Radiosity Solution Materials\n";
        file << "# Each material represents the computed radiosity B for one patch\n";
        file << "# Light sources are pure white (255,255,255)\n\n";
        
        // Auto-normalize: find brightest indirectly lit patch
        float maxIndirect = 0.0f;
        for (const auto& patch : patches) {
            if (patch.emission.length() <= 0.1f) {  // Not a light source
                Vector3 color = patch.B * exposure;
                float maxComp = std::max(std::max(color.x, color.y), color.z);
                maxIndirect = std::max(maxIndirect, maxComp);
            }
        }
        
        float normalizationFactor = 1.0f;
        if (maxIndirect > 0.0f && DEBUG_NO_TONE_MAPPING) {
            normalizationFactor = 1.0f / maxIndirect;
            std::cout << "  OBJ normalization: " << normalizationFactor << " (max indirect=" << maxIndirect << ")\n";
        }
        
        // Write one material per patch
        for (size_t i = 0; i < patches.size(); i++) {
            const Patch& patch = patches[i];
            
            Vector3 color;
            if (patch.emission.length() > 0.1f) {
                // Light sources are pure white
                color = Vector3(1.0f, 1.0f, 1.0f);
            } else {
                // Indirect lighting: tone-mapped or normalized
                color = DEBUG_NO_TONE_MAPPING ? (patch.B * exposure * normalizationFactor) : toneMap(patch.B, exposure);
            }
            
            // Clamp to [0, 1] just in case
            color.x = std::max(0.0f, std::min(1.0f, color.x));
            color.y = std::max(0.0f, std::min(1.0f, color.y));
            color.z = std::max(0.0f, std::min(1.0f, color.z));
            
            file << "newmtl radiosity_" << i << "\n";
            file << "Ka " << color.x << " " << color.y << " " << color.z << "\n";  // Ambient = radiosity
            file << "Kd " << color.x << " " << color.y << " " << color.z << "\n";  // Diffuse = radiosity
            file << "Ks 0 0 0\n";  // No specular
            file << "Ke 0 0 0\n";  // No emission (radiosity already in Ka/Kd)
            file << "Ns 0.0\n";
            file << "d 1.0\n";  // Fully opaque
            file << "illum 0\n\n";  // Color on, ambient off (uses Ka directly)
        }
        
        file.close();
        return true;
    }
    
    /**
     * Write MTL (material) file for Cornell Box
     */
    static bool writeMTL(const std::string& filename, const std::vector<Patch>& patches) {
        std::ofstream file(filename);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open MTL file for writing: " << filename << "\n";
            return false;
        }
        
        file << "# Radiosity Renderer - Cornell Box Materials\n\n";
        
        // Write standard materials (avoid duplicates)
        std::vector<std::string> materials = {"white", "red", "green", "light"};
        
        for (const auto& matName : materials) {
            Vector3 color;
            Vector3 emission(0, 0, 0);
            
            if (matName == "white") {
                color = Vector3(0.73f, 0.73f, 0.73f);
            } else if (matName == "red") {
                color = Vector3(0.63f, 0.065f, 0.05f);
            } else if (matName == "green") {
                color = Vector3(0.14f, 0.45f, 0.091f);
            } else if (matName == "light") {
                color = Vector3(0.78f, 0.78f, 0.78f);
                emission = Vector3(1.0f, 1.0f, 1.0f);
            }
            
            file << "newmtl " << matName << "\n";
            file << "Ka " << color.x << " " << color.y << " " << color.z << "\n";
            file << "Kd " << color.x << " " << color.y << " " << color.z << "\n";
            file << "Ke " << emission.x << " " << emission.y << " " << emission.z << "\n";
            file << "Ns 10.0\n";
            file << "illum 1\n\n";
        }
        
        file.close();
        return true;
    }
};

} // namespace output
} // namespace radiosity
