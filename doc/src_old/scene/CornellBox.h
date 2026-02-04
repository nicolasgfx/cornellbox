#pragma once

#include "core/Scene.h"
#include "Material.h"
#include "math/Vector3.h"
#include "math/MathUtils.h"
#include <cmath>

namespace radiosity {
namespace scene {

using math::Vector3;
using core::RadiosityScene;

/**
 * Cornell Box scene definition
 * Standard dimensions and colors from the original Cornell Box
 * Builds directly into RadiosityScene (OptiX-ready)
 */
class CornellBox {
public:
    RadiosityScene scene;
    
    // Cornell Box dimensions (standard)
    static constexpr float WIDTH = 552.8f;   // X dimension
    static constexpr float HEIGHT = 548.8f;  // Y dimension
    static constexpr float DEPTH = 559.2f;   // Z dimension
    
    // Default constructor
    CornellBox() = default;
    
    // Build the Cornell Box with specified subdivisions
    void build(int wallSubdivision = 10, int boxSubdivision = 5) {
        // Store starting patch index
        size_t patchStart = scene.patches.size();
        
        // Cornell Box room is just a large box with front removed and inward normals
        createBox(
            Vector3(WIDTH/2, HEIGHT/2, DEPTH/2),  // Center
            WIDTH, HEIGHT, DEPTH,                  // Dimensions
            0.0f,                                  // No rotation
            wallSubdivision,
            Material::white(),                     // Default white
            true,                                  // Inward normals for room
            true                                   // Skip front face
        );
        
        // Override left and right wall materials (red/green)
        // With subdivision N: floor has N² patches, ceiling has N² patches, back has N² patches
        // Left wall starts at patchStart + 3*N², right wall at patchStart + 4*N²
        size_t patchesPerWall = wallSubdivision * wallSubdivision;
        size_t leftWallStart = patchStart + 3 * patchesPerWall;
        size_t rightWallStart = patchStart + 4 * patchesPerWall;
        
        // Set red reflectance for all left wall patches
        for (size_t i = leftWallStart; i < leftWallStart + patchesPerWall && i < scene.patches.size(); i++) {
            scene.patches[i].reflectance = Material::red().reflectance;
        }
        
        // Set green reflectance for all right wall patches  
        for (size_t i = rightWallStart; i < rightWallStart + patchesPerWall && i < scene.patches.size(); i++) {
            scene.patches[i].reflectance = Material::green().reflectance;
        }
        
        // DEBUG: Check left wall normals
        std::cout << "\n=== LEFT WALL NORMAL VERIFICATION ===\n";
        std::cout << "Left wall patches: " << leftWallStart << " to " << (leftWallStart + patchesPerWall - 1) << "\n";
        for (size_t i = leftWallStart; i < leftWallStart + std::min(size_t(5), patchesPerWall) && i < scene.patches.size(); i++) {
            const auto& p = scene.patches[i];
            std::cout << "  Patch " << i << ": center=(" << p.center.x << "," << p.center.y << "," << p.center.z
                      << ") normal=(" << p.normal.x << "," << p.normal.y << "," << p.normal.z << ")";
            if (p.normal.x < 0.9f) {
                std::cout << " ⚠ WARNING: Normal should point +X (into room)!";
            }
            std::cout << "\n";
        }
        
        // Add internal boxes (tall left, short right)
        buildBoxes(boxSubdivision);
        
        // DON'T create separate light geometry!
        // Instead, mark center ceiling patches as emissive
        markCeilingAsLight(wallSubdivision);
        
        scene.initializeRadiosity();
    }
    
    // Print scene statistics
    void printStats() const {
        scene.printStats();
    }
    
private:
    /**
     * Create a box with specified dimensions, rotation, and normal direction
     * @param center Box center position
     * @param width Box width (X dimension)
     * @param height Box height (Y dimension)  
     * @param depth Box depth (Z dimension)
     * @param rotationY Rotation around Y axis in radians
     * @param subdivision Mesh subdivision for each face
     * @param material Material for all faces
     * @param inwardNormals If true, normals point inward (for room). If false, outward (for objects)
     * @param skipFront If true, don't create front face (for open Cornell Box room)
     */
    void createBox(
        const Vector3& center,
        float width, float height, float depth,
        float rotationY,
        int subdivision,
        const Material& material,
        bool inwardNormals = false,
        bool skipFront = false)
    {
        float halfWidth = width / 2.0f;
        float halfHeight = height / 2.0f;
        float halfDepth = depth / 2.0f;
        
        // 8 corners of the box (before rotation)
        // Bottom face (Y = -halfHeight)
        Vector3 corners[8] = {
            Vector3(-halfWidth, -halfHeight, -halfDepth),  // 0: bottom-left-back
            Vector3( halfWidth, -halfHeight, -halfDepth),  // 1: bottom-right-back
            Vector3( halfWidth, -halfHeight,  halfDepth),  // 2: bottom-right-front
            Vector3(-halfWidth, -halfHeight,  halfDepth),  // 3: bottom-left-front
            // Top face (Y = +halfHeight)
            Vector3(-halfWidth,  halfHeight, -halfDepth),  // 4: top-left-back
            Vector3( halfWidth,  halfHeight, -halfDepth),  // 5: top-right-back
            Vector3( halfWidth,  halfHeight,  halfDepth),  // 6: top-right-front
            Vector3(-halfWidth,  halfHeight,  halfDepth)   // 7: top-left-front
        };
        
        // Rotate around Y axis and translate
        for (int i = 0; i < 8; i++) {
            Vector3& c = corners[i];
            float x = c.x * std::cos(rotationY) - c.z * std::sin(rotationY);
            float z = c.x * std::sin(rotationY) + c.z * std::cos(rotationY);
            c.x = x + center.x;
            c.y = c.y + center.y;
            c.z = z + center.z;
        }
        
        // Create 6 faces with proper normal orientation
        // For outward normals: face points away from box center
        // For inward normals: face points toward box center (reverse vertex order)
        
        // Bottom face (floor for room, bottom for objects)
        if (inwardNormals) {
            // Floor of room: normal points UP into room
            scene.addQuadPatches(corners[0], corners[3], corners[2], corners[1],
                              material, subdivision, subdivision);
        } else {
            // Bottom of object: normal points DOWN (skipped for boxes on floor)
            // Don't create bottom face for objects
        }
        
        // Top face (ceiling for room, top for objects)
        if (inwardNormals) {
            // Ceiling of room: normal points DOWN into room
            scene.addQuadPatches(corners[4], corners[5], corners[6], corners[7],
                              material, subdivision, subdivision);
        } else {
            // Top of object: normal points UP
            scene.addQuadPatches(corners[4], corners[7], corners[6], corners[5],
                              material, subdivision, subdivision);
        }
        
        // Back face (Z = -halfDepth)
        if (inwardNormals) {
            // Back wall of room: normal points FORWARD into room (+Z)
            scene.addQuadPatches(corners[0], corners[1], corners[5], corners[4],
                              material, subdivision, subdivision);
        } else {
            // Back of object: normal points BACKWARD (-Z)
            scene.addQuadPatches(corners[0], corners[4], corners[5], corners[1],
                              material, subdivision, subdivision);
        }
        
        // Left face (X = -halfWidth)
        if (inwardNormals) {
            // Left wall of room: normal points RIGHT into room (+X)
            scene.addQuadPatches(corners[0], corners[4], corners[7], corners[3],
                              material, subdivision, subdivision);
        } else {
            // Left side of object: normal points LEFT (-X)
            scene.addQuadPatches(corners[0], corners[3], corners[7], corners[4],
                              material, subdivision, subdivision);
        }
        
        // Right face (X = +halfWidth)
        if (inwardNormals) {
            // Right wall of room: normal points LEFT into room (-X)
            scene.addQuadPatches(corners[1], corners[2], corners[6], corners[5],
                              material, subdivision, subdivision);
        } else {
            // Right side of object: normal points RIGHT (+X)
            scene.addQuadPatches(corners[1], corners[5], corners[6], corners[2],
                              material, subdivision, subdivision);
        }
        
        // Front face (Z = +halfDepth) - skip for open Cornell Box room
        if (!skipFront) {
            if (inwardNormals) {
                // Front wall (if it existed): normal points BACKWARD into room (-Z)
                scene.addQuadPatches(corners[2], corners[3], corners[7], corners[6],
                                  material, subdivision, subdivision);
            } else {
                // Front of object: normal points FORWARD (+Z)
                scene.addQuadPatches(corners[2], corners[6], corners[7], corners[3],
                                  material, subdivision, subdivision);
            }
        }
    }
    
    void buildBoxes(int subdivision) {
        // Short box: front-right position (original Cornell Box layout)
        {
            float boxHeight = 165.0f;
            Vector3 center(368, boxHeight/2, 351);  // Front-right (larger Z)
            float size = 165.0f;
            float angle = 18.0f * math::DEG_TO_RAD;
            
            createBox(center, size, boxHeight, size, angle, subdivision,
                     Material::white(), false, false);  // Outward normals
        }
        
        // Tall box: back-left position (original Cornell Box layout)
        {
            float boxHeight = 330.0f;
            Vector3 center(186, boxHeight/2, 169);  // Back-left (smaller Z)
            float size = 165.0f;
            float angle = -18.0f * math::DEG_TO_RAD;
            
            createBox(center, size, boxHeight, size, angle, subdivision,
                     Material::white(), false, false);  // Outward normals
        }
    }
    
    void markCeilingAsLight(int wallSubdivision) {
        // Mark center ceiling patches as emissive (no additional geometry!)
        // Ceiling patches start after floor patches
        size_t patchesPerWall = wallSubdivision * wallSubdivision;
        size_t ceilingStart = patchesPerWall;  // After floor
        size_t ceilingEnd = ceilingStart + patchesPerWall;
        
        // Mark center ceiling patches as emissive
        int centerMin = wallSubdivision / 2 - 1;
        int centerMax = wallSubdivision / 2 + 1;
        
        for (int v = 0; v < wallSubdivision; v++) {
            for (int u = 0; u < wallSubdivision; u++) {
                size_t patchId = ceilingStart + v * wallSubdivision + u;
                
                // Make center patches emissive
                if (u >= centerMin && u < centerMax && v >= centerMin && v < centerMax) {
                    if (patchId < scene.patches.size()) {
                        scene.patches[patchId].emission = Material::areaLight().emission;
                    }
                }
            }
        }
    }
};

}  // namespace scene
}  // namespace radiosity
