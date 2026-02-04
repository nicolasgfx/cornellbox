#pragma once

#include "math/Vector3.h"
#include <vector>
#include <cstdint>

namespace radiosity {
namespace core {

using math::Vector3;

/**
 * Patch - the fundamental radiosity element
 * Represents a surface element that emits and reflects light
 * 
 * OptiX Interoperability:
 * - Patches correspond to logical surfaces
 * - Multiple triangles can belong to one patch
 * - Triangle hits in OptiX map back to patch IDs
 */
struct Patch {
    // Geometry (precomputed, immutable)
    Vector3 center;      // Center point of patch
    Vector3 normal;      // Surface normal
    float area;          // Surface area
    
    // Material properties
    Vector3 emission;    // Emitted radiance (RGB) - for lights
    Vector3 reflectance; // Diffuse reflectance (RGB) - albedo [0,1]
    
    // Radiosity state (updated during iteration)
    Vector3 B;           // Current radiosity (total exitant)
    Vector3 B_unshot;    // Unshot radiosity (for progressive refinement)
    
    // Mesh reference
    int firstTriangleIndex;  // Start index in scene triangle list
    int triangleCount;       // Number of triangles in this patch
    
    // Constructor
    Patch() 
        : center(0,0,0), normal(0,1,0), area(0)
        , emission(0,0,0), reflectance(0.5f,0.5f,0.5f)
        , B(0,0,0), B_unshot(0,0,0)
        , firstTriangleIndex(0), triangleCount(0) {}
    
    // Check if this is a light source
    bool isEmissive() const {
        return !emission.isNearZero();
    }
    
    // Initialize radiosity from emission
    void initializeRadiosity() {
        B = emission;
        B_unshot = emission;
    }
    
    // Get unshot energy magnitude (for selecting shooting patch)
    float unshotMagnitude() const {
        return (B_unshot.x + B_unshot.y + B_unshot.z) / 3.0f;
    }
    
    // DEBUG: Print patch info
    void print(const char* label = nullptr) const {
        if (label) std::cout << label << ":\n";
        std::cout << "  Center: " << center << "\n";
        std::cout << "  Normal: " << normal << "\n";
        std::cout << "  Area: " << area << "\n";
        std::cout << "  Reflectance: " << reflectance << "\n";
        std::cout << "  Emission: " << emission << "\n";
        std::cout << "  B: " << B << "\n";
        std::cout << "  B_unshot: " << B_unshot << "\n";
    }
};

} // namespace core
} // namespace radiosity
