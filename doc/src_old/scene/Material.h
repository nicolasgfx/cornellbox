#pragma once

#include "math/Vector3.h"

namespace radiosity {
namespace scene {

using math::Vector3;

/**
 * Material properties for surfaces
 * Focused on diffuse reflection for radiosity
 */
class Material {
public:
    Vector3 reflectance;  // Diffuse reflectance (albedo) [0-1] for each RGB channel
    Vector3 emission;     // Emitted radiance (for light sources)
    std::string name;
    
    // Constructors
    Material() 
        : reflectance(0.5f, 0.5f, 0.5f), emission(0, 0, 0), name("default") {}
    
    Material(const Vector3& reflectance, const std::string& name = "unnamed")
        : reflectance(reflectance), emission(0, 0, 0), name(name) {}
    
    Material(const Vector3& reflectance, const Vector3& emission, const std::string& name = "unnamed")
        : reflectance(reflectance), emission(emission), name(name) {}
    
    // Check if this is an emissive material (light source)
    bool isEmissive() const {
        return !emission.isNearZero();
    }
    
    // Get total emitted power (for energy calculations)
    float emissivePower() const {
        return (emission.x + emission.y + emission.z) / 3.0f;
    }
    
    // Clamp reflectance to physically valid range [0, 1]
    void clampReflectance() {
        reflectance = reflectance.clamped(0.0f, 1.0f);
    }
    
    // Predefined materials for Cornell Box
    static Material white() {
        return Material(Vector3(0.73f, 0.73f, 0.73f), "white");
    }
    
    static Material red() {
        return Material(Vector3(0.63f, 0.065f, 0.05f), "red");
    }
    
    static Material green() {
        return Material(Vector3(0.14f, 0.45f, 0.091f), "green");
    }
    
    // Area light (typical Cornell Box light)
    static Material areaLight(float intensity = 15.0f) {
        return Material(
            Vector3(0.78f, 0.78f, 0.78f),  // Slight reflectance
            Vector3(intensity, intensity, intensity),  // White emission
            "area_light"
        );
    }
    
    // DEBUG: Print material properties
    void print(const char* label = nullptr) const {
        if (label) {
            std::cout << label << ":\n";
        }
        std::cout << "Material: " << name << "\n";
        std::cout << "  Reflectance: " << reflectance << "\n";
        std::cout << "  Emission: " << emission << "\n";
        std::cout << "  Emissive: " << (isEmissive() ? "yes" : "no") << "\n";
    }
};

} // namespace scene
} // namespace radiosity
