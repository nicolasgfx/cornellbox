#pragma once

#include "math/Vector3.h"
#include "core/Patch.h"
#include "core/Scene.h"
#include "radiosity/MonteCarloFormFactor.h"
#include "visibility/VisibilityTester.h"
#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace radiosity {

using math::Vector3;
using core::Patch;
using core::RadiosityScene;

/**
 * Progressive Radiosity Renderer
 * 
 * Implements the shooting method for radiosity:
 * 1. Find patch with most unshot radiosity
 * 2. Shoot radiosity to all visible patches
 * 3. Update radiosity values
 * 4. Repeat until convergence
 * 
 * Week 4: Core radiosity solver implementation
 */
class RadiosityRenderer {
public:
    struct Config {
        float convergenceThreshold = 0.01f;  // Stop when unshot energy < this
        int maxIterations = 100;              // Maximum iterations
        bool verbose = true;                  // Print progress
        bool debugOutput = false;             // Detailed debug info
        bool debugNoDistance = false;         // DEBUG: Disable distance attenuation
        bool debugAttenuateReflectance = false;  // DEBUG: Reduce reflectance to 25%
    };
    
    struct Statistics {
        int iterations = 0;
        float initialEnergy = 0.0f;
        float finalEnergy = 0.0f;
        float unshotEnergy = 0.0f;
        float maxUnshotPatch = 0.0f;
        bool converged = false;
    };
    
    RadiosityRenderer(const Config& config = Config()) 
        : config(config) {}
    
    /**
     * Solve radiosity for the scene
     * Returns statistics about the solution
     */
    Statistics solve(RadiosityScene& scene, visibility::VisibilityTester* visibilityTester = nullptr) {
        if (config.verbose) {
            std::cout << "\n=== RADIOSITY SOLVER ===" << std::endl;
            std::cout << "Patches: " << scene.patches.size() << std::endl;
            std::cout << "Convergence threshold: " << config.convergenceThreshold << std::endl;
            std::cout << "Max iterations: " << config.maxIterations << std::endl;
            std::cout << std::endl;
        }
        
        Statistics stats;
        
        // Initialize all patches
        for (auto& patch : scene.patches) {
            patch.initializeRadiosity();
        }
        
        // Compute initial energy
        stats.initialEnergy = computeTotalEnergy(scene.patches);
        
        if (config.verbose) {
            std::cout << "Initial energy: " << stats.initialEnergy << std::endl;
            std::cout << "\nBeginning progressive refinement...\n" << std::endl;
        }
        
        // Pre-calculate visibility AND form factors using Monte Carlo
        // Returns: {formFactors, visibilityMatrix}
        auto result = 
            radiosity::MonteCarloFormFactor::calculateMatrixWithVisibility(scene.patches, visibilityTester, config.verbose, config.debugNoDistance);
        formFactors = result.first;
        visibilityMatrix = result.second;
        
        if (config.verbose) {
            std::cout << "\n=== FORM FACTORS FROM LIGHT (Patch 15) ===" << std::endl;
            const Patch& light = scene.patches[15];
            std::cout << "  Light: center=(" << light.center.x << ", " << light.center.y << ", " << light.center.z 
                      << "), normal=(" << light.normal.x << ", " << light.normal.y << ", " << light.normal.z 
                      << "), area=" << light.area << std::endl;
            for (size_t i = 0; i < scene.patches.size(); i++) {
                if (i == 15) continue;  // Skip self
                float F_light_i = formFactors[15][i];
                if (F_light_i > 1e-9f) {
                    const Patch& target = scene.patches[i];
                    std::cout << "  F(15→" << i << ") = " << std::scientific 
                              << std::setprecision(6) << F_light_i 
                              << "  [target center=(" << std::fixed << std::setprecision(1)
                              << target.center.x << "," << target.center.y << "," << target.center.z 
                              << "), normal=(" << std::setprecision(2)
                              << target.normal.x << "," << target.normal.y << "," << target.normal.z << ")]"
                              << std::endl;
                }
            }
            std::cout << std::endl;
        }
        
        // Full radiosity iteration loop (n² approach)
        // In each iteration, ALL patches with unshot energy shoot to ALL other patches
        for (stats.iterations = 0; stats.iterations < config.maxIterations; stats.iterations++) {
            // Check convergence BEFORE iteration
            stats.unshotEnergy = computeUnshotEnergy(scene.patches);
            if (stats.unshotEnergy < config.convergenceThreshold) {
                if (config.verbose) {
                    std::cout << "\nConverged at iteration " << stats.iterations << std::endl;
                    std::cout << "Unshot energy: " << stats.unshotEnergy << std::endl;
                }
                stats.converged = true;
                break;
            }
            
            // Print iteration header
            if (config.verbose && (stats.iterations < 10 || stats.iterations % 10 == 0)) {
                std::cout << "\n=== Iteration " << stats.iterations 
                          << " | Unshot energy: " << std::setprecision(6) << stats.unshotEnergy << " ===" << std::endl;
            }
            
            // Accumulate energy transfers from ALL patches with unshot energy
            // This is the n² approach: every patch shoots to every other patch
            std::vector<Vector3> deltaB(scene.patches.size(), Vector3(0, 0, 0));
            
            int shootingPatches = 0;
            float maxUnshotMag = 0.0f;
            
            // For each patch with unshot energy
            for (size_t i = 0; i < scene.patches.size(); i++) {
                const Patch& shooter = scene.patches[i];
                float unshotMag = shooter.unshotMagnitude();
                
                if (unshotMag < 1e-8f) continue; // Skip patches with negligible unshot energy
                
                shootingPatches++;
                if (unshotMag > maxUnshotMag) {
                    maxUnshotMag = unshotMag;
                }
                
                // Debug: Show first few shooting patches in first 3 iterations
                if (config.verbose && stats.iterations < 3 && shootingPatches <= 5) {
                    std::cout << "  Patch " << std::setw(4) << i 
                              << " shoots: B_unshot=(" << std::setprecision(4) 
                              << shooter.B_unshot.x << "," << shooter.B_unshot.y << "," << shooter.B_unshot.z << ")"
                              << " | mag=" << std::setprecision(6) << unshotMag << std::endl;
                }
                
                // Shoot to all other patches (n² operation)
                for (size_t j = 0; j < scene.patches.size(); j++) {
                    if (i == j) continue;
                    
                    const Patch& receiver = scene.patches[j];
                    float F_ij = formFactors[i][j];
                    
                    // Energy transfer: ΔB_j = ρ_j × F_ij × B_unshot[i]
                    Vector3 transfer = receiver.reflectance * (F_ij * shooter.B_unshot);
                    deltaB[j] += transfer;
                }
            }
            
            if (config.verbose && (stats.iterations < 10 || stats.iterations % 10 == 0)) {
                std::cout << "  → " << shootingPatches << " patches shot energy (max unshot: " 
                          << std::setprecision(6) << maxUnshotMag << ")" << std::endl;
            }
            
            stats.maxUnshotPatch = maxUnshotMag;
            
            // Apply accumulated energy and clear B_unshot for all patches
            for (size_t i = 0; i < scene.patches.size(); i++) {
                scene.patches[i].B += deltaB[i];
                scene.patches[i].B_unshot = deltaB[i]; // Next iteration's unshot energy
            }
        }
        
        // Final statistics
        stats.finalEnergy = computeTotalEnergy(scene.patches);
        stats.unshotEnergy = computeUnshotEnergy(scene.patches);
        
        if (config.verbose) {
            // Energy conservation check
            float emissiveEnergy = 0.0f;
            for (const auto& patch : scene.patches) {
                if (patch.isEmissive()) {
                    emissiveEnergy += patch.emission.x * patch.area;
                }
            }
            float energyMultiplier = (emissiveEnergy > 1e-6f) ? (stats.finalEnergy / emissiveEnergy) : 0.0f;
            
            std::cout << "\n=== ENERGY CONSERVATION CHECK ===" << std::endl;
            std::cout << "Initial emissive energy: " << emissiveEnergy << std::endl;
            std::cout << "Final total energy: " << stats.finalEnergy << std::endl;
            std::cout << "Energy multiplier: " << energyMultiplier << std::endl;
            std::cout << "Expected (closed room): 1 / (1 - ρ_avg) ≈ 2.0 for ρ=0.5" << std::endl;
            std::cout << std::endl;
            
            printFinalStatistics(stats);
            
            // Print ALL patch brightnesses for debugging
            std::cout << "=== ALL PATCH BRIGHTNESSES ===" << std::endl;
            for (size_t i = 0; i < scene.patches.size(); i++) {
                const Patch& p = scene.patches[i];
                std::cout << "  Patch " << std::setw(2) << i 
                          << ": B=(" << std::fixed << std::setprecision(2) 
                          << p.B.x << ", " << p.B.y << ", " << p.B.z << ")"
                          << " n=(" << std::setprecision(2)
                          << p.normal.x << "," << p.normal.y << "," << p.normal.z << ")"
                          << (p.isEmissive() ? " [LIGHT]" : "")
                          << std::endl;
            }
            std::cout << std::endl;
        }
        
        // DEBUG: Show reflectance of colored walls
        if (config.verbose) {
            std::cout << "\n=== DEBUG: Wall Reflectances ===" << std::endl;
            for (size_t i = 0; i < std::min(size_t(50), scene.patches.size()); i++) {
                const Patch& p = scene.patches[i];
                if (p.reflectance.x != p.reflectance.y || p.reflectance.y != p.reflectance.z) {
                    std::cout << "  Patch " << i << ": reflectance=(" 
                              << p.reflectance.x << ", " << p.reflectance.y << ", " << p.reflectance.z 
                              << ") - COLORED WALL" << std::endl;
                }
            }
        }
        
        // Debug: Check form factors TO patch 6 (black box face)
        if (config.verbose && scene.patches.size() > 6) {
            std::cout << "\n=== DEBUG: Form factors TO Patch 6 ===" << std::endl;
            std::cout << "Patch 6 properties: center=(" << scene.patches[6].center.x << "," 
                      << scene.patches[6].center.y << "," << scene.patches[6].center.z 
                      << "), normal=(" << scene.patches[6].normal.x << ","
                      << scene.patches[6].normal.y << "," << scene.patches[6].normal.z << ")" << std::endl;
            std::cout << "Can receive from:" << std::endl;
            int count = 0;
            for (size_t i = 0; i < scene.patches.size(); i++) {
                if (i == 6) continue;
                float F_i6 = formFactors[i][6];
                if (F_i6 > 1e-9f) {
                    std::cout << "  F(" << i << "->6) = " << std::scientific << F_i6 
                              << " (patch " << i << " B=" << std::fixed << std::setprecision(2)
                              << scene.patches[i].B.x << ")" << std::endl;
                    count++;
                }
            }
            if (count == 0) {
                std::cout << "  NO patches can illuminate patch 6! (all form factors are zero)" << std::endl;
            }
            std::cout << std::endl;
        }
        
        return stats;
    }
    
    /**
     * Get form factor matrix (for debugging/visualization)
     */
    const std::vector<std::vector<float>>& getFormFactors() const {
        return formFactors;
    }
    
    /**
     * Get pure visibility matrix (0.0-1.0 ray hit fractions)
     * For accumulated visibility visualization WITHOUT geometric terms
     */
    const std::vector<std::vector<float>>& getVisibilityMatrix() const {
        return visibilityMatrix;
    }
    
private:
    Config config;
    std::vector<std::vector<float>> formFactors;      // Full form factors (visibility * geometry)
    std::vector<std::vector<float>> visibilityMatrix; // Pure visibility fractions [0,1]
    
    /**
     * Find patch with maximum unshot radiosity magnitude
     * Returns -1 if no patch has unshot radiosity
     */
    int findBrightestPatch(const std::vector<Patch>& patches) const {
        int brightestIdx = -1;
        float maxUnshot = 0.0f;
        
        for (size_t i = 0; i < patches.size(); i++) {
            float unshot = patches[i].unshotMagnitude();
            if (unshot > maxUnshot) {
                maxUnshot = unshot;
                brightestIdx = static_cast<int>(i);
            }
        }
        
        return brightestIdx;
    }
    
    /**
     * Shoot radiosity from shooter patch to all others
     */
    void shootRadiosity(
        std::vector<Patch>& patches,
        int shooterIdx,
        const std::vector<std::vector<float>>& formFactors)
    {
        Patch& shooter = patches[shooterIdx];
        Vector3 deltaRad = shooter.B_unshot;
        
        // Shoot to all other patches
        for (size_t j = 0; j < patches.size(); j++) {
            if (static_cast<int>(j) == shooterIdx) continue;
            
            Patch& receiver = patches[j];
            float F_ij = formFactors[shooterIdx][j];
            
            if (F_ij > 1e-6f) {  // Skip negligible form factors
                // Energy received = shooter unshot * receiver reflectance * form factor
                float reflectanceMultiplier = config.debugAttenuateReflectance ? 2.0f : 1.0f;
                Vector3 deltaB = Vector3(
                    deltaRad.x * receiver.reflectance.x * F_ij * reflectanceMultiplier,
                    deltaRad.y * receiver.reflectance.y * F_ij * reflectanceMultiplier,
                    deltaRad.z * receiver.reflectance.z * F_ij * reflectanceMultiplier
                );
                
                // ATOMIC DEBUG: Show color transfer for first colored receiver
                static int debugCount = 0;
                if (debugCount < 5 && (receiver.reflectance.x != receiver.reflectance.y || receiver.reflectance.y != receiver.reflectance.z)) {
                    std::cout << "      → Transfer to patch " << j << ": deltaRad=(" << deltaRad.x << "," << deltaRad.y << "," << deltaRad.z
                              << ") × reflectance=(" << receiver.reflectance.x << "," << receiver.reflectance.y << "," << receiver.reflectance.z
                              << ") × F=" << F_ij << " = deltaB=(" << deltaB.x << "," << deltaB.y << "," << deltaB.z << ")" << std::endl;
                    debugCount++;
                }
                
                // Update receiver's radiosity and unshot radiosity
                receiver.B = receiver.B + deltaB;
                receiver.B_unshot = receiver.B_unshot + deltaB;
            }
        }
        
        // Clear shooter's unshot radiosity
        shooter.B_unshot = Vector3(0.0f, 0.0f, 0.0f);
    }
    
    /**
     * Compute total radiosity energy in scene
     */
    float computeTotalEnergy(const std::vector<Patch>& patches) const {
        float total = 0.0f;
        for (const auto& patch : patches) {
            // Energy = average radiosity * area
            float avgB = (patch.B.x + patch.B.y + patch.B.z) / 3.0f;
            total += avgB * patch.area;
        }
        return total;
    }
    
    /**
     * Compute total unshot energy in scene
     */
    float computeUnshotEnergy(const std::vector<Patch>& patches) const {
        float total = 0.0f;
        for (const auto& patch : patches) {
            total += patch.unshotMagnitude();
        }
        return total;
    }
    
    /**
     * Print final statistics
     */
    void printFinalStatistics(const Statistics& stats) const {
        std::cout << "\n=== RADIOSITY SOLUTION COMPLETE ===" << std::endl;
        std::cout << "Iterations: " << stats.iterations << std::endl;
        std::cout << "Converged: " << (stats.converged ? "YES" : "NO (max iterations reached)") << std::endl;
        std::cout << "Initial energy: " << stats.initialEnergy << std::endl;
        std::cout << "Final energy: " << stats.finalEnergy << std::endl;
        std::cout << "Unshot energy: " << stats.unshotEnergy << std::endl;
        
        // Energy conservation check
        float energyDiff = std::abs(stats.finalEnergy - stats.initialEnergy);
        float energyError = (stats.initialEnergy > 0.0f) ? 
            (energyDiff / stats.initialEnergy) * 100.0f : 0.0f;
        
        std::cout << "Energy difference: " << energyDiff 
                  << " (" << std::fixed << std::setprecision(2) << energyError << "%)" << std::endl;
        
        if (energyError < 5.0f) {
            std::cout << "✓ Energy conservation: PASS" << std::endl;
        } else {
            std::cout << "⚠ Energy conservation: WARNING (error > 5%)" << std::endl;
        }
        
        std::cout << std::endl;
    }
};

} // namespace radiosity
