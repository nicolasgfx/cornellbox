#pragma once

#include "math/Vector3.h"
#include "math/MathUtils.h"
#include "core/Patch.h"
#include "visibility/VisibilityTester.h"
#include <cmath>
#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <iostream>
#include <chrono>

namespace radiosity {

using math::Vector3;
using math::PI;
using core::Patch;
using visibility::VisibilityTester;

/**
 * Monte Carlo Form Factor Calculator with SEPARATED visibility caching
 * 
 * CRITICAL: Cache stores ONLY visibility fractions (0.0 to 1.0), NOT form factors!
 * Form factors are computed fresh each time using cached visibility + geometric terms.
 * 
 * Two-stage process:
 * 1. Compute/cache visibility: Shoot 16 rays per patch pair, store hit fraction
 * 2. Compute form factors: Load visibility, compute cosines/distance/area
 */
class MonteCarloFormFactor {
public:
    static constexpr int NUM_SAMPLES = 8;   // 8 samples in 2x4 optimized grid for good coverage
    
    // DEBUG: Global flag to disable distance attenuation for testing
    static bool DEBUG_NO_DISTANCE_ATTENUATION;
    
    // Effective distance parameter (r² softening)
    // r²_eff = r² + α * A_receiver
    static constexpr float EFFECTIVE_DISTANCE_ALPHA = 0.5f;
    
    // DEBUG: Enable loading/saving VISIBILITY from/to cache file
    static bool DEBUG_ENABLE_CACHE;
    static std::string DEBUG_CACHE_FILENAME;
    
    static const std::array<Vector3, NUM_SAMPLES>& getUVSamples() {
        static const std::array<Vector3, NUM_SAMPLES> samples = {{
            // 8 samples in optimized 2x4 stratified grid for good triangle coverage
            // Row 1 (y=0.25): 4 samples
            Vector3(0.125f, 0.25f, 0.0f), Vector3(0.375f, 0.25f, 0.0f),
            Vector3(0.625f, 0.25f, 0.0f), Vector3(0.875f, 0.25f, 0.0f),
            // Row 2 (y=0.75): 4 samples
            Vector3(0.125f, 0.75f, 0.0f), Vector3(0.375f, 0.75f, 0.0f),
            Vector3(0.625f, 0.75f, 0.0f), Vector3(0.875f, 0.75f, 0.0f)
        }};
        return samples;
    }
    
    static Vector3 uvToWorld(
        const Vector3& v0, const Vector3& v1, const Vector3& v2, const Vector3& v3,
        float u, float v)
    {
        float u0 = (1.0f - u) * (1.0f - v);
        float u1 = u * (1.0f - v);
        float u2 = u * v;
        float u3 = (1.0f - u) * v;
        return v0 * u0 + v1 * u1 + v2 * u2 + v3 * u3;
    }
    
    static void getPatchVertices(
        const Patch& patch,
        Vector3& v0, Vector3& v1, Vector3& v2, Vector3& v3)
    {
        Vector3 up = std::abs(patch.normal.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
        Vector3 right = patch.normal.cross(up).normalized();
        Vector3 tangent = right.cross(patch.normal).normalized();
        
        float halfSize = std::sqrt(patch.area) * 0.5f;
        v0 = patch.center - right * halfSize - tangent * halfSize;
        v1 = patch.center + right * halfSize - tangent * halfSize;
        v2 = patch.center + right * halfSize + tangent * halfSize;
        v3 = patch.center - right * halfSize + tangent * halfSize;
    }
    
    /**
     * Compute ONLY visibility between two patches (for caching)
     * Tests 16 rays and returns fraction that hit (0.0 = fully occluded, 1.0 = fully visible)
     * THIS IS THE ONLY THING THAT SHOULD BE CACHED!
     */
    static float computeVisibility(
        const Patch& patchI,
        const Patch& patchJ,
        int patchIId,
        int patchJId,
        VisibilityTester* visTester)
    {
        if (&patchI == &patchJ) return 0.0f;
        
        // CRITICAL: Check if patches can geometrically see each other
        // Reject only backwards rays (hemisphere check)
        Vector3 centerDir = (patchJ.center - patchI.center).normalized();
        
        // Ray must point into positive hemisphere of patchI
        float cosI = patchI.normal.dot(centerDir);
        if (cosI <= 0.0f) {
            return 0.0f;  // Ray backwards - geometrically impossible
        }
        
        // Ray must hit front face of patchJ
        float cosJ = patchJ.normal.dot(-centerDir);
        if (cosJ <= 0.0f) {
            return 0.0f;  // Would hit back face - geometrically impossible
        }
        
        if (!visTester || !visTester->isInitialized()) return 1.0f;
        
        // Get patch vertices (4 corners of each quad)
        Vector3 v0_i, v1_i, v2_i, v3_i, v0_j, v1_j, v2_j, v3_j;
        getPatchVertices(patchI, v0_i, v1_i, v2_i, v3_i);
        getPatchVertices(patchJ, v0_j, v1_j, v2_j, v3_j);
        
        // Call OptiX kernel which does 8-sample Monte Carlo integration internally
        float visibility = visTester->testPatchVisibility(
            patchI, v0_i, v1_i, v2_i, v3_i,
            patchJ, v0_j, v1_j, v2_j, v3_j,
            patchIId, patchJId
        );
        
        return visibility;  // Already averaged [0.0, 1.0]
    }
    
    /**
     * Calculate form factor using PRE-COMPUTED visibility fraction
     * Computes fresh geometric terms (cosines, distance, area) every time
     */
    static float calculate(
        const Patch& patchI,
        const Patch& patchJ,
        float visibilityFraction,
        bool noDistanceAttenuation = false)
    {
        if (&patchI == &patchJ) return 0.0f;
        if (visibilityFraction < 1e-6f) return 0.0f;
        
        Vector3 centerDir = (patchJ.center - patchI.center).normalized();
        if (patchI.normal.dot(centerDir) <= 0.0f || patchJ.normal.dot(-centerDir) <= 0.0f) {
            return 0.0f;
        }
        
        Vector3 v0_i, v1_i, v2_i, v3_i, v0_j, v1_j, v2_j, v3_j;
        getPatchVertices(patchI, v0_i, v1_i, v2_i, v3_i);
        getPatchVertices(patchJ, v0_j, v1_j, v2_j, v3_j);
        
        auto samples = getUVSamples();
        float sumContribution = 0.0f;
        int validSamples = 0;
        
        for (int s = 0; s < NUM_SAMPLES; ++s) {
            const Vector3& uv = samples[s];
            Vector3 pointI = uvToWorld(v0_i, v1_i, v2_i, v3_i, uv.x, uv.y);
            Vector3 pointJ = uvToWorld(v0_j, v1_j, v2_j, v3_j, uv.x, uv.y);
            
            Vector3 r_ij = pointJ - pointI;
            float distSquared = r_ij.lengthSquared();
            if (distSquared < 1e-6f) continue;
            
            float dist = std::sqrt(distSquared);
            Vector3 dir = r_ij / dist;
            
            float cosI = patchI.normal.dot(dir);
            float cosJ = patchJ.normal.dot(-dir);
            if (cosI <= 0.0f || cosJ <= 0.0f) continue;
            
            // Use pre-computed visibility
            float effectiveDistSquared = distSquared + EFFECTIVE_DISTANCE_ALPHA * patchJ.area;
            
            float contribution;
            if (noDistanceAttenuation || DEBUG_NO_DISTANCE_ATTENUATION) {
                contribution = (cosI * cosJ * visibilityFraction) / PI;
            } else {
                contribution = (cosI * cosJ * visibilityFraction) / (PI * effectiveDistSquared);
            }
            sumContribution += contribution;
            validSamples++;
        }
        
        if (validSamples == 0) return 0.0f;
        return patchJ.area * (sumContribution / validSamples);
    }
    
    /**
     * Save VISIBILITY matrix to binary cache (values are 0.0 to 1.0)
     */
    static bool saveVisibilityToCache(const std::vector<std::vector<float>>& matrix, const std::string& filename) {
        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "ERROR: Could not open cache file for writing: " << filename << std::endl;
            return false;
        }
        
        size_t n = matrix.size();
        file.write(reinterpret_cast<const char*>(&n), sizeof(n));
        
        for (size_t i = 0; i < n; i++) {
            file.write(reinterpret_cast<const char*>(matrix[i].data()), n * sizeof(float));
        }
        
        file.close();
        std::cout << "✓ Visibility matrix cached to: " << filename << std::endl;
        std::cout << "  Size: " << n << "x" << n << " = " << (n * n) << " entries" << std::endl;
        std::cout << "  File size: " << (n * n * sizeof(float) / 1024.0f / 1024.0f) << " MB" << std::endl;
        std::cout << "  ⚠ IMPORTANT: Values are visibility fractions [0.0, 1.0], NOT form factors!" << std::endl;
        return true;
    }
    
    /**
     * Load VISIBILITY matrix from binary cache
     * IMPORTANT: Fills in symmetric values since visibility is commutative
     */
    static bool loadVisibilityFromCache(std::vector<std::vector<float>>& matrix, const std::string& filename, size_t expectedSize) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return false;
        }
        
        size_t n;
        file.read(reinterpret_cast<char*>(&n), sizeof(n));
        
        if (n != expectedSize) {
            std::cerr << "ERROR: Cache size mismatch. Expected " << expectedSize << ", got " << n << std::endl;
            file.close();
            return false;
        }
        
        matrix.resize(n, std::vector<float>(n));
        for (size_t i = 0; i < n; i++) {
            file.read(reinterpret_cast<char*>(matrix[i].data()), n * sizeof(float));
        }
        
        file.close();
        
        // NOTE: No symmetry assumption - visibility is computed/stored for ALL directions
        // Each direction i->j is independently computed and cached
        
        std::cout << "✓ Visibility matrix loaded from cache: " << filename << std::endl;
        std::cout << "  Size: " << n << "x" << n << " = " << (n * n) << " entries" << std::endl;
        std::cout << "  Values: 0.0 (occluded) to 1.0 (visible) - ray hit fractions" << std::endl;
        std::cout << "  Skipped expensive ray tracing!" << std::endl;
        return true;
    }
    
    /**
     * Calculate form factors AND visibility matrix with TWO-STAGE process:
     * Returns: pair<formFactors, visibilityMatrix>
     */
    static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    calculateMatrixWithVisibility(
        const std::vector<core::Patch>& patches,
        visibility::VisibilityTester* visTester = nullptr,
        bool verbose = false,
        bool noDistanceAttenuation = false)
    {
        auto result = calculateMatrixInternal(patches, visTester, verbose, noDistanceAttenuation);
        return {result.first, result.second}; // {formFactors, visibility}
    }
    
    /**
     * Calculate full form factor matrix (backward compatibility)
     * TWO-STAGE PROCESS:
     * 1. Compute/load visibility matrix (expensive ray tracing, cacheable)
     * 2. Compute form factors from visibility (cheap geometric computation)
     */
    static std::vector<std::vector<float>> calculateMatrix(
        const std::vector<core::Patch>& patches,
        visibility::VisibilityTester* visTester = nullptr,
        bool verbose = false,
        bool noDistanceAttenuation = false)
    {
        auto result = calculateMatrixInternal(patches, visTester, verbose, noDistanceAttenuation);
        return result.first; // Return only formFactors for backward compatibility
    }
    
private:
    /**
     * Internal implementation that returns both matrices
     */
    static std::pair<std::vector<std::vector<float>>, std::vector<std::vector<float>>> 
    calculateMatrixInternal(
        const std::vector<core::Patch>& patches,
        visibility::VisibilityTester* visTester = nullptr,
        bool verbose = false,
        bool noDistanceAttenuation = false)
    {
        size_t n = patches.size();
        
        // ========================================
        // STAGE 1: COMPUTE OR LOAD VISIBILITY
        // ========================================
        std::vector<std::vector<float>> visibilityMatrix(n, std::vector<float>(n, 0.0f));
        bool visibilityLoaded = false;
        
        if (DEBUG_ENABLE_CACHE && !DEBUG_CACHE_FILENAME.empty()) {
            if (loadVisibilityFromCache(visibilityMatrix, DEBUG_CACHE_FILENAME, n)) {
                visibilityLoaded = true;
                if (verbose) {
                    std::cout << "\n⚡ Using cached visibility - ray tracing skipped!\n" << std::endl;
                    
                    // DEBUG: Check cached values
                    std::cout << "  DEBUG: First 10 cached visibility values:\n";
                    int count = 0;
                    for (size_t i = 0; i < std::min(size_t(5), n) && count < 10; ++i) {
                        for (size_t j = 0; j < std::min(size_t(5), n) && count < 10; ++j) {
                            if (i != j) {
                                std::cout << "    cached[" << i << "][" << j << "] = " << visibilityMatrix[i][j] << "\n";
                                count++;
                            }
                        }
                    }
                }
            } else if (verbose) {
                std::cout << "Cache miss - will compute visibility and save to: " << DEBUG_CACHE_FILENAME << "\n" << std::endl;
            }
        }
        
        if (!visibilityLoaded) {
            if (verbose) {
                std::cout << "=== STAGE 1: COMPUTING VISIBILITY (Ray Tracing) ===" << std::endl;
                std::cout << "Method: 16 rays per patch pair (1:1 UV sampling)" << std::endl;
                std::cout << "Patches: " << n << std::endl;
                std::cout << "Total rays: " << (n * (n - 1) * NUM_SAMPLES) << std::endl;
                std::cout << "Starting visibility computation..." << std::endl;
                std::cout.flush();  // Force output before potentially long computation
            }
            
            auto visStartTime = std::chrono::high_resolution_clock::now();
            size_t visProcessed = 0;
            
            for (size_t i = 0; i < n; i++) {
                for (size_t j = 0; j < n; j++) {
                    if (i == j) {
                        visibilityMatrix[i][j] = 0.0f;
                    } else {
                        visibilityMatrix[i][j] = computeVisibility(patches[i], patches[j], static_cast<int>(i), static_cast<int>(j), visTester);
                    }
                }
                visProcessed++;
                
                if (verbose && (visProcessed % 100 == 0 || visProcessed == n)) {
                    auto currentTime = std::chrono::high_resolution_clock::now();
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - visStartTime).count();
                    double avgTimePerPatch = elapsed / (double)visProcessed;
                    double remainingMs = avgTimePerPatch * (n - visProcessed);
                    int remainingSec = (int)(remainingMs / 1000);
                    int remainingMin = remainingSec / 60;
                    remainingSec %= 60;
                    
                    std::cout << "Visibility: " << visProcessed << "/" << n << " patches";
                    if (remainingMin > 0) {
                        std::cout << " [ETA: " << remainingMin << "m " << remainingSec << "s]";
                    } else if (remainingSec > 0) {
                        std::cout << " [ETA: " << remainingSec << "s]";
                    }
                    std::cout << std::endl;
                    std::cout.flush();  // Force immediate output
                }
            }
            
            if (verbose) {
                auto visEndTime = std::chrono::high_resolution_clock::now();
                auto visElapsed = std::chrono::duration_cast<std::chrono::seconds>(visEndTime - visStartTime).count();
                std::cout << "✓ Visibility computation complete (" << visElapsed << "s)\n" << std::endl;
            }
            
            // Save visibility to cache
            if (DEBUG_ENABLE_CACHE && !DEBUG_CACHE_FILENAME.empty()) {
                saveVisibilityToCache(visibilityMatrix, DEBUG_CACHE_FILENAME);
                std::cout << std::endl;
            }
        }
        
        // ========================================
        // STAGE 2: COMPUTE FORM FACTORS
        // ========================================
        std::vector<std::vector<float>> matrix(n, std::vector<float>(n, 0.0f));
        
        if (verbose) {
            std::cout << "=== STAGE 2: COMPUTING FORM FACTORS (Geometric Terms) ===" << std::endl;
            std::cout << "Method: Using cached visibility + fresh geometric computation" << std::endl;
            std::cout << "Effective distance: r²_eff = r² + " << EFFECTIVE_DISTANCE_ALPHA << " * A_receiver" << std::endl;
            if (noDistanceAttenuation || DEBUG_NO_DISTANCE_ATTENUATION) {
                std::cout << "⚠ DEBUG MODE: Distance attenuation DISABLED" << std::endl;
            }
        }
        
        auto startTime = std::chrono::high_resolution_clock::now();
        size_t lastReportedPatch = 0;
        
        for (size_t i = 0; i < n; i++) {
            for (size_t j = 0; j < n; j++) {
                if (i == j) {
                    matrix[i][j] = 0.0f;
                } else {
                    // Use pre-computed visibility from matrix
                    matrix[i][j] = calculate(patches[i], patches[j], visibilityMatrix[i][j], noDistanceAttenuation);
                }
            }
            
            if (verbose && (i % 5 == 0 || i == n - 1)) {
                auto currentTime = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime).count();
                size_t patchesProcessed = i + 1;
                double avgTimePerPatch = elapsed / (double)patchesProcessed;
                double estimatedRemainingMs = avgTimePerPatch * (n - patchesProcessed);
                
                int remainingSec = (int)(estimatedRemainingMs / 1000);
                int remainingMin = remainingSec / 60;
                remainingSec %= 60;
                
                std::cout << "Form factors: " << patchesProcessed << "/" << n << " patches";
                if (remainingMin > 0) {
                    std::cout << " [ETA: " << remainingMin << "m " << remainingSec << "s]";
                } else if (remainingSec > 0) {
                    std::cout << " [ETA: " << remainingSec << "s]";
                }
                std::cout << std::endl;
                lastReportedPatch = patchesProcessed;
            }
        }
        
        if (verbose) {
            auto endTime = std::chrono::high_resolution_clock::now();
            auto totalElapsed = std::chrono::duration_cast<std::chrono::seconds>(endTime - startTime).count();
            std::cout << "✓ Form factor computation complete (" << totalElapsed << "s)\n" << std::endl;
            
            // Validate reciprocity
            float maxReciprocityError = 0.0f;
            for (size_t i = 0; i < n; i++) {
                for (size_t j = i + 1; j < n; j++) {
                    float Ai = patches[i].area;
                    float Aj = patches[j].area;
                    float leftSide = Ai * matrix[i][j];
                    float rightSide = Aj * matrix[j][i];
                    
                    if (leftSide > 1e-9f || rightSide > 1e-9f) {
                        float error = std::abs(leftSide - rightSide) / std::max(leftSide, rightSide);
                        maxReciprocityError = std::max(maxReciprocityError, error);
                    }
                }
            }
            
            std::cout << "=== FORM FACTOR STATISTICS ===" << std::endl;
            std::cout << "Maximum reciprocity error: " << (maxReciprocityError * 100.0f) << "%" << std::endl;
        }
        
        if (verbose) {
            std::cout << "\n=== RADIOSITY SOLUTION COMPLETE ===\n" << std::endl;
        }
        
        return {matrix, visibilityMatrix}; // Return both form factors and visibility
    }
    
public:
};

// Define static members
inline bool MonteCarloFormFactor::DEBUG_NO_DISTANCE_ATTENUATION = false;
// DISABLED: Force recomputation on every run (do not load cache by default)
inline bool MonteCarloFormFactor::DEBUG_ENABLE_CACHE = false;
inline std::string MonteCarloFormFactor::DEBUG_CACHE_FILENAME = "visibility.cache";

} // namespace radiosity
