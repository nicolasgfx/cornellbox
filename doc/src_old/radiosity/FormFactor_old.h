#pragma once

// Prevent Windows.h from defining min/max macros
#define NOMINMAX

#include "math/Vector3.h"
#include "math/MathUtils.h"
#include "core/Patch.h"
#include "visibility/VisibilityTester.h"
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace radiosity {

using math::Vector3;
using math::PI;
using core::Patch;
using visibility::VisibilityTester;

/**
 * Form Factor Calculator
 * 
 * Computes form factors (view factors) between patches using Monte Carlo integration:
 * F_ij = (1 / A_i) ∫∫ (cos θi * cos θj * V(x,y)) / (π * r²) dA_j dA_i
 * 
 * Monte Carlo approximation:
 * F_ij ≈ (1/N) Σ (cos θi * cos θj * V(x,y)) / (π * r²)
 * 
 * Where:
 * - N = number of sample pairs
 * - θi = angle between patch i normal and vector from sample on i to sample on j
 * - θj = angle between patch j normal and vector from sample on j to sample on i  
 * - V(x,y) = visibility between sample points (0 = occluded, 1 = visible)
 * - r = distance between sample points
 * 
 * Form factors satisfy:
 * - Reciprocity: A_i * F_ij = A_j * F_ji
 * - Sum constraint: Σ F_ij ≈ 1.0 (closed environment)
 */
class FormFactorCalculator {
public:
    struct Config {
        int samplesPerPatch = 16;      // N samples per patch for Monte Carlo integration
        bool useMonteCarloSampling = true;  // Use MC sampling (vs point-to-point)
        bool verbose = false;           // Debug output
    };
    
    FormFactorCalculator(const Config& config = Config()) 
        : config(config) {}
    
    /**
     * Calculate form factor from patch i to patch j using Monte Carlo sampling
     * Samples points on both patch surfaces and averages visibility
     */
    float calculate(
        const Patch& patchI, 
        const Patch& patchJ,
        const VisibilityTester* visibilityTester = nullptr) const
    {
        // Don't compute self-interaction
        if (&patchI == &patchJ) {
            return 0.0f;
        }
        
        // Quick rejection: patches facing away
        Vector3 centerDir = (patchJ.center - patchI.center).normalized();
        float centerCosI = patchI.normal.dot(centerDir);
        float centerCosJ = patchJ.normal.dot(-centerDir);
        
        if (centerCosI <= 0.0f || centerCosJ <= 0.0f) {
            return 0.0f;
        }
        
        // Use Monte Carlo sampling if enabled, otherwise fall back to point-to-point
        if (config.useMonteCarloSampling && config.samplesPerPatch > 1) {
            return calculateMonteCarlo(patchI, patchJ, visibilityTester);
        } else {
            return calculatePointToPoint(patchI, patchJ, visibilityTester);
        }
    }
    
private:
    Config config;
    
    /**
     * Point-to-point approximation (original method - fast but inaccurate)
     */
    float calculatePointToPoint(
        const Patch& patchI,
        const Patch& patchJ,
        const VisibilityTester* visibilityTester) const
    {
        // Don't compute self-interaction
        if (&patchI == &patchJ) {
            return 0.0f;
        }
        
    /**
     * Point-to-point approximation (original method - fast but inaccurate)
     */
    float calculatePointToPoint(
        const Patch& patchI,
        const Patch& patchJ,
        const VisibilityTester* visibilityTester) const
    {
        // Vector from i to j
        Vector3 r_ij = patchJ.center - patchI.center;
        float distanceSquared = r_ij.lengthSquared();
        
        // Avoid division by zero for coincident patches
        if (distanceSquared < 1e-6f) {
            return 0.0f;
        }
        
        float distance = std::sqrt(distanceSquared);
        Vector3 direction = r_ij / distance;
        
        // Angle between patch i normal and direction to j
        float cosTheta_i = patchI.normal.dot(direction);
        
        // Angle between patch j normal and direction from j to i (flip direction)
        float cosTheta_j = patchJ.normal.dot(-direction);
        
        // Both patches must face each other
        if (cosTheta_i <= 0.0f || cosTheta_j <= 0.0f) {
            return 0.0f;
        }
        
        // Visibility test (defaults to 1.0 if no tester provided)
        float visibility = 1.0f;
        if (visibilityTester && visibilityTester->isInitialized()) {
            visibility = visibilityTester->testVisibility(patchI.center, patchJ.center);
        }
        
        // Form factor formula:
        // F_ij = (A_j * cos θi * cos θj * V_ij) / (π * r² * A_i)
        float formFactor = (patchJ.area * cosTheta_i * cosTheta_j * visibility) / 
                          (PI * distanceSquared * patchI.area);
        
        return formFactor;
    }
    
    /**
     * Monte Carlo integration over patch surfaces
     * Samples random points on both patches and averages contributions
     */
    float calculateMonteCarlo(
        const Patch& patchI,
        const Patch& patchJ,
        const VisibilityTester* visibilityTester) const
    {
        // Build local coordinate frames for both patches
        Vector3 tangentI, bitangentI;
        buildTangentFrame(patchI.normal, tangentI, bitangentI);
        
        Vector3 tangentJ, bitangentJ;
        buildTangentFrame(patchJ.normal, tangentJ, bitangentJ);
        
        // Accumulate form factor contributions
        float sumContribution = 0.0f;
        int validSamples = 0;
        
        // Sample points on patch i
        for (int si = 0; si < config.samplesPerPatch; si++) {
            // Random point on patch i (uniform distribution)
            Vector3 pointI = samplePointOnPatch(patchI, tangentI, bitangentI);
            
            // Sample points on patch j
            for (int sj = 0; sj < config.samplesPerPatch; sj++) {
                // Random point on patch j
                Vector3 pointJ = samplePointOnPatch(patchJ, tangentJ, bitangentJ);
                
                // Vector from sample on i to sample on j
                Vector3 r_ij = pointJ - pointI;
                float distanceSquared = r_ij.lengthSquared();
                
                if (distanceSquared < 1e-6f) {
                    continue; // Skip coincident samples
                }
                
                float distance = std::sqrt(distanceSquared);
                Vector3 direction = r_ij / distance;
                
                // Cosine terms
                float cosTheta_i = patchI.normal.dot(direction);
                float cosTheta_j = patchJ.normal.dot(-direction);
                
                // Skip if patches face away at this sample
                if (cosTheta_i <= 0.0f || cosTheta_j <= 0.0f) {
                    continue;
                }
                
                // Visibility test (ray trace from pointI to pointJ)
                float visibility = 1.0f;
                if (visibilityTester && visibilityTester->isInitialized()) {
                    visibility = visibilityTester->testVisibility(pointI, pointJ);
                }
                
                // Accumulate contribution: (cos θi * cos θj * V) / (π * r²)
                float contribution = (cosTheta_i * cosTheta_j * visibility) / (PI * distanceSquared);
                sumContribution += contribution;
                validSamples++;
            }
        }
        
        // Average over all samples
        if (validSamples == 0) {
            return 0.0f;
        }
        
        // Form factor = (A_j / A_i) * average_contribution
        float formFactor = (patchJ.area / patchI.area) * (sumContribution / static_cast<float>(validSamples));
        
        return formFactor;
    }
    
    /**
     * Build tangent frame for a surface normal
     */
    void buildTangentFrame(const Vector3& normal, Vector3& tangent, Vector3& bitangent) const
    {
        // Choose an arbitrary vector not parallel to normal
        Vector3 up = std::abs(normal.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
        tangent = normal.cross(up).normalized();
        bitangent = normal.cross(tangent).normalized();
    }
    
    /**
     * Sample a random point on a patch surface
     * Assumes patch is square and centered at patch.center
     */
    Vector3 samplePointOnPatch(
        const Patch& patch,
        const Vector3& tangent,
        const Vector3& bitangent) const
    {
        // Assume square patch: area = size²
        float size = std::sqrt(patch.area);
        float halfSize = size * 0.5f;
        
        // Random offset in [-halfSize, +halfSize] for both directions
        float u = math::Random::range(-halfSize, halfSize);
        float v = math::Random::range(-halfSize, halfSize);
        
        // Point on patch surface
        return patch.center + tangent * u + bitangent * v;
    }

public:
    /**
     * Calculate all form factors from one patch to all others
     * Returns vector of form factors in same order as patches
     */
    std::vector<float> calculateFromPatch(
        const Patch& fromPatch,
        const std::vector<Patch>& allPatches,
        const VisibilityTester* visibilityTester = nullptr) const
    {
        float distanceSquared = r_ij.lengthSquared();
        
        // Avoid division by zero for coincident patches
        if (distanceSquared < 1e-6f) {
            return 0.0f;
        }
        
        float distance = std::sqrt(distanceSquared);
        Vector3 direction = r_ij / distance;
        
        // Angle between patch i normal and direction to j
        float cosTheta_i = patchI.normal.dot(direction);
        
        // Angle between patch j normal and direction from j to i (flip direction)
        float cosTheta_j = patchJ.normal.dot(-direction);
        
        // Both patches must face each other
        if (cosTheta_i <= 0.0f || cosTheta_j <= 0.0f) {
            return 0.0f;
        }
        
        // Visibility test (defaults to 1.0 if no tester provided)
        float visibility = 1.0f;
        if (visibilityTester && visibilityTester->isInitialized()) {
            visibility = visibilityTester->testPatchVisibility(patchI, patchJ);
            
            static int vis_call_count = 0;
            if (vis_call_count < 3) {
                std::cout << "[FormFactor] Visibility test #" << vis_call_count 
                          << ": patches " << &patchI << " -> " << &patchJ
                          << " visibility=" << visibility 
                          << " cosI=" << cosTheta_i << " cosJ=" << cosTheta_j << "\n";
                vis_call_count++;
            }
        }
        
        // Form factor formula (classical radiosity definition):
        // F_ij = fraction of energy leaving patch i that arrives at patch j
        // For point-to-point approximation:
        // F_ij = (A_j * cos θi * cos θj * V_ij) / (π * r² * A_i)
        float formFactor = (patchJ.area * cosTheta_i * cosTheta_j * visibility) / 
                          (PI * distanceSquared * patchI.area);
        
        static int ff_debug_count = 0;
        if (ff_debug_count < 5 && formFactor > 0.0f) {
            std::cout << "[FormFactor] Non-zero FF: " << formFactor 
                      << " A_i=" << patchI.area << " A_j=" << patchJ.area
                      << " dist=" << std::sqrt(distanceSquared) << "\n";
            ff_debug_count++;
        }
        
        return formFactor;
    }
    
    /**
     * Calculate all form factors from one patch to all others
     * Returns vector of form factors in same order as patches
     */
    static std::vector<float> calculateFromPatch(
        const Patch& fromPatch,
        const std::vector<Patch>& allPatches,
        const VisibilityTester* visibilityTester = nullptr)
    {
        std::vector<float> formFactors;
        formFactors.reserve(allPatches.size());
        
        for (const auto& toPatch : allPatches) {
            float ff = calculate(fromPatch, toPatch, visibilityTester);
            formFactors.push_back(ff);
        }
        
        return formFactors;
    }
    
    /**
     * Calculate full NxN form factor matrix
     * WARNING: O(N²) computation, can be slow for many patches
     */
    static std::vector<std::vector<float>> calculateMatrix(
        const std::vector<Patch>& patches,
        const VisibilityTester* visibilityTester = nullptr,
        bool verbose = false)
    {
        size_t N = patches.size();
        std::vector<std::vector<float>> matrix(N, std::vector<float>(N, 0.0f));
        
        if (verbose) {
            std::cout << "\nCalculating " << N << "x" << N << " form factor matrix...\n";
        }
        
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                matrix[i][j] = calculate(patches[i], patches[j], visibilityTester);
            }
            
            if (verbose && (i % 10 == 0 || i == N - 1)) {
                std::cout << "  Progress: " << (i + 1) << "/" << N << " rows computed\n";
            }
        }
        
        if (verbose) {
            std::cout << "✓ Form factor matrix complete\n";
        }
        
        return matrix;
    }
    
    /**
     * Validate form factor properties
     * Returns true if all checks pass
     */
    static bool validate(
        const std::vector<Patch>& patches,
        const std::vector<std::vector<float>>& formFactorMatrix,
        float reciprocityTolerance = 0.01f,
        float sumTolerance = 0.1f)
    {
        size_t N = patches.size();
        bool allValid = true;
        
        std::cout << "\n=== Form Factor Validation ===\n";
        
        // Test 1: Reciprocity - A_i * F_ij should equal A_j * F_ji
        std::cout << "Test 1: Reciprocity (A_i * F_ij = A_j * F_ji)\n";
        size_t reciprocityFailures = 0;
        float maxReciprocityError = 0.0f;
        
        for (size_t i = 0; i < N; i++) {
            for (size_t j = i + 1; j < N; j++) {
                float left = patches[i].area * formFactorMatrix[i][j];
                float right = patches[j].area * formFactorMatrix[j][i];
                float error = std::abs(left - right);
                
                maxReciprocityError = std::max(maxReciprocityError, error);
                
                if (error > reciprocityTolerance) {
                    reciprocityFailures++;
                    if (reciprocityFailures <= 5) {  // Show first 5 failures
                        std::cout << "  FAIL: Patch " << i << "→" << j 
                                  << " A_i*F_ij=" << left 
                                  << " A_j*F_ji=" << right 
                                  << " error=" << error << "\n";
                    }
                }
            }
        }
        
        if (reciprocityFailures == 0) {
            std::cout << "  ✓ PASS: All " << (N*(N-1)/2) << " pairs satisfy reciprocity\n";
            std::cout << "  Max error: " << maxReciprocityError << "\n";
        } else {
            std::cout << "  ✗ FAIL: " << reciprocityFailures << " pairs violate reciprocity\n";
            std::cout << "  Max error: " << maxReciprocityError << "\n";
            allValid = false;
        }
        
        // Test 2: Sum constraint - Σ F_ij should be ≈ 1.0 for closed environment
        std::cout << "\nTest 2: Sum Constraint (Σ F_ij ≈ 1.0 for closed environment)\n";
        size_t sumFailures = 0;
        float minSum = 2.0f, maxSum = 0.0f;
        
        for (size_t i = 0; i < N; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < N; j++) {
                sum += formFactorMatrix[i][j];
            }
            
            minSum = std::min(minSum, sum);
            maxSum = std::max(maxSum, sum);
            
            if (std::abs(sum - 1.0f) > sumTolerance) {
                sumFailures++;
                if (sumFailures <= 5) {
                    std::cout << "  Patch " << i << " sum: " << sum 
                              << " (error: " << (sum - 1.0f) << ")\n";
                }
            }
        }
        
        std::cout << "  Sum range: [" << minSum << ", " << maxSum << "]\n";
        if (sumFailures == 0) {
            std::cout << "  ✓ PASS: All patches have sum ≈ 1.0\n";
        } else {
            std::cout << "  ⚠ WARNING: " << sumFailures << " patches deviate from 1.0\n";
            std::cout << "  (This is expected if environment is not fully closed)\n";
        }
        
        std::cout << "\nValidation " << (allValid ? "PASSED" : "FAILED") << "\n";
        return allValid;
    }
    
    /**
     * Print form factor statistics
     */
    static void printStatistics(const std::vector<std::vector<float>>& formFactorMatrix) {
        size_t N = formFactorMatrix.size();
        
        float minFF = 1.0f, maxFF = 0.0f;
        size_t nonZeroCount = 0;
        float totalSum = 0.0f;
        
        for (size_t i = 0; i < N; i++) {
            for (size_t j = 0; j < N; j++) {
                float ff = formFactorMatrix[i][j];
                if (ff > 1e-6f) {
                    minFF = std::min(minFF, ff);
                    maxFF = std::max(maxFF, ff);
                    nonZeroCount++;
                    totalSum += ff;
                }
            }
        }
        
        std::cout << "\n=== Form Factor Statistics ===\n";
        std::cout << "Matrix size: " << N << "x" << N << " = " << (N*N) << " entries\n";
        std::cout << "Non-zero: " << nonZeroCount << " (" 
                  << (100.0f * nonZeroCount / (N*N)) << "%)\n";
        std::cout << "Range: [" << minFF << ", " << maxFF << "]\n";
        std::cout << "Average (non-zero): " << (totalSum / nonZeroCount) << "\n";
        std::cout << "Total sum: " << totalSum << " (expected ≈ " << N << ")\n";
    }
};

}  // namespace radiosity
