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
 * Form Factor Calculator (Point-to-Point)
 * 
 * Computes form factors between patch centers using single ray visibility:
 * F_ij = (A_j * cos θi * cos θj * V) / (π * r² * A_i)
 */
class FormFactorCalculator {
public:
    /**
     * Calculate form factor from patch i to patch j (center-to-center)
     */
    static float calculate(
        const Patch& patchI, 
        const Patch& patchJ,
        const VisibilityTester* visibilityTester = nullptr)
    {
        // Don't compute self-interaction
        if (&patchI == &patchJ) {
            return 0.0f;
        }
        
        // Vector from i to j
        Vector3 r_ij = patchJ.center - patchI.center;
        float distSquared = r_ij.lengthSquared();
        
        if (distSquared < 1e-6f) {
            return 0.0f;
        }
        
        float dist = std::sqrt(distSquared);
        Vector3 dir = r_ij / dist;
        
        // Angles
        float cosI = patchI.normal.dot(dir);
        float cosJ = patchJ.normal.dot(-dir);
        
        // Patches must face each other
        if (cosI <= 0.0f || cosJ <= 0.0f) {
            return 0.0f;
        }
        
        // Visibility check
        float visibility = 1.0f;
        if (visibilityTester && visibilityTester->isInitialized()) {
            visibility = visibilityTester->testVisibility(patchI.center, patchJ.center) ? 1.0f : 0.0f;
        }
        
        // Form factor formula
        float formFactor = (patchJ.area * cosI * cosJ * visibility) / (PI * distSquared * patchI.area);
        
        return formFactor;
    }
    
    /**
     * Calculate form factors from one patch to all others
     */
    static std::vector<float> calculateFromPatch(
        const std::vector<Patch>& patches,
        size_t patchIndex,
        const VisibilityTester* visibilityTester = nullptr)
    {
        std::vector<float> formFactors(patches.size());
        const Patch& patchI = patches[patchIndex];
        
        for (size_t j = 0; j < patches.size(); ++j) {
            formFactors[j] = calculate(patchI, patches[j], visibilityTester);
        }
        
        return formFactors;
    }
    
    /**
     * Calculate full form factor matrix
     */
    static std::vector<std::vector<float>> calculateMatrix(
        const std::vector<Patch>& patches,
        const VisibilityTester* visibilityTester = nullptr,
        bool verbose = false)
    {
        size_t n = patches.size();
        std::vector<std::vector<float>> matrix(n, std::vector<float>(n, 0.0f));
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i != j) {
                    matrix[i][j] = calculate(patches[i], patches[j], visibilityTester);
                }
            }
        }
        
        if (verbose) {
            std::cout << "\n=== Form Factor Matrix ===\n";
            for (size_t i = 0; i < n; ++i) {
                std::cout << "Row " << i << ": ";
                for (size_t j = 0; j < n; ++j) {
                    std::cout << std::scientific << matrix[i][j] << " ";
                }
                std::cout << "\n";
            }
        }
        
        return matrix;
    }
};

} // namespace radiosity
