#pragma once

//
// FormFactor.h - DEPRECATED - OLD APPROACH
// 
// This file contains the OLD centroid-based form factor computation.
// It has been replaced by HemisphereFormFactorKernels.cu + FormFactorPrecompute.h
// 
// NEW APPROACH: Hemispherical ray casting with Monte Carlo sampling
// - See: src/radiosity/FormFactorPrecompute.h
// - Kernel: src/gpu/HemisphereFormFactorKernels.cu
// - Storage: src/radiosity/FormFactorCache.h (CSR format)
//
// This file is kept for reference only.
//

#include <vector>
#include <algorithm>
#include <cmath>
#include <limits>
#include "../mesh/MeshData.h"

namespace FormFactor {

// DEPRECATED: Old centroid-based approach
// Use FormFactorPrecompute::computeFormFactors() instead
inline void computeFormFactors_OLD(const PatchSoA& patches,
                                const std::vector<float>& visibilityCache,
                                const std::vector<float>& geometricKernelCache,
                                std::vector<std::vector<float>>& formFactors) {
    uint32_t N = static_cast<uint32_t>(patches.area.size());
    
    formFactors.resize(N);
    for (uint32_t i = 0; i < N; ++i) {
        formFactors[i].resize(N, 0.0f);
    }
    
    std::cout << "Computing form factors for " << N << " patches...\n";
    
    // Find an emissive patch for debugging
    uint32_t emissivePatch = UINT32_MAX;
    for (uint32_t i = 0; i < N; ++i) {
        if (patches.emit_r[i] > 0.1f || patches.emit_g[i] > 0.1f || patches.emit_b[i] > 0.1f) {
            emissivePatch = i;
            std::cout << "\n[DEBUG] Found emissive patch " << i << ":\n";
            std::cout << "  Area: " << patches.area[i] << "\n";
            std::cout << "  Normal: (" << patches.nx[i] << ", " << patches.ny[i] << ", " << patches.nz[i] << ")\n";
            std::cout << "  Centroid: (" << patches.cx[i] << ", " << patches.cy[i] << ", " << patches.cz[i] << ")\n";
            std::cout << "  Emission: (" << patches.emit_r[i] << ", " << patches.emit_g[i] << ", " << patches.emit_b[i] << ")\n";
            break;
        }
    }
    
    uint32_t totalPairs = 0;
    uint32_t visiblePairs = 0;
    uint32_t nonZeroFormFactors = 0;
    uint32_t debugCount = 0;
    
    // Process all triangle pairs
    for (uint32_t i = 0; i < N; ++i) {
        if (i % 500 == 0) {
            std::cout << "  Progress: " << i << "/" << N << "\r" << std::flush;
        }
        
        float area_i = patches.area[i];
        float cx_i = patches.cx[i];
        float cy_i = patches.cy[i];
        float cz_i = patches.cz[i];
        float nx_i = patches.nx[i];
        float ny_i = patches.ny[i];
        float nz_i = patches.nz[i];
        
        for (uint32_t j = 0; j < N; ++j) {
            if (i == j) continue;
            
            totalPairs++;
            
            // Get precomputed values from OptiX caches
            float V_ij, K_avg;
            if (i < j) {
                size_t idx = (size_t(i) * (2 * N - i - 1)) / 2 + (j - i - 1);
                V_ij = visibilityCache[idx];
                K_avg = geometricKernelCache[idx];
            } else {
                size_t idx = (size_t(j) * (2 * N - j - 1)) / 2 + (i - j - 1);
                V_ij = visibilityCache[idx];
                K_avg = geometricKernelCache[idx];
            }
             
            // Skip if no valid samples (both should be zero)
            if (V_ij < 1e-6f || K_avg < 1e-12f) continue;
         
            visiblePairs++;
            
            float area_j = patches.area[j];
            
            // Form factor from area sampling:
            // F[i,j] = (A_j / N_samples) * Σ V_k * K_k
            // Where K_avg = (1/N_samples) * Σ K_k
            // So: F[i,j] = A_j * K_avg * (N_samples / N_samples) * V_ij
            // Simplifies to: F[i,j] = A_j * K_avg * V_ij
            // 
            // Note: K_avg already accounts for the 1/N_samples averaging
            // V_ij is the visibility fraction
            // A_j gives the correct area weighting
            float F = area_j * K_avg * V_ij;
            
            formFactors[i][j] = F;
            
            if (F > 1e-8f) {
                nonZeroFormFactors++;
            }
            
            // Debug output for first few pairs from emissive patch
            if (i == emissivePatch && debugCount < 5 && F > 1e-6f) {
                std::cout << "\n[DEBUG] Form factor from emissive patch " << i << " to patch " << j << ":\n";
                std::cout << "  Visibility V: " << V_ij << "\n";
                std::cout << "  Averaged kernel K_avg: " << K_avg << "\n";
                std::cout << "  Area_j/Area_i: " << (area_j/area_i) << "\n";
                std::cout << "  Form factor F[" << i << "][" << j << "]: " << F << "\n";
                debugCount++;
            }
        }
    }
    
    std::cout << "  Progress: " << N << "/" << N << " - Done!\n";
    std::cout << "\nForm factor statistics:\n";
    std::cout << "  Total pairs: " << totalPairs << "\n";
    std::cout << "  Visible pairs (V > 0): " << visiblePairs << " (" 
              << (100.0f * visiblePairs / totalPairs) << "%)\n";
    std::cout << "  Non-zero form factors: " << nonZeroFormFactors << " (" 
              << (100.0f * nonZeroFormFactors / totalPairs) << "%)\n";
    
    // Row-sum diagnostics before normalization to catch stability issues early
    float minRowSum = std::numeric_limits<float>::max();
    float maxRowSum = 0.0f;
    double avgAccumulator = 0.0;
    uint32_t aboveOneRows = 0;
    uint32_t aboveHardLimit = 0;
    const float softLimit = 1.0f + 1e-3f;
    const float hardLimit = 1.5f;

    for (uint32_t i = 0; i < N; ++i) {
        float rowSum = 0.0f;
        for (uint32_t j = 0; j < N; ++j) {
            rowSum += formFactors[i][j];
        }
        
        minRowSum = std::min(minRowSum, rowSum);
        maxRowSum = std::max(maxRowSum, rowSum);
        avgAccumulator += rowSum;
        
        if (rowSum > softLimit) {
            aboveOneRows++;
        }
        if (rowSum > hardLimit) {
            aboveHardLimit++;
        }
    }

    if (N == 0) {
        minRowSum = 0.0f;
        maxRowSum = 0.0f;
    }

    double avgRowSum = (N > 0) ? (avgAccumulator / static_cast<double>(N)) : 0.0;
    std::cout << "\nRow-sum stats: min " << minRowSum
              << ", max " << maxRowSum
              << ", avg " << static_cast<float>(avgRowSum)
              << ", count>1 " << aboveOneRows
              << ", count>" << hardLimit << " " << aboveHardLimit << "\n";

    // Normalize form factors so each row sums to ≤ 1.0
    std::cout << "\n--- Normalizing Form Factors ---\n";
    uint32_t normalizedRows = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float rowSum = 0.0f;
        for (uint32_t j = 0; j < N; ++j) {
            rowSum += formFactors[i][j];
        }

        if (rowSum > 1.0f) {
            float scale = 1.0f / rowSum;
            for (uint32_t j = 0; j < N; ++j) {
                formFactors[i][j] *= scale;
            }
            normalizedRows++;
        }
    }

    std::cout << "Normalized " << normalizedRows << "/" << N << " rows with sum > 1.0\n";
}

// Validate form factors (optional)
inline void validateFormFactors(const std::vector<std::vector<float>>& formFactors,
                                 const PatchSoA& patches) {
    uint32_t N = static_cast<uint32_t>(formFactors.size());
    
    float maxSum = 0.0f;
    uint32_t violations = 0;
    uint32_t severeViolations = 0;
    float maxFormFactor = 0.0f;
    
    for (uint32_t i = 0; i < N; ++i) {
        float sum = 0.0f;
        for (uint32_t j = 0; j < N; ++j) {
            sum += formFactors[i][j];
            maxFormFactor = std::max(maxFormFactor, formFactors[i][j]);
        }
        
        if (sum > maxSum) maxSum = sum;
        if (sum > 1.05f) violations++;
        if (sum > 2.0f) severeViolations++;
    }
    
    // Check reflectances
    float maxRho = 0.0f;
    uint32_t rhoViolations = 0;
    for (uint32_t i = 0; i < N; ++i) {
        float rho = std::max({patches.rho_r[i], patches.rho_g[i], patches.rho_b[i]});
        maxRho = std::max(maxRho, rho);
        if (rho > 1.0f) rhoViolations++;
    }
    
    std::cout << "\nForm factor validation:\n";
    std::cout << "  Max sum_j F[i,j]: " << maxSum << " (should be ≤ 1.0)\n";
    std::cout << "  Max individual F[i,j]: " << maxFormFactor << "\n";
    std::cout << "  Violations (sum > 1.05): " << violations << "/" << N << "\n";
    std::cout << "  Severe violations (sum > 2.0): " << severeViolations << "/" << N << "\n";
    std::cout << "  Max reflectance: " << maxRho << " (should be ≤ 1.0)\n";
    std::cout << "  Reflectance violations: " << rhoViolations << "/" << N << "\n";
    
    if (maxSum > 1.5f || severeViolations > 0) {
        std::cout << "\n  WARNING: Form factors are unstable! Solver will likely diverge.\n";
        std::cout << "  Consider using form factor normalization or clamping.\n";
    }
}

} // namespace FormFactor
