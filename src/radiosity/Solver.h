#pragma once

#include <vector>
#include <algorithm>
#include <cmath>
#include "../mesh/MeshData.h"
#include "FormFactorCache.h"

namespace Solver {

// Progressive refinement radiosity solver using CSR form factors
inline void progressiveRefinement(PatchSoA& patches,
                                   const FormFactorCache::FormFactorCSR& formFactors,
                                   uint32_t maxIterations = 100,
                                   float epsilon = 0.001f) {
    uint32_t N = static_cast<uint32_t>(patches.area.size());
    
    if (N != formFactors.numPatches) {
        std::cerr << "ERROR: Patch count mismatch (patches: " << N 
                  << ", form factors: " << formFactors.numPatches << ")\n";
        return;
    }
    
    std::cout << "\n=== Progressive Refinement Solver (Hemisphere Form Factors) ===\n";
    std::cout << "Patches: " << N << "\n";
    std::cout << "Max iterations: " << maxIterations << "\n";
    std::cout << "Convergence threshold: " << epsilon << "\n";
    std::cout << "Form factor density: " << (100.0 * formFactors.nonZeros() / (uint64_t(N) * N)) << "%\n\n";
    
    // Enforce reflectance bounds to keep the solve stable
    uint32_t clampedBelow = 0;
    uint32_t clampedAbove = 0;
    for (uint32_t i = 0; i < N; ++i) {
        auto clampRho = [&](float& value) {
            if (value < 0.0f) {
                value = 0.0f;
                clampedBelow++;
            } else if (value > 1.0f) {
                value = 1.0f;
                clampedAbove++;
            }
        };

        clampRho(patches.rho_r[i]);
        clampRho(patches.rho_g[i]);
        clampRho(patches.rho_b[i]);
    }

    if (clampedBelow > 0 || clampedAbove > 0) {
        std::cout << "Adjusted reflectance values (" << clampedBelow
                  << " below 0, " << clampedAbove << " above 1).\n";
    }

    // Initialize: B = emit, Bu = emit
    for (uint32_t i = 0; i < N; ++i) {
        patches.B_r[i] = patches.emit_r[i];
        patches.B_g[i] = patches.emit_g[i];
        patches.B_b[i] = patches.emit_b[i];
        
        patches.Bu_r[i] = patches.emit_r[i];
        patches.Bu_g[i] = patches.emit_g[i];
        patches.Bu_b[i] = patches.emit_b[i];
    }
    
    // Debug: Count emissive patches
    uint32_t emissiveCount = 0;
    float totalEmission = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        float emit = patches.emit_r[i] + patches.emit_g[i] + patches.emit_b[i];
        if (emit > 1e-6f) {
            emissiveCount++;
            totalEmission += emit;
        }
    }
    std::cout << "Emissive patches: " << emissiveCount << " (total emission: " << totalEmission << ")\n";
    std::cout << std::endl;
    
    // Progressive refinement iterations
    for (uint32_t iter = 0; iter < maxIterations; ++iter) {
        // Find patch with maximum unshot radiosity
        float maxBu = 0.0f;
        uint32_t maxPatch = 0;
        
        for (uint32_t i = 0; i < N; ++i) {
            float Bu_mag = std::abs(patches.Bu_r[i]) + 
                          std::abs(patches.Bu_g[i]) + 
                          std::abs(patches.Bu_b[i]);
            if (Bu_mag > maxBu) {
                maxBu = Bu_mag;
                maxPatch = i;
            }
        }
        
        // Check convergence
        if (maxBu < epsilon) {
            std::cout << "Converged after " << iter << " iterations\n";
            std::cout << "Final max unshot: " << maxBu << "\n";
            break;
        }
        
        if (iter % 10 == 0) {
            std::cout << "  Iteration " << iter << ": max unshot = " << maxBu 
                      << " (patch " << maxPatch << ")\n";
        }
        
        // Shoot radiosity from maxPatch to all other patches using CSR
        uint32_t p = maxPatch;
        float Bu_r_p = patches.Bu_r[p];
        float Bu_g_p = patches.Bu_g[p];
        float Bu_b_p = patches.Bu_b[p];
        
        // Iterate over non-zero form factors in row p
        uint32_t start = formFactors.rowPtr[p];
        uint32_t end = formFactors.rowPtr[p + 1];
        
        for (uint32_t k = start; k < end; ++k) {
            uint32_t j = formFactors.colIdx[k];
            float F_pj = formFactors.values[k];
            
            // Compute radiosity update: ΔB_j = ρ_j * F[p,j] * Bu_p
            float dB_r = patches.rho_r[j] * F_pj * Bu_r_p;
            float dB_g = patches.rho_g[j] * F_pj * Bu_g_p;
            float dB_b = patches.rho_b[j] * F_pj * Bu_b_p;
            
            // Update radiosity and unshot
            patches.B_r[j] += dB_r;
            patches.B_g[j] += dB_g;
            patches.B_b[j] += dB_b;
            
            patches.Bu_r[j] += dB_r;
            patches.Bu_g[j] += dB_g;
            patches.Bu_b[j] += dB_b;
        }
        
        // Clear unshot for the shooting patch
        patches.Bu_r[p] = 0.0f;
        patches.Bu_g[p] = 0.0f;
        patches.Bu_b[p] = 0.0f;
        
        // Debug logging for first iteration
        if (iter == 0) {
            std::cout << "\n  *** First iteration debug ***\n";
            std::cout << "      Shooting patch: " << p << "\n";
            std::cout << "      Non-zero form factors: " << (end - start) << "\n";
            
            // Count patches with radiosity
            uint32_t litPatches = 0;
            for (uint32_t i = 0; i < N; ++i) {
                float B = patches.B_r[i] + patches.B_g[i] + patches.B_b[i];
                if (B > 1e-6f) litPatches++;
            }
            std::cout << "      Patches with B > 0: " << litPatches << "/" << N << "\n\n";
        }
    }
    
    std::cout << "\n=== Solution Statistics ===\n";
    
    // Compute min/max/avg radiosity
    float minB = 1e10f, maxB = 0.0f, avgB = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        float B_mag = patches.B_r[i] + patches.B_g[i] + patches.B_b[i];
        minB = std::min(minB, B_mag);
        maxB = std::max(maxB, B_mag);
        avgB += B_mag;
    }
    if (N > 0) avgB /= float(N);
    
    std::cout << "Radiosity range: [" << minB << ", " << maxB << "]\n";
    std::cout << "Average radiosity: " << avgB << "\n";
}

// DEPRECATED: Old dense form factor interface (kept for compatibility)
inline void progressiveRefinement_OLD(PatchSoA& patches,
                                       const std::vector<std::vector<float>>& formFactors,
                                       uint32_t maxIterations = 100,
                                       float epsilon = 0.001f) {
    std::cout << "\nWARNING: Using deprecated dense form factor solver\n";
    std::cout << "         Please switch to CSR-based solver\n\n";
    
    // Convert dense to CSR and call new solver
    FormFactorCache::FormFactorCSR csr = FormFactorCache::denseToCSR(
        formFactors, "unknown", 0);
    
    progressiveRefinement(patches, csr, maxIterations, epsilon);
}

} // namespace Solver

