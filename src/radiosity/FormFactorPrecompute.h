#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>
#include "../mesh/MeshData.h"
#include "../gpu/OptiXContext.h"
#include "FormFactorCache.h"

namespace FormFactorPrecompute {

// Configuration for hemisphere form factor computation
struct HemisphereConfig {
    std::string profile;      // "low", "medium", "high"
    uint32_t originSamples;   // Origins per patch (1 or 4)
    uint32_t dirSamples;      // Directions per origin (256-2048)
    bool useCache;            // Load from cache if available
    bool forceRecompute;      // Ignore cache, always recompute
    
    HemisphereConfig() 
        : profile("low"), originSamples(1), dirSamples(256), 
          useCache(true), forceRecompute(false) {}
    
    uint32_t totalRays() const { return originSamples * dirSamples; }
};

// Get recommended sampling parameters for each profile
inline HemisphereConfig getProfileConfig(const std::string& profile) {
    HemisphereConfig config;
    config.profile = profile;
    
    if (profile == "low") {
        config.originSamples = 1;
        config.dirSamples = 256;
    } else if (profile == "medium") {
        config.originSamples = 1;
        config.dirSamples = 512;
    } else if (profile == "high") {
        config.originSamples = 4;
        config.dirSamples = 512;
    } else {
        std::cerr << "Unknown profile: " << profile << ", using 'low'\n";
        config.originSamples = 1;
        config.dirSamples = 256;
    }
    
    return config;
}

// Main precompute function
// Returns CSR form factors (from cache or computed)
inline FormFactorCache::FormFactorCSR computeFormFactors(
    const MeshData& mesh,
    const PatchSoA& patches,
    const HemisphereConfig& config,
    OptiXContext::Context& optixContext) {
    
    const uint32_t N = static_cast<uint32_t>(patches.area.size());
    const uint32_t R = config.totalRays();
    
    std::cout << "\n=== Hemisphere Form Factor Precompute ===\n";
    std::cout << "Profile: " << config.profile << "\n";
    std::cout << "Patches: " << N << "\n";
    std::cout << "Origin samples: " << config.originSamples << "\n";
    std::cout << "Direction samples: " << config.dirSamples << "\n";
    std::cout << "Total rays/patch: " << R << "\n";
    std::cout << "Total rays: " << (uint64_t(N) * R) << "\n";
    
    // Check cache first
    std::string cacheFile = FormFactorCache::getCacheFilename(
        config.profile, config.originSamples, config.dirSamples);
    
    if (config.useCache && !config.forceRecompute) {
        FormFactorCache::FormFactorCSR cached;
        if (FormFactorCache::loadCSR(cacheFile, cached)) {
            if (cached.numPatches == N) {
                std::cout << "✓ Using cached form factors\n";
                FormFactorCache::validateCSR(cached);
                return cached;
            } else {
                std::cout << "Cache patch count mismatch (expected " << N 
                          << ", got " << cached.numPatches << "), recomputing\n";
            }
        }
    }
    
    // Ensure output/cache directory exists
    std::filesystem::create_directories("output/cache");
    
    // Compute scene epsilon based on scene bounds
    float sceneSize = 0.0f;
    for (const auto& v : mesh.vertices) {
        sceneSize = std::max(sceneSize, std::abs(v.x));
        sceneSize = std::max(sceneSize, std::abs(v.y));
        sceneSize = std::max(sceneSize, std::abs(v.z));
    }
    float sceneEpsilon = sceneSize * 1e-4f;
    
    std::cout << "\nScene epsilon: " << sceneEpsilon << "\n";
    std::cout << "Computing form factors via OptiX...\n";
    
    // For LOW profile, use dense storage (feasible up to ~1000 triangles)
    // For MEDIUM/HIGH, we'd need sparse accumulation (future enhancement)
    bool useDense = (config.profile == "low" || N < 2000);
    
    if (!useDense) {
        std::cerr << "ERROR: Sparse accumulation not yet implemented for " << config.profile << "\n";
        std::cerr << "       Please use --low profile for now\n";
        std::exit(1);
    }
    
    // Allocate dense form factor matrix on GPU
    size_t matrixSize = size_t(N) * N;
    float* d_formFactors = nullptr;
    CUDA_CHECK(cudaMalloc(&d_formFactors, matrixSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_formFactors, 0, matrixSize * sizeof(float)));
    
    std::cout << "Allocated dense matrix: " << (matrixSize * sizeof(float) / (1024*1024)) << " MB\n";
    
    // Build OptiX pipeline for hemisphere casting
    // Note: This requires extending OptiXContext with hemisphere pipeline
    // For now, we'll outline the structure - full integration needed
    
    std::cout << "\n TODO: OptiX hemisphere pipeline launch\n";
    std::cout << " - Load hemisphere kernels (HemisphereFormFactorKernels.cu)\n";
    std::cout << " - Configure launch params with patches and sampling config\n";
    std::cout << " - Launch OptiX with dims [N, R]\n";
    std::cout << " - Download results from d_formFactors\n";
    
    // Placeholder: Download results (in real impl, this happens after OptiX launch)
    std::vector<float> h_formFactors(matrixSize);
    CUDA_CHECK(cudaMemcpy(h_formFactors.data(), d_formFactors, 
                          matrixSize * sizeof(float), cudaMemcpyDeviceToHost));
    
    CUDA_CHECK(cudaFree(d_formFactors));
    
    // Convert to CSR
    std::cout << "\nConverting to CSR format...\n";
    FormFactorCache::FormFactorCSR csr = FormFactorCache::denseFlatToCSR(
        h_formFactors.data(), N, config.profile, R);
    
    std::cout << "Non-zero entries: " << csr.nonZeros() 
              << " (" << (100.0 * csr.nonZeros() / matrixSize) << "% density)\n";
    
    // Validate
    FormFactorCache::validateCSR(csr);
    
    // Save to cache
    FormFactorCache::saveCSR(cacheFile, csr);
    
    return csr;
}

// Debug export: visualize form factor distribution as "exposure" heatmap
inline void exportFormFactorDebug(const PatchSoA& patches,
                                   const FormFactorCache::FormFactorCSR& formFactors,
                                   const std::string& outputPath) {
    std::cout << "\n=== Exporting Form Factor Debug Visualization ===\n";
    
    // Compute "exposure" = sum_j F[i,j] for each patch
    auto rowSums = formFactors.computeRowSums();
    
    // Map to [0, 1] for visualization
    float minSum = *std::min_element(rowSums.begin(), rowSums.end());
    float maxSum = *std::max_element(rowSums.begin(), rowSums.end());
    
    std::cout << "Exposure range: [" << minSum << ", " << maxSum << "]\n";
    
    // Create debug PatchSoA with exposure as color
    PatchSoA debugPatches = patches;
    for (uint32_t i = 0; i < rowSums.size(); ++i) {
        float exposure = (rowSums[i] - minSum) / (maxSum - minSum + 1e-6f);
        
        // Heatmap: blue (low) -> cyan -> green -> yellow -> red (high)
        if (exposure < 0.25f) {
            float t = exposure / 0.25f;
            debugPatches.B_r[i] = 0.0f;
            debugPatches.B_g[i] = t;
            debugPatches.B_b[i] = 1.0f;
        } else if (exposure < 0.5f) {
            float t = (exposure - 0.25f) / 0.25f;
            debugPatches.B_r[i] = 0.0f;
            debugPatches.B_g[i] = 1.0f;
            debugPatches.B_b[i] = 1.0f - t;
        } else if (exposure < 0.75f) {
            float t = (exposure - 0.5f) / 0.25f;
            debugPatches.B_r[i] = t;
            debugPatches.B_g[i] = 1.0f;
            debugPatches.B_b[i] = 0.0f;
        } else {
            float t = (exposure - 0.75f) / 0.25f;
            debugPatches.B_r[i] = 1.0f;
            debugPatches.B_g[i] = 1.0f - t;
            debugPatches.B_b[i] = 0.0f;
        }
    }
    
    // Export as OBJ (requires ObjExporter - assuming it exists)
    std::cout << "Debug export: " << outputPath << "\n";
    std::cout << "(Implementation requires ObjExporter module)\n";
}

} // namespace FormFactorPrecompute
