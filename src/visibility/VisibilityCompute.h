#pragma once

#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include "../mesh/MeshData.h"
#include "../gpu/OptiXContext.h"
#include "VisibilityCache.h"

namespace VisibilityCompute {

// High-level visibility computation
// Returns visibility matrix and geometric kernel matrix (both upper triangle only, float format)
inline std::pair<std::vector<float>, std::vector<float>> compute(
                                   const Mesh& mesh,
                                   const std::string& profile,
                                   uint32_t samplesPerPair,
                                   bool useCache = true) {
    uint32_t numTriangles = static_cast<uint32_t>(mesh.indices.size());
    
    // Cache temporarily disabled for geometric kernel feature
    // TODO: Add geometric kernel to cache format
    std::cout << "Cache disabled pending geometric kernel format update, computing...\n";
    
    // Ensure cache directory exists
    std::filesystem::path cacheDir("output/cache");
    if (!std::filesystem::exists(cacheDir)) {
        std::cout << "Creating cache directory: " << cacheDir << "\n";
        std::filesystem::create_directories(cacheDir);
    }
    
    // Compute visibility using OptiX
    std::cout << "\nComputing visibility with OptiX...\n";
    std::cout << "  Triangles: " << numTriangles << "\n";
    std::cout << "  Samples per pair: " << samplesPerPair << "\n";
    std::cout << "  Triangle pairs: " << (size_t(numTriangles) * (numTriangles - 1)) / 2 << "\n";
    
    OptiXContext::Context optixCtx;
    
    // Load PTX module
    std::string ptxPath = "build/optix_kernels.ptx";
    if (!std::filesystem::exists(ptxPath)) {
        std::cerr << "PTX file not found: " << ptxPath << "\n";
        std::cerr << "Make sure to compile the OptiX kernels first.\n";
        std::exit(1);
    }
    
    optixCtx.createModule(ptxPath);
    optixCtx.createProgramGroups();
    optixCtx.createPipeline();
    optixCtx.createSBT();
    
    // Build GAS
    std::cout << "Building OptiX acceleration structure...\n";
    optixCtx.buildGAS(mesh);
    
    // Compute visibility and geometric kernels
    std::cout << "Ray tracing visibility and geometric kernels...\n";
    std::vector<float> visibility;
    std::vector<float> geometricKernel;
    optixCtx.computeVisibility(mesh, visibility, geometricKernel, samplesPerPair);

    asdasdasdad
    
    std::cout << "Visibility computation complete.\n";
    
    // Cache saving disabled for now (TODO: add geometric kernel to cache format)
    
    return {visibility, geometricKernel};
}

// Compute AO-like visibility score per triangle
// visScore[i] = average visibility from triangle i to all other triangles
inline std::vector<float> computeVisibilityScores(const std::vector<float>& visibility,
                                                   uint32_t numTriangles) {
    std::vector<float> visScores(numTriangles, 0.0f);
    
    for (uint32_t i = 0; i < numTriangles; ++i) {
        float totalVis = 0.0f;
        
        for (uint32_t j = 0; j < numTriangles; ++j) {
            if (i == j) continue;
            
            // Get visibility from cache (symmetric matrix stored as upper triangle)
            float vis;
            if (i < j) {
                size_t idx = (size_t(i) * (2 * numTriangles - i - 1)) / 2 + (j - i - 1);
                vis = visibility[idx];
            } else {
                size_t idx = (size_t(j) * (2 * numTriangles - j - 1)) / 2 + (i - j - 1);
                vis = visibility[idx];
            }
            totalVis += vis;
        }
        
        visScores[i] = totalVis / float(numTriangles - 1);
    }
    
    return visScores;
}

// Compute occlusion (inverse of visibility)
inline std::vector<float> computeOcclusionScores(const std::vector<float>& visibility,
                                                  uint32_t numTriangles) {
    std::vector<float> visScores = computeVisibilityScores(visibility, numTriangles);
    
    std::vector<float> occScores(numTriangles);
    for (uint32_t i = 0; i < numTriangles; ++i) {
        occScores[i] = 1.0f - visScores[i];
    }
    
    return occScores;
}

} // namespace VisibilityCompute
