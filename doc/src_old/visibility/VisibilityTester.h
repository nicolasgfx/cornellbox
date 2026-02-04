#pragma once

#include "math/Vector3.h"
#include "geometry/IndexedMesh.h"
#include "core/Patch.h"
#include <vector>
#include <memory>
#include <iostream>
#include <random>
#include <limits>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <utility>

#ifdef USE_OPTIX
#include "visibility/OptiXContext.h"
#endif

namespace radiosity {
namespace visibility {

using math::Vector3;
using geometry::IndexedMesh;
using core::Patch;

/**
 * Visibility Tester using OptiX ray tracing
 * 
 * This class handles:
 * - OptiX context initialization
 * - Mesh upload to GPU
 * - Ray casting for visibility queries
 * - Binary visibility testing (0 = occluded, 1 = visible)
 * 
 * Week 3: OptiX integration implemented
 */
class VisibilityTester {
public:
    VisibilityTester() : initialized(false), meshPtr(nullptr) {
#ifdef USE_OPTIX
        std::cout << "VisibilityTester created (OptiX enabled)\n";
#else
        std::cout << "VisibilityTester created (stub mode - no OptiX)\n";
#endif
    }
    
    ~VisibilityTester() {
        cleanup();
    }
    
    /**
     * Initialize OptiX context and upload mesh
     */
    bool initialize(const IndexedMesh& mesh) {
        if (initialized) {
            std::cerr << "WARNING: VisibilityTester already initialized\n";
            return true;
        }
        
        std::cout << "Initializing visibility tester...\n";
        std::cout << "  Mesh: " << mesh.vertexCount() << " vertices, " 
                  << mesh.triangleCount() << " triangles\n";
        
        meshVertexCount = mesh.vertexCount();
        meshTriangleCount = mesh.triangleCount();
        meshPtr = &mesh;
        
#ifdef USE_OPTIX
        // Initialize OptiX
        optixContext = std::make_unique<OptiXContext>();
        
        if (!optixContext->initialize()) {
            std::cerr << "ERROR: Failed to initialize OptiX context\n";
            return false;
        }
        
        // Upload mesh to GPU
        const float* vertexData = reinterpret_cast<const float*>(mesh.getVertexDataPtr());
        const uint32_t* indexData = mesh.getIndexDataPtr();
        
        // DEBUG: Verify data consistency
        std::cout << "\n=== MESH DATA VERIFICATION ===\n";
        std::cout << "Vertex data pointer: " << (void*)vertexData << "\n";
        std::cout << "Index data pointer: " << (void*)indexData << "\n";
        std::cout << "First 3 vertices (XYZ):";
        for (int i = 0; i < 9 && i < (int)mesh.vertexCount() * 3; i++) {
            if (i % 3 == 0) std::cout << "\n  V" << i/3 << ": ";
            std::cout << vertexData[i] << " ";
        }
        std::cout << "\nFirst 3 triangles (indices):";
        for (int i = 0; i < 9 && i < (int)mesh.triangleCount() * 3; i++) {
            if (i % 3 == 0) std::cout << "\n  T" << i/3 << ": ";
            std::cout << indexData[i] << " ";
        }
        std::cout << "\n";
        
        if (!optixContext->uploadMesh(vertexData, mesh.vertexCount(), 
                                     indexData, mesh.triangleCount())) {
            std::cerr << "ERROR: Failed to upload mesh to GPU\n";
            return false;
        }
        
        std::cout << "✓ Visibility tester ready (OptiX ray tracing)\n";
#else
        std::cout << "✓ Visibility tester ready (stub mode)\n";
#endif
        
        initialized = true;
        return true;
    }
    
    /**
     * Test visibility between two points
     * Returns: 1.0 if visible, 0.0 if occluded
     * @param from Source point
     * @param to Target point
     * @param sourceNormal Normal at source (for hemisphere validation)
     * @param targetPatchId ID of target patch to ignore in intersection tests
     */
    float testVisibility(const Vector3& from, const Vector3& to, const Vector3& sourceNormal, int targetPatchId = -1) const {
        if (!initialized) {
            std::cerr << "ERROR: VisibilityTester not initialized\n";
            return 0.0f;
        }
        
#ifdef USE_OPTIX
        if (optixContext && optixContext->isInitialized()) {
            // Calculate ray direction and distance
            Vector3 dir = to - from;
            float distance = dir.length();
            
            static int debugCount = 0;
            if (debugCount < 3) {
                std::cout << "  [Ray #" << debugCount << "] from=(" << from.x << "," << from.y << "," << from.z << ")";
                std::cout << " to=(" << to.x << "," << to.y << "," << to.z << ")";
                std::cout << " distance=" << distance << "\n";
                debugCount++;
            }
            
            dir = dir.normalized();
            
            // Cast ray using OptiX with source normal and target patch ID
            float origin[3] = { from.x, from.y, from.z };
            float direction[3] = { dir.x, dir.y, dir.z };
            float source_normal[3] = { sourceNormal.x, sourceNormal.y, sourceNormal.z };
            
            return optixContext->traceRay(origin, direction, source_normal, distance, targetPatchId);
        }
#endif
        
        // Stub mode: assume all patches visible
        (void)from;
        (void)to;
        return 1.0f;
    }
    
    /**
     * Test visibility between two patches
     * Uses patch centers and adds small epsilon offset along normals
     */
    /**
     * Test visibility between two patches with 8-sample Monte Carlo (GPU-side)
     * Passes patch vertices to GPU which does the sampling internally
     */
    float testPatchVisibility(
        const Patch& fromPatch,
        const Vector3& from_v0, const Vector3& from_v1, const Vector3& from_v2, const Vector3& from_v3,
        const Patch& toPatch,
        const Vector3& to_v0, const Vector3& to_v1, const Vector3& to_v2, const Vector3& to_v3,
        int fromPatchId, int toPatchId) const 
    {
        (void)from_v0; (void)from_v1; (void)from_v2; (void)from_v3;
        (void)to_v0; (void)to_v1; (void)to_v2; (void)to_v3;
#ifdef USE_OPTIX
        if (optixContext && optixContext->isInitialized() && meshPtr) {
            std::vector<TriangleSample> sourceTriangles;
            std::vector<TriangleSample> targetTriangles;
            buildTriangleSamples(fromPatch, fromPatchId, sourceTriangles);
            buildTriangleSamples(toPatch, toPatchId, targetTriangles);

            if (sourceTriangles.empty() || targetTriangles.empty()) {
                return 0.0f;
            }

            float totalSourceArea = 0.0f;
            float totalTargetArea = 0.0f;
            for (const auto& tri : sourceTriangles) {
                totalSourceArea += tri.area;
            }
            for (const auto& tri : targetTriangles) {
                totalTargetArea += tri.area;
            }

            if (totalSourceArea <= std::numeric_limits<float>::epsilon() ||
                totalTargetArea <= std::numeric_limits<float>::epsilon()) {
                return 0.0f;
            }

            constexpr float AREA_SAMPLE_DENSITY = 0.15f;
            constexpr int MAX_SAMPLES = 64;
            int sampleCount = std::max(1, static_cast<int>(std::ceil(AREA_SAMPLE_DENSITY * toPatch.area)));
            sampleCount = std::min(sampleCount, MAX_SAMPLES);

            uint64_t seed = static_cast<uint64_t>(fromPatchId) * 73856093ull ^
                             static_cast<uint64_t>(toPatchId) * 19349663ull;
            std::mt19937 rng(static_cast<uint32_t>(seed));
            std::uniform_real_distribution<float> dist(0.0f, std::nextafter(1.0f, 0.0f));

            unsigned int visibleCount = 0;
            for (int s = 0; s < sampleCount; ++s) {
                float sourceSelect = dist(rng);
                float targetSelect = dist(rng);

                int sourceIndex = selectTriangleIndex(sourceTriangles, sourceSelect, totalSourceArea);
                int targetIndex = selectTriangleIndex(targetTriangles, targetSelect, totalTargetArea);
                if (sourceIndex < 0 || targetIndex < 0) {
                    continue;
                }

                auto [uA, vA] = randomBarycentric(rng, dist);
                auto [uB, vB] = randomBarycentric(rng, dist);

                float sourceUV[2] = {uA, vA};
                float targetUV[2] = {uB, vB};

                unsigned int rayResult = optixContext->traceTriangleSample(
                    sourceTriangles[sourceIndex].data,
                    targetTriangles[targetIndex].data,
                    sourceUV,
                    targetUV
                );

                visibleCount += (rayResult > 0u) ? 1u : 0u;
            }

            return sampleCount > 0 ? static_cast<float>(visibleCount) / static_cast<float>(sampleCount) : 0.0f;
        }
#endif
        return 1.0f;  // Fallback: assume visible
    }
    
    /**
     * DEPRECATED: Old single-point test (kept for compatibility with test code)
     * Test visibility between two patches
     * Uses patch centers and adds small epsilon offset along normals
     */
    float testPatchVisibility(const Patch& fromPatch, const Patch& toPatch, int fromPatchId, int toPatchId) const {
        // Offset slightly along normals to avoid self-intersection
        const float epsilon = 1.0f;
        Vector3 fromPoint = fromPatch.center + fromPatch.normal * epsilon;
        Vector3 toPoint = toPatch.center + toPatch.normal * epsilon;
        
        return testVisibility(fromPoint, toPoint, fromPatch.normal, toPatchId);
    }
    
    /**
     * Batch visibility test for a patch to all other patches
     * More efficient than individual queries
     */
    std::vector<float> testPatchVisibilityBatch(
        const Patch& fromPatch,
        int fromPatchId,
        const std::vector<Patch>& toPatches) const 
    {
        std::vector<float> results;
        results.reserve(toPatches.size());
        
        for (size_t i = 0; i < toPatches.size(); ++i) {
            results.push_back(testPatchVisibility(fromPatch, toPatches[i], fromPatchId, static_cast<int>(i)));
        }
        
        return results;
    }
    
    bool isInitialized() const { return initialized; }
    
    size_t getMeshVertexCount() const { return meshVertexCount; }
    size_t getMeshTriangleCount() const { return meshTriangleCount; }
    
private:
#ifdef USE_OPTIX
    struct TriangleSample {
        OptiXContext::TriangleData data;
        float area;
    };

    void buildTriangleSamples(const Patch& patch, int patchId, std::vector<TriangleSample>& out) const;
    int selectTriangleIndex(const std::vector<TriangleSample>& samples, float randomValue, float totalArea) const;
    static std::pair<float, float> randomBarycentric(std::mt19937& rng, std::uniform_real_distribution<float>& dist);
#endif

    void cleanup() {
        if (initialized) {
#ifdef USE_OPTIX
            optixContext.reset();
#endif
            initialized = false;
            meshPtr = nullptr;
            std::cout << "Visibility tester cleaned up\n";
        }
    }
    
    bool initialized;
    size_t meshVertexCount;
    size_t meshTriangleCount;
    
#ifdef USE_OPTIX
    std::unique_ptr<OptiXContext> optixContext;
#endif

    const IndexedMesh* meshPtr;
};

#ifdef USE_OPTIX
inline void VisibilityTester::buildTriangleSamples(const Patch& patch, int patchId, std::vector<TriangleSample>& out) const {
    out.clear();
    if (!meshPtr || patch.triangleCount <= 0) {
        return;
    }

    out.reserve(static_cast<size_t>(patch.triangleCount));
    size_t meshTriangleCount = meshPtr->triangleCount();
    for (int t = 0; t < patch.triangleCount; ++t) {
        size_t triIndex = static_cast<size_t>(patch.firstTriangleIndex + t);
        if (triIndex >= meshTriangleCount) {
            continue;
        }

        Vector3 v0, v1, v2;
        meshPtr->getTriangleVertices(triIndex, v0, v1, v2);
        float area = meshPtr->getTriangleArea(triIndex);
        if (area <= std::numeric_limits<float>::epsilon()) {
            continue;
        }

        TriangleSample sample = {};
        sample.data.patch_id = patchId;
        for (int i = 0; i < 3; ++i) {
            sample.data.v0[i] = v0[i];
            sample.data.v1[i] = v1[i];
            sample.data.v2[i] = v2[i];
        }
        sample.area = area;
        out.push_back(sample);
    }
}

inline int VisibilityTester::selectTriangleIndex(const std::vector<TriangleSample>& samples, float randomValue, float totalArea) const {
    if (samples.empty() || totalArea <= std::numeric_limits<float>::epsilon()) {
        return -1;
    }

    float target = randomValue * totalArea;
    float cumulative = 0.0f;
    for (size_t i = 0; i < samples.size(); ++i) {
        cumulative += samples[i].area;
        if (target <= cumulative) {
            return static_cast<int>(i);
        }
    }
    return static_cast<int>(samples.size() - 1);
}

inline std::pair<float, float> VisibilityTester::randomBarycentric(std::mt19937& rng, std::uniform_real_distribution<float>& dist) {
    float u = dist(rng);
    float v = dist(rng);
    return {u, v};
}
#endif

}  // namespace visibility
}  // namespace radiosity
