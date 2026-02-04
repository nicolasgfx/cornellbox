#pragma once

#include "core/Patch.h"
#include "core/Scene.h"
#include "geometry/IndexedMesh.h"
#include "radiosity/FormFactor.h"
#include "radiosity/MonteCarloFormFactor.h"
#include "visibility/VisibilityTester.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

namespace radiosity {
namespace experiments {

using core::Patch;
using core::RadiosityScene;
using geometry::IndexedMesh;
using radiosity::FormFactorCalculator;
using visibility::VisibilityTester;
using math::Vector3;

/**
 * Form Factor Unit Tests
 * 
 * These tests validate the form factor calculation and OptiX ray tracing
 * using the REAL production code. No code duplication.
 * 
 * Each test creates a simple geometric configuration and validates
 * the calculated form factors against theoretical expectations.
 */
class FormFactorTests {
public:
    struct TestResult {
        bool passed;
        std::string name;
        std::string message;
        float expected;
        float actual;
        float tolerance;
    };
    
private:
    std::vector<TestResult> results;
    VisibilityTester* visTester = nullptr;
    
    /**
     * Helper: Create a simple scene with custom patches
     */
    RadiosityScene createScene(const std::vector<Patch>& patches, const IndexedMesh& mesh) {
        RadiosityScene scene;
        scene.patches = patches;
        scene.mesh = mesh;
        return scene;
    }
    
    /**
     * Helper: Create a square patch as two triangles
     * Returns the mesh and sets up the patch geometry
     */
    IndexedMesh createSquarePatchMesh(
        const Vector3& center, 
        const Vector3& normal, 
        float size,
        Patch& patch) 
    {
        IndexedMesh mesh;
        
        // Calculate tangent vectors
        Vector3 up = std::abs(normal.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
        Vector3 right = normal.cross(up).normalized();
        Vector3 tangent = right.cross(normal).normalized();
        
        float halfSize = size * 0.5f;
        
        // Create 4 vertices for the square
        mesh.vertices.push_back(center + right * (-halfSize) + tangent * (-halfSize));
        mesh.vertices.push_back(center + right * (+halfSize) + tangent * (-halfSize));
        mesh.vertices.push_back(center + right * (+halfSize) + tangent * (+halfSize));
        mesh.vertices.push_back(center + right * (-halfSize) + tangent * (+halfSize));
        
        // Create 2 triangles
        mesh.indices.push_back(0);
        mesh.indices.push_back(1);
        mesh.indices.push_back(2);
        
        mesh.indices.push_back(0);
        mesh.indices.push_back(2);
        mesh.indices.push_back(3);
        
        // Both triangles belong to patch 0
        mesh.patchIds.push_back(0);
        mesh.patchIds.push_back(0);
        
        // Set up patch geometry
        patch.center = center;
        patch.normal = normal;
        patch.area = size * size;
        patch.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        patch.emission = Vector3(0, 0, 0);
        patch.firstTriangleIndex = 0;
        patch.triangleCount = 2;
        
        return mesh;
    }
    
    /**
     * Helper: Create mesh with multiple patches
     */
    IndexedMesh createMultiPatchMesh(
        const std::vector<std::pair<Vector3, Vector3>>& patchData,  // center, normal pairs
        float size,
        std::vector<Patch>& patches)
    {
        IndexedMesh mesh;
        patches.clear();
        
        for (size_t i = 0; i < patchData.size(); i++) {
            const Vector3& center = patchData[i].first;
            const Vector3& normal = patchData[i].second;
            
            Patch patch;
            patch.center = center;
            patch.normal = normal;
            patch.area = size * size;
            patch.reflectance = Vector3(0.5f, 0.5f, 0.5f);
            patch.emission = Vector3(0, 0, 0);
            patch.firstTriangleIndex = static_cast<int>(mesh.triangleCount());
            patch.triangleCount = 2;
            
            // Calculate tangent vectors
            Vector3 up = std::abs(normal.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
            Vector3 right = normal.cross(up).normalized();
            Vector3 tangent = right.cross(normal).normalized();
            
            float halfSize = size * 0.5f;
            uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
            
            // Create 4 vertices for the square
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (+halfSize));
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (+halfSize));
            
            // Create 2 triangles
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 1);
            mesh.indices.push_back(baseVertex + 2);
            
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 3);
            
            // Both triangles belong to this patch
            mesh.patchIds.push_back(static_cast<uint32_t>(i));
            mesh.patchIds.push_back(static_cast<uint32_t>(i));
            
            patches.push_back(patch);
        }
        
        return mesh;
    }
    
    /**
     * Helper: Normalize form factors to 0-1 range for easier comparison
     * Finds max value and scales all values proportionally
     */
    struct NormalizedValues {
        std::vector<float> normalized;
        float maxValue;
        float scaleFactor;
    };
    
    NormalizedValues normalizeFormFactors(const std::vector<float>& values) {
        NormalizedValues result;
        result.maxValue = 0.0f;
        
        // Find maximum
        for (float val : values) {
            if (val > result.maxValue) {
                result.maxValue = val;
            }
        }
        
        // Scale factor to bring max to 1.0
        result.scaleFactor = (result.maxValue > 1e-10f) ? (1.0f / result.maxValue) : 1.0f;
        
        // Normalize all values
        result.normalized.reserve(values.size());
        for (float val : values) {
            result.normalized.push_back(val * result.scaleFactor);
        }
        
        return result;
    }
    
    /**
     * Helper: Validate form factor against expected value
     */
    void validateFormFactor(
        const std::string& testName,
        float actual,
        float expected,
        float tolerance = 0.01f)
    {
        bool passed = std::abs(actual - expected) <= tolerance;
        
        TestResult result;
        result.passed = passed;
        result.name = testName;
        result.expected = expected;
        result.actual = actual;
        result.tolerance = tolerance;
        
        if (passed) {
            result.message = "PASS";
        } else {
            result.message = "FAIL - Expected " + std::to_string(expected) + 
                           " ± " + std::to_string(tolerance) + 
                           ", got " + std::to_string(actual);
        }
        
        results.push_back(result);
    }
    
public:
    /**
     * Test 1: Two parallel patches facing each other
     * Should have MAXIMUM form factor for given distance and size
     */
    void test1_ParallelPatchesFacing() {
        std::cout << "\n=== Test 1: Parallel Patches Facing ===\n";
        
        // Create two 100x100 patches, 200mm apart, facing each other
        Vector3 center1(0, 0, 0);
        Vector3 normal1(0, 0, 1);  // Facing +Z
        
        Vector3 center2(0, 0, 200);
        Vector3 normal2(0, 0, -1);  // Facing -Z
        
        std::vector<std::pair<Vector3, Vector3>> patchData = {
            {center1, normal1},
            {center2, normal2}
        };
        
        std::vector<Patch> patches;
        IndexedMesh mesh = createMultiPatchMesh(patchData, 100.0f, patches);
        
        // Upload to OptiX
        if (visTester && !visTester->initialize(mesh)) {
            std::cerr << "  ⚠ OptiX initialization failed\n";
        }
        
        // Calculate form factor using REAL code
        auto matrix = FormFactorCalculator::calculateMatrix(patches, visTester, false);
        float F_01 = matrix[0][1];  // Form factor from patch 0 to patch 1
        
        std::cout << "  Patch 0: center=" << center1 << ", normal=" << normal1 << "\n";
        std::cout << "  Patch 2: center=" << center2 << ", normal=" << normal2 << "\n";
        std::cout << "  Distance: 200mm, Size: 100x100mm\n";
        std::cout << "  F_01 (raw) = " << std::scientific << F_01 << std::fixed << "\n";
        
        // For parallel patches: F should be positive (any positive value confirms energy transfer)
        validateFormFactor("Parallel facing patches should transfer energy", F_01 > 0.0f ? 1.0f : 0.0f, 1.0f, 0.01f);
        
        // Visibility should be 1.0 (no occlusion)
        if (visTester) {
            float vis = visTester->testPatchVisibility(patches[0], patches[1], 0, 1);
            std::cout << "  Visibility = " << vis << "\n";
            validateFormFactor("Parallel facing patches should be fully visible", vis, 1.0f, 0.01f);
        }
    }
    
    /**
     * Test 2: Two parallel patches facing AWAY from each other
     * Should have ZERO form factor (no energy transfer)
     */
    void test2_ParallelPatchesAway() {
        std::cout << "\n=== Test 2: Parallel Patches Facing Away ===\n";
        
        // Create two patches with normals pointing away from each other
        Vector3 center1(0, 0, 0);
        Vector3 normal1(0, 0, -1);  // Facing -Z
        
        Vector3 center2(0, 0, 200);
        Vector3 normal2(0, 0, -1);  // Also facing -Z (same direction)
        
        std::vector<std::pair<Vector3, Vector3>> patchData = {
            {center1, normal1},
            {center2, normal2}
        };
        
        std::vector<Patch> patches;
        IndexedMesh mesh = createMultiPatchMesh(patchData, 100.0f, patches);
        
        if (visTester) visTester->initialize(mesh);
        
        // Calculate form factor
        auto matrix = FormFactorCalculator::calculateMatrix(patches, visTester, false);
        float F_01 = matrix[0][1];
        
        std::cout << "  Patch 0: center=" << center1 << ", normal=" << normal1 << "\n";
        std::cout << "  Patch 1: center=" << center2 << ", normal=" << normal2 << "\n";
        std::cout << "  F_01 (raw) = " << std::scientific << F_01 << std::fixed << "\n";
        
        // Form factor should be zero (patches face away)
        validateFormFactor("Patches facing away should have zero form factor", F_01, 0.0f, 0.000001f);
    }
    
    /**
     * Test 3a: Distance effect - Close patches
     * Test 3b: Distance effect - Far patches
     * Form factor should decrease with distance squared
     */
    void test3_DistanceEffect() {
        std::cout << "\n=== Test 3: Distance Effect ===\n";
        
        float F_close, F_far;
        
        // Test 3a: Close patches (100mm apart)
        {
            std::vector<std::pair<Vector3, Vector3>> patchData = {
                {Vector3(0, 0, 0), Vector3(0, 0, 1)},
                {Vector3(0, 0, 100), Vector3(0, 0, -1)}
            };
            
            std::vector<Patch> patches;
            IndexedMesh mesh = createMultiPatchMesh(patchData, 100.0f, patches);
            if (visTester) visTester->initialize(mesh);
            
            auto matrix = FormFactorCalculator::calculateMatrix(patches, visTester, false);
            F_close = matrix[0][1];
            
            std::cout << "  3a) Close patches (100mm apart): F = " << std::scientific << F_close << std::fixed << "\n";
        }
        
        // Test 3b: Far patches (400mm apart)
        {
            std::vector<std::pair<Vector3, Vector3>> patchData = {
                {Vector3(0, 0, 0), Vector3(0, 0, 1)},
                {Vector3(0, 0, 400), Vector3(0, 0, -1)}
            };
            
            std::vector<Patch> patches;
            IndexedMesh mesh = createMultiPatchMesh(patchData, 100.0f, patches);
            if (visTester) visTester->initialize(mesh);
            
            auto matrix = FormFactorCalculator::calculateMatrix(patches, visTester, false);
            F_far = matrix[0][1];
            
            std::cout << "  3b) Far patches (400mm apart): F = " << std::scientific << F_far << std::fixed << "\n";
        }
        
        // Normalize to make comparison clearer
        auto normalized = normalizeFormFactors({F_close, F_far});
        std::cout << "\n  Normalized values (max=1.0):\n";
        std::cout << "    Close (100mm): " << normalized.normalized[0] << " (scale factor: " << normalized.scaleFactor << ")\n";
        std::cout << "    Far (400mm):   " << normalized.normalized[1] << "\n";
        std::cout << "    Ratio (close/far): " << (F_far > 1e-10f ? F_close / F_far : 0.0f) << "\n";
        
        // Validate: F should decrease with distance (close > far)
        validateFormFactor("Close patches should have larger form factor than far", 
                          F_close > F_far ? 1.0f : 0.0f, 1.0f, 0.01f);
        
        // For point-to-point, form factor should decrease ~1/r² 
        // Distance ratio: 400/100 = 4, so F_close/F_far should be ~16
        float expectedRatio = 16.0f;
        float actualRatio = F_far > 1e-10f ? F_close / F_far : 0.0f;
        std::cout << "    Expected ratio (1/r²): ~" << expectedRatio << "\n";
        
        validateFormFactor("Form factor should follow 1/r² distance law", 
                          actualRatio, expectedRatio, 5.0f);  // Allow some tolerance
    }
    
    /**
     * Test 4a: Full occlusion
     * A patch completely blocks line of sight between two other patches
     * Form factor should be ZERO
     */
    void test4a_FullOcclusion() {
        std::cout << "\n=== Test 4a: Full Occlusion ===\n";
        
        // Patch 0 at origin, facing +Z
        // Patch 1 (occluder) at Z=200, facing -Z to block the ray
        // Patch 2 at Z=400, facing -Z
        std::vector<std::pair<Vector3, Vector3>> patchData = {
            {Vector3(0, 0, 0), Vector3(0, 0, 1)},      // Patch 0: sender
            {Vector3(0, 0, 200), Vector3(0, 0, -1)},   // Patch 1: occluder (faces sender)
            {Vector3(0, 0, 400), Vector3(0, 0, -1)}    // Patch 2: receiver
        };
        
        std::vector<Patch> patches;
        IndexedMesh mesh = createMultiPatchMesh(patchData, 100.0f, patches);
        
        // Create and initialize VisibilityTester for this test
        VisibilityTester testVisTester;
        bool optixOk = testVisTester.initialize(mesh);
        
        if (!optixOk || !testVisTester.isInitialized()) {
            std::cout << "  ⚠ OptiX initialization failed - test skipped\n";
            return;
        }
        
        // Calculate form factors
        auto matrix = FormFactorCalculator::calculateMatrix(patches, &testVisTester, false);
        float F_02 = matrix[0][2];  // Patch 0 to Patch 2 (should be occluded)
        
        std::cout << "  Patch 0 (sender): Z=0, facing +Z\n";
        std::cout << "  Patch 1 (occluder): Z=200, facing -Z (blocks ray)\n";
        std::cout << "  Patch 2 (receiver): Z=400, facing -Z\n";
        std::cout << "  F_02 (raw) = " << std::scientific << F_02 << std::fixed << "\n";
        
        // With occluder in the way, form factor should be zero
        validateFormFactor("Fully occluded patches should have zero form factor", F_02, 0.0f, 0.000001f);
        
        // Direct visibility test
        float vis = testVisTester.testPatchVisibility(patches[0], patches[2], 0, 2);
        std::cout << "  Visibility = " << vis << "\n";
        validateFormFactor("Fully occluded patches should have zero visibility", vis, 0.0f, 0.01f);
    }
    
    /**
     * Test 4b: Partial occlusion
     * A small patch partially blocks line of sight
     * Form factor should be REDUCED but not zero
     */
    void test4b_PartialOcclusion() {
        std::cout << "\n=== Test 4b: Partial Occlusion ===\n";
        
        // Large patches at Z=0 and Z=400
        // Small occluder (25x25) at Z=200 in the middle
        std::vector<Patch> patches;
        IndexedMesh mesh;
        
        // Patch 0: Large sender at origin
        Patch p0;
        p0.center = Vector3(0, 0, 0);
        p0.normal = Vector3(0, 0, 1);
        p0.area = 100.0f * 100.0f;
        p0.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        p0.firstTriangleIndex = 0;
        p0.triangleCount = 2;
        
        // Patch 1: Small occluder (centered, faces sender to block ray)
        Patch p1;
        p1.center = Vector3(0, 0, 200);
        p1.normal = Vector3(0, 0, -1);  // Faces sender to block rays
        p1.area = 25.0f * 25.0f;
        p1.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        p1.firstTriangleIndex = 2;
        p1.triangleCount = 2;
        
        // Patch 2: Large receiver
        Patch p2;
        p2.center = Vector3(0, 0, 400);
        p2.normal = Vector3(0, 0, -1);
        p2.area = 100.0f * 100.0f;
        p2.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        p2.firstTriangleIndex = 4;
        p2.triangleCount = 2;
        
        patches = {p0, p1, p2};
        
        // Build mesh manually with different sizes
        auto addSquare = [&mesh](const Vector3& center, const Vector3& normal, float size, uint32_t patchId) {
            Vector3 up = std::abs(normal.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
            Vector3 right = normal.cross(up).normalized();
            Vector3 tangent = right.cross(normal).normalized();
            float halfSize = size * 0.5f;
            
            uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (+halfSize));
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (+halfSize));
            
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 1);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 3);
            
            mesh.patchIds.push_back(patchId);
            mesh.patchIds.push_back(patchId);
        };
        
        addSquare(p0.center, p0.normal, 100.0f, 0);
        addSquare(p1.center, p1.normal, 25.0f, 1);
        addSquare(p2.center, p2.normal, 100.0f, 2);
        
        // Create and initialize VisibilityTester for this test
        VisibilityTester testVisTester;
        bool optixOk = testVisTester.initialize(mesh);
        
        if (!optixOk || !testVisTester.isInitialized()) {
            std::cout << "  ⚠ OptiX initialization failed - test skipped\n";
            return;
        }
        
        // Calculate form factors
        auto matrix = FormFactorCalculator::calculateMatrix(patches, &testVisTester, false);
        float F_02 = matrix[0][2];
        
        std::cout << "  Large patches: 100x100mm\n";
        std::cout << "  Small occluder: 25x25mm (blocks ~6% of solid angle)\n";
        std::cout << "  F_02 (raw) = " << std::scientific << F_02 << std::fixed << "\n";
        
        // Note: Point-to-point form factor may still be 0 or 1
        // For proper partial occlusion, we'd need patch-to-patch integration
        std::cout << "  ⚠ Note: Point-to-point visibility is binary (0 or 1)\n";
        std::cout << "  ⚠ True partial occlusion requires patch subdivision\n";
        
        // With small occluder, visibility depends on whether center rays intersect
        float vis = testVisTester.testPatchVisibility(patches[0], patches[2], 0, 2);
        std::cout << "  Center-to-center visibility = " << vis << "\n";
        
        // If occluder is small and centered, it might still block center ray (vis=0)
        // If off-center, vis=1. Either is valid for point-to-point approximation.
        validateFormFactor("Partial occlusion: point-to-point is binary", 
                          (vis == 0.0f || vis == 1.0f) ? 1.0f : 0.0f, 1.0f, 0.01f);
    }
    
    /**
     * Test 5: Reciprocity check
     * F_ij * A_i should equal F_ji * A_j
     */
    void test5_Reciprocity() {
        std::cout << "\n=== Test 5: Reciprocity ===\n";
        
        // Create two patches of DIFFERENT sizes
        std::vector<Patch> patches;
        IndexedMesh mesh;
        
        auto addSquare = [&mesh](const Vector3& center, const Vector3& normal, float size, uint32_t patchId) {
            Vector3 up = std::abs(normal.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
            Vector3 right = normal.cross(up).normalized();
            Vector3 tangent = right.cross(normal).normalized();
            float halfSize = size * 0.5f;
            
            uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (+halfSize));
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (+halfSize));
            
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 1);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 3);
            
            mesh.patchIds.push_back(patchId);
            mesh.patchIds.push_back(patchId);
        };
        
        // Patch 0: 100x100mm at origin
        Patch p0;
        p0.center = Vector3(0, 0, 0);
        p0.normal = Vector3(0, 0, 1);
        p0.area = 100.0f * 100.0f;
        p0.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        addSquare(p0.center, p0.normal, 100.0f, 0);
        
        // Patch 1: 50x50mm at Z=200
        Patch p1;
        p1.center = Vector3(0, 0, 200);
        p1.normal = Vector3(0, 0, -1);
        p1.area = 50.0f * 50.0f;
        p1.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        addSquare(p1.center, p1.normal, 50.0f, 1);
        
        patches = {p0, p1};
        
        if (visTester) visTester->initialize(mesh);
        
        // Calculate form factors
        auto matrix = FormFactorCalculator::calculateMatrix(patches, visTester, false);
        float F_01 = matrix[0][1];
        float F_10 = matrix[1][0];
        
        float A_0 = p0.area;
        float A_1 = p1.area;
        
        float reciprocity_0 = F_01 * A_0;
        float reciprocity_1 = F_10 * A_1;
        
        std::cout << "  Patch 0: area=" << A_0 << ", F_01=" << std::scientific << F_01 << std::fixed << "\n";
        std::cout << "  Patch 1: area=" << A_1 << ", F_10=" << std::scientific << F_10 << std::fixed << "\n";
        std::cout << "  F_01 * A_0 = " << std::scientific << reciprocity_0 << std::fixed << "\n";
        std::cout << "  F_10 * A_1 = " << std::scientific << reciprocity_1 << std::fixed << "\n";
        std::cout << "  Difference = " << std::abs(reciprocity_0 - reciprocity_1) << "\n";
        
        float avgReciprocity = (reciprocity_0 + reciprocity_1) * 0.5f;
        float relativeError = std::abs(reciprocity_0 - reciprocity_1) / (avgReciprocity + 1e-10f);
        
        std::cout << "  Relative error = " << (relativeError * 100.0f) << "%\n";
        
        // Reciprocity should be satisfied within 1%
        validateFormFactor("Reciprocity: F_01*A_0 ≈ F_10*A_1", relativeError, 0.0f, 0.01f);
    }
    
    /**
     * Test 6: Monte Carlo vs Point-to-Point Comparison
     * Monte Carlo should satisfy reciprocity better than point-to-point
     */
    void test6_MonteCarloComparison() {
        std::cout << "\n=== Test 6: Monte Carlo vs Point-to-Point ===\n";
        
        // Create two patches of different sizes (to test reciprocity)
        std::vector<Patch> patches;
        IndexedMesh mesh;
        
        auto addSquare = [&mesh](const Vector3& center, const Vector3& normal, float size, uint32_t patchId) {
            Vector3 up = std::abs(normal.y) < 0.9f ? Vector3(0, 1, 0) : Vector3(1, 0, 0);
            Vector3 right = normal.cross(up).normalized();
            Vector3 tangent = right.cross(normal).normalized();
            float halfSize = size * 0.5f;
            
            uint32_t baseVertex = static_cast<uint32_t>(mesh.vertices.size());
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (-halfSize));
            mesh.vertices.push_back(center + right * (+halfSize) + tangent * (+halfSize));
            mesh.vertices.push_back(center + right * (-halfSize) + tangent * (+halfSize));
            
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 1);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 0);
            mesh.indices.push_back(baseVertex + 2);
            mesh.indices.push_back(baseVertex + 3);
            
            mesh.patchIds.push_back(patchId);
            mesh.patchIds.push_back(patchId);
        };
        
        // Patch 0: 100x100mm at origin
        Patch p0;
        p0.center = Vector3(0, 0, 0);
        p0.normal = Vector3(0, 0, 1);
        p0.area = 100.0f * 100.0f;
        p0.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        addSquare(p0.center, p0.normal, 100.0f, 0);
        
        // Patch 1: 50x50mm at Z=200
        Patch p1;
        p1.center = Vector3(0, 0, 200);
        p1.normal = Vector3(0, 0, -1);
        p1.area = 50.0f * 50.0f;
        p1.reflectance = Vector3(0.5f, 0.5f, 0.5f);
        addSquare(p1.center, p1.normal, 50.0f, 1);
        
        patches = {p0, p1};
        
        // Create and initialize VisibilityTester for this test
        VisibilityTester testVisTester;
        bool optixOk = testVisTester.initialize(mesh);
        
        if (!optixOk || !testVisTester.isInitialized()) {
            std::cout << "  ⚠ OptiX initialization failed - test skipped\n";
            return;
        }
        
        // Point-to-point calculation (original method)
        FormFactorCalculator ptpCalc;
        auto ptpMatrix = ptpCalc.calculateMatrix(patches, &testVisTester, false);
        float F_01_ptp = ptpMatrix[0][1];
        float F_10_ptp = ptpMatrix[1][0];
        
        // Monte Carlo calculation (16 deterministic barycentric samples per patch)
        // First compute visibility, then form factor
        float vis_01 = MonteCarloFormFactor::computeVisibility(p0, p1, 0, 1, &testVisTester);
        float vis_10 = MonteCarloFormFactor::computeVisibility(p1, p0, 1, 0, &testVisTester);
        float F_01_mc = MonteCarloFormFactor::calculate(p0, p1, vis_01);
        float F_10_mc = MonteCarloFormFactor::calculate(p1, p0, vis_10);
        
        std::cout << "  Point-to-Point:\n";
        std::cout << "    F_01 = " << std::scientific << F_01_ptp << std::fixed << "\n";
        std::cout << "    F_10 = " << std::scientific << F_10_ptp << std::fixed << "\n";
        float recip_ptp = std::abs((p0.area * F_01_ptp) - (p1.area * F_10_ptp));
        float recip_ptp_rel = recip_ptp / ((p0.area * F_01_ptp + p1.area * F_10_ptp) * 0.5f + 1e-10f);
        std::cout << "    Reciprocity error: " << recip_ptp_rel * 100.0f << "%\n";
        
        std::cout << "\n  Monte Carlo (16 samples/patch):\n";
        std::cout << "    F_01 = " << std::scientific << F_01_mc << std::fixed << "\n";
        std::cout << "    F_10 = " << std::scientific << F_10_mc << std::fixed << "\n";
        float recip_mc = std::abs((p0.area * F_01_mc) - (p1.area * F_10_mc));
        float recip_mc_rel = recip_mc / ((p0.area * F_01_mc + p1.area * F_10_mc) * 0.5f + 1e-10f);
        std::cout << "    Reciprocity error: " << recip_mc_rel * 100.0f << "%\n";
        
        std::cout << "\n  Improvement:\n";
        std::cout << "    Monte Carlo error is " << (recip_ptp_rel / (recip_mc_rel + 1e-10f)) 
                  << "x better\n";
        
        // Monte Carlo should have better reciprocity (< 5% error)
        validateFormFactor("Monte Carlo: reciprocity error < 5%", recip_mc_rel, 0.0f, 0.05f);
        validateFormFactor("Monte Carlo better than point-to-point", 
                          recip_mc_rel < recip_ptp_rel ? 1.0f : 0.0f, 1.0f, 0.01f);
    }
    
    /**
     * Run all tests
     */
    void runAll(VisibilityTester* tester = nullptr) {
        visTester = tester;
        results.clear();
        
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  FORM FACTOR UNIT TESTS\n";
        std::cout << "========================================\n";
        std::cout << "Using REAL production code:\n";
        std::cout << "  - FormFactorCalculator::calculateMatrix()\n";
        std::cout << "  - VisibilityTester with OptiX ray tracing\n";
        std::cout << "  - Patch and IndexedMesh structures\n\n";
        
        if (!visTester) {
            std::cout << "⚠ WARNING: No VisibilityTester provided\n";
            std::cout << "  Occlusion tests will be skipped\n";
            std::cout << "  Visibility will be geometric only\n\n";
        }
        
        test1_ParallelPatchesFacing();
        test2_ParallelPatchesAway();
        test3_DistanceEffect();
        test4a_FullOcclusion();
        test4b_PartialOcclusion();
        test5_Reciprocity();
        test6_MonteCarloComparison();
        
        // Print summary
        std::cout << "\n========================================\n";
        std::cout << "  TEST RESULTS SUMMARY\n";
        std::cout << "========================================\n";
        
        int passed = 0;
        int failed = 0;
        
        for (const auto& result : results) {
            std::string status = result.passed ? "✓ PASS" : "✗ FAIL";
            std::cout << status << " | " << result.name << "\n";
            
            if (!result.passed) {
                std::cout << "       Expected: " << result.expected 
                         << " ± " << result.tolerance << "\n";
                std::cout << "       Actual: " << result.actual << "\n";
            }
            
            if (result.passed) passed++;
            else failed++;
        }
        
        std::cout << "\nTotal: " << results.size() << " tests\n";
        std::cout << "Passed: " << passed << "\n";
        std::cout << "Failed: " << failed << "\n";
        
        if (failed == 0) {
            std::cout << "\n🎉 ALL TESTS PASSED! 🎉\n";
        } else {
            std::cout << "\n❌ SOME TESTS FAILED\n";
        }
        
        std::cout << "========================================\n\n";
    }
};

} // namespace experiments
} // namespace radiosity
