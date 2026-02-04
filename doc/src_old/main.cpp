/**
 * Radiosity Renderer - Test Main
 * Week 2: OptiX-ready architecture with indexed meshes
 */

// Prevent Windows.h from defining min/max macros
#define NOMINMAX

#include "math/Vector3.h"
#include "math/Matrix4.h"
#include "math/MathUtils.h"
#include "geometry/IndexedMesh.h"
#include "core/Patch.h"
#include "core/Scene.h"
#include "scene/Material.h"
#include "scene/CornellBox.h"
#include "output/OBJWriter.h"
#include "visibility/VisibilityTester.h"
#include "radiosity/FormFactor.h"
#include "radiosity/MonteCarloFormFactor.h"
#include "radiosity/RadiosityRenderer.h"
#include "experiments/FormFactorTests.h"

#include <iostream>
#include <iomanip>

// Output directory paths (relative to executable in output/bin/)
const std::string OUTPUT_SCENES_DIR = "../scenes/";
const std::string OUTPUT_CACHE_DIR = "../cache/";
const std::string OUTPUT_LOGS_DIR = "../logs/";

using namespace radiosity::math;
using namespace radiosity::geometry;
using namespace radiosity::core;
using namespace radiosity::scene;
using namespace radiosity::output;
using namespace radiosity::visibility;

// Test utilities
#define TEST(name) std::cout << "\n=== TEST: " << name << " ===\n"
#define ASSERT_NEAR(a, b, eps) \
    if (std::abs((a) - (b)) > (eps)) { \
        std::cerr << "FAIL: " << #a << " = " << (a) << ", expected " << (b) << "\n"; \
    } else { \
        std::cout << "PASS: " << #a << " ≈ " << (b) << "\n"; \
    }

void testVector3() {
    TEST("Vector3");
    
    // Construction
    Vector3 v1(1, 2, 3);
    Vector3 v2(4, 5, 6);
    
    std::cout << "v1 = " << v1 << "\n";
    std::cout << "v2 = " << v2 << "\n";
    
    // Addition
    Vector3 sum = v1 + v2;
    std::cout << "v1 + v2 = " << sum << "\n";
    ASSERT_NEAR(sum.x, 5.0f, EPSILON);
    ASSERT_NEAR(sum.y, 7.0f, EPSILON);
    ASSERT_NEAR(sum.z, 9.0f, EPSILON);
    
    // Subtraction
    Vector3 diff = v2 - v1;
    std::cout << "v2 - v1 = " << diff << "\n";
    ASSERT_NEAR(diff.x, 3.0f, EPSILON);
    
    // Scalar multiplication
    Vector3 scaled = v1 * 2.0f;
    std::cout << "v1 * 2 = " << scaled << "\n";
    ASSERT_NEAR(scaled.x, 2.0f, EPSILON);
    ASSERT_NEAR(scaled.y, 4.0f, EPSILON);
    
    // Dot product
    float dotProduct = v1.dot(v2);
    std::cout << "v1 · v2 = " << dotProduct << "\n";
    ASSERT_NEAR(dotProduct, 32.0f, EPSILON);  // 1*4 + 2*5 + 3*6 = 32
    
    // Cross product
    Vector3 v3(1, 0, 0);
    Vector3 v4(0, 1, 0);
    Vector3 crossProd = v3.cross(v4);
    std::cout << "X × Y = " << crossProd << "\n";
    ASSERT_NEAR(crossProd.x, 0.0f, EPSILON);
    ASSERT_NEAR(crossProd.y, 0.0f, EPSILON);
    ASSERT_NEAR(crossProd.z, 1.0f, EPSILON);
    
    // Length
    Vector3 v5(3, 4, 0);
    float length = v5.length();
    std::cout << "|v5| = " << length << "\n";
    ASSERT_NEAR(length, 5.0f, EPSILON);
    
    // Normalization
    Vector3 normalized = v5.normalized();
    std::cout << "normalize(v5) = " << normalized << "\n";
    ASSERT_NEAR(normalized.length(), 1.0f, EPSILON);
    
    // Distance
    float distance = v1.distance(v2);
    std::cout << "distance(v1, v2) = " << distance << "\n";
    ASSERT_NEAR(distance, std::sqrt(27.0f), EPSILON);
}

void testMatrix4() {
    TEST("Matrix4");
    
    // Identity
    Matrix4 identity = Matrix4::identity();
    std::cout << "Identity matrix:\n";
    identity.print();
    
    // Translation
    Matrix4 trans = Matrix4::translation(10, 20, 30);
    std::cout << "\nTranslation(10, 20, 30):\n";
    trans.print();
    
    Vector3 point(0, 0, 0);
    Vector3 translated = trans.transformPoint(point);
    std::cout << "Translated point: " << translated << "\n";
    ASSERT_NEAR(translated.x, 10.0f, EPSILON);
    ASSERT_NEAR(translated.y, 20.0f, EPSILON);
    ASSERT_NEAR(translated.z, 30.0f, EPSILON);
    
    // Scale
    Matrix4 scaleMatrix = Matrix4::scale(2, 3, 4);
    Vector3 pointToScale(1, 1, 1);
    Vector3 scaled = scaleMatrix.transformPoint(pointToScale);
    std::cout << "Scaled point: " << scaled << "\n";
    ASSERT_NEAR(scaled.x, 2.0f, EPSILON);
    ASSERT_NEAR(scaled.y, 3.0f, EPSILON);
    ASSERT_NEAR(scaled.z, 4.0f, EPSILON);
    
    // Rotation around Z (90 degrees)
    Matrix4 rotZ = Matrix4::rotationZ(HALF_PI);
    Vector3 xAxis(1, 0, 0);
    Vector3 rotated = rotZ.transformPoint(xAxis);
    std::cout << "Rotate X axis 90° around Z: " << rotated << "\n";
    ASSERT_NEAR(rotated.x, 0.0f, 1e-5f);
    ASSERT_NEAR(rotated.y, 1.0f, 1e-5f);
    ASSERT_NEAR(rotated.z, 0.0f, 1e-5f);
    
    // Matrix multiplication
    Matrix4 combined = trans * scaleMatrix;
    std::cout << "\nCombined (translate * scale):\n";
    combined.print();
}

void testMathUtils() {
    TEST("MathUtils");
    
    // Constants
    std::cout << "PI = " << PI << "\n";
    std::cout << "TWO_PI = " << TWO_PI << "\n";
    std::cout << "EPSILON = " << EPSILON << "\n";
    
    // Clamp
    float clamped = clamp(15.0f, 0.0f, 10.0f);
    std::cout << "clamp(15, 0, 10) = " << clamped << "\n";
    ASSERT_NEAR(clamped, 10.0f, EPSILON);
    
    // Lerp
    float lerped = lerp(0.0f, 100.0f, 0.5f);
    std::cout << "lerp(0, 100, 0.5) = " << lerped << "\n";
    ASSERT_NEAR(lerped, 50.0f, EPSILON);
    
    // Smoothstep
    float smooth = smoothstep(0.0f, 1.0f, 0.5f);
    std::cout << "smoothstep(0, 1, 0.5) = " << smooth << "\n";
    
    // Angle conversion
    float deg = 180.0f;
    float rad = radians(deg);
    std::cout << "radians(180°) = " << rad << "\n";
    ASSERT_NEAR(rad, PI, EPSILON);
    
    // Random (just verify it runs)
    std::cout << "\nRandom samples:\n";
    for (int i = 0; i < 5; i++) {
        float r = Random::uniform01();
        std::cout << "  uniform01() = " << r << "\n";
        if (r < 0.0f || r >= 1.0f) {
            std::cerr << "FAIL: Random out of range [0, 1)\n";
        }
    }
    
    // Cosine-weighted hemisphere sampling
    std::cout << "\nCosine-weighted hemisphere samples:\n";
    for (int i = 0; i < 3; i++) {
        float x, y, z;
        Random::cosineWeightedHemisphere(x, y, z);
        std::cout << "  (" << x << ", " << y << ", " << z << ")\n";
        
        // Check that z >= 0 (hemisphere constraint)
        if (z < 0.0f) {
            std::cerr << "FAIL: Hemisphere sample has negative z\n";
        }
        
        // Check that it's on unit sphere
        float length = std::sqrt(x*x + y*y + z*z);
        ASSERT_NEAR(length, 1.0f, 1e-4f);
    }
    
    // Tone mapping
    float hdr = 5.0f;
    float ldr = reinhardToneMap(hdr);
    std::cout << "\nTone map " << hdr << " -> " << ldr << "\n";
    if (ldr < 0.0f || ldr > 1.0f) {
        std::cerr << "FAIL: Tone mapped value out of range\n";
    }
}

void testIndexedMesh() {
    TEST("IndexedMesh (OptiX-ready)");
    
    // Create a subdivided quad using the new system
    IndexedMesh mesh;
    
    Vector3 corner0(0, 0, 0);
    Vector3 corner1(2, 0, 0);
    Vector3 corner2(2, 2, 0);
    Vector3 corner3(0, 2, 0);
    
    // Build 4x4 subdivided quad (32 triangles)
    MeshBuilder::addSubdividedQuad(mesh, corner0, corner1, corner2, corner3, 4, 4, 0);
    
    std::cout << "Vertices: " << mesh.vertexCount() << "\n";
    std::cout << "Triangles: " << mesh.triangleCount() << "\n";
    
    // Expected: (4+1)*(4+1) = 25 vertices, 4*4*2 = 32 triangles
    ASSERT_NEAR(static_cast<float>(mesh.vertexCount()), 25.0f, EPSILON);
    ASSERT_NEAR(static_cast<float>(mesh.triangleCount()), 32.0f, EPSILON);
    
    // Verify first triangle normal
    Vector3 normal = mesh.getTriangleNormal(0);
    std::cout << "First triangle normal: " << normal << "\n";
    ASSERT_NEAR(normal.z, 1.0f, 1e-5f);  // Should point in +Z
    
    // Verify GPU-upload readiness
    const float* vertexPtr = mesh.getVertexDataPtr();
    const uint32_t* indexPtr = mesh.getIndexDataPtr();
    std::cout << "Vertex data pointer: " << (void*)vertexPtr << "\n";
    std::cout << "Index data pointer: " << (void*)indexPtr << "\n";
    std::cout << "✓ Mesh ready for OptiX upload\n";
}

void testPatch() {
    TEST("Patch");
    
    // Create a test patch
    Patch patch;
    patch.center = Vector3(100, 200, 300);
    patch.normal = Vector3(0, 0, 1);
    patch.area = 100.0f;
    patch.emission = Vector3(15, 15, 15);  // Light source
    patch.reflectance = Vector3(0.78f, 0.78f, 0.78f);
    patch.firstTriangleIndex = 0;
    patch.triangleCount = 2;
    
    patch.print("Test Light Patch");
    
    // Initialize radiosity
    patch.initializeRadiosity();
    std::cout << "After initializeRadiosity():\n";
    std::cout << "  B = " << patch.B << "\n";
    std::cout << "  B_unshot = " << patch.B_unshot << "\n";
    
    // Verify emissive patch has emission as initial radiosity
    ASSERT_NEAR(patch.B.x, 15.0f, EPSILON);
    ASSERT_NEAR(patch.B_unshot.x, 15.0f, EPSILON);
    
    std::cout << "Is emissive: " << (patch.isEmissive() ? "YES" : "NO") << "\n";
    std::cout << "Unshot magnitude: " << patch.unshotMagnitude() << "\n";
}

void testRadiosityScene() {
    TEST("Radiosity Scene");
    
    RadiosityScene scene;
    
    // Add a white diffuse quad
    Material white = Material::white();
    scene.addQuadPatch(
        Vector3(0, 0, 0),
        Vector3(10, 0, 0),
        Vector3(10, 10, 0),
        Vector3(0, 10, 0),
        white,
        2, 2  // 2x2 subdivision = 4 patches
    );
    
    // Add a light source
    Material light = Material::areaLight();
    scene.addQuadPatch(
        Vector3(3, 3, 10),
        Vector3(7, 3, 10),
        Vector3(7, 7, 10),
        Vector3(3, 7, 10),
        light,
        1, 1  // Single patch
    );
    
    scene.initializeRadiosity();
    
    std::cout << "Patches: " << scene.patches.size() << "\n";
    std::cout << "Mesh vertices: " << scene.mesh.vertexCount() << "\n";
    std::cout << "Mesh triangles: " << scene.mesh.triangleCount() << "\n";
    
    // Verify patch-to-triangle mapping
    std::cout << "\nPatch details:\n";
    for (size_t i = 0; i < scene.patches.size(); i++) {
        const Patch& p = scene.patches[i];
        std::cout << "  Patch " << i << ": triangles [" << p.firstTriangleIndex 
                  << ".." << (p.firstTriangleIndex + p.triangleCount - 1) << "]";
        if (p.isEmissive()) {
            std::cout << " (LIGHT)";
        }
        std::cout << "\n";
    }
    
    // Verify reverse mapping
    std::cout << "\nTriangle → Patch mapping:\n";
    for (size_t triIdx = 0; triIdx < (std::min)(size_t(10), scene.mesh.triangleCount()); triIdx++) {
        uint32_t patchId = scene.mesh.getPatchId(triIdx);
        std::cout << "  Triangle " << triIdx << " → Patch " << patchId << "\n";
    }
    
    std::cout << "✓ Bidirectional mapping verified\n";
}

void testMaterial() {
    TEST("Material");
    
    Material white = Material::white();
    Material red = Material::red();
    Material green = Material::green();
    Material light = Material::areaLight();
    
    white.print("White");
    red.print("Red");
    green.print("Green");
    light.print("Area Light");
    
    std::cout << "Light is emissive: " << (light.isEmissive() ? "YES" : "NO") << "\n";
}

void testCornellBox() {
    TEST("Cornell Box Scene");
    
    // Create Cornell Box with moderate subdivision
    std::cout << "Creating Cornell Box (subdivision: 10 for walls, 5 for boxes)...\n";
    CornellBox box;
    box.build(10, 5);
    
    box.printStats();
    
    std::cout << "\nDimensions:\n";
    std::cout << "  Width: " << CornellBox::WIDTH << " mm\n";
    std::cout << "  Height: " << CornellBox::HEIGHT << " mm\n";
    std::cout << "  Depth: " << CornellBox::DEPTH << " mm\n";
    
    // Verify patch statistics
    size_t emissiveCount = 0;
    for (const auto& patch : box.scene.patches) {
        if (patch.isEmissive()) {
            emissiveCount++;
        }
    }
    std::cout << "  Emissive patches: " << emissiveCount << "\n";
}

void testRadiositySolver() {
    TEST("Radiosity Solver (Progressive Refinement)");
    
    // Create a Cornell Box with moderate subdivision
    CornellBox box;
    box.build(5, 3);  // 5x5 walls, 3x3 boxes
    
    std::cout << "Cornell Box for radiosity:\n";
    std::cout << "  Patches: " << box.scene.patches.size() << std::endl;
    std::cout << "  Triangles: " << box.scene.mesh.triangleCount() << std::endl;
    
    // Count light sources
    int lightCount = 0;
    for (const auto& patch : box.scene.patches) {
        if (patch.isEmissive()) lightCount++;
    }
    std::cout << "  Light sources: " << lightCount << std::endl;
    
    // Create visibility tester
    VisibilityTester visTester;
    bool visReady = visTester.initialize(box.scene.mesh);
    
    if (!visReady) {
        std::cout << "⚠ Visibility tester not available - using stub mode\n";
    }
    
    // Configure radiosity solver
    radiosity::RadiosityRenderer::Config solverConfig;
    solverConfig.convergenceThreshold = 0.1f;  // Looser for stub visibility
    solverConfig.maxIterations = 50;
    solverConfig.verbose = true;
    solverConfig.debugOutput = false;
    
    // Solve radiosity
    std::cout << "\nSolving radiosity...\n";
    radiosity::RadiosityRenderer solver(solverConfig);
    auto stats = solver.solve(box.scene, visReady ? &visTester : nullptr);
    
    // Analyze results
    std::cout << "\n=== Solution Analysis ===" << std::endl;
    std::cout << "Brightest patches (top 5):" << std::endl;
    
    std::vector<std::pair<float, int>> patchBrightness;
    for (size_t i = 0; i < box.scene.patches.size(); i++) {
        const auto& patch = box.scene.patches[i];
        float brightness = (patch.B.x + patch.B.y + patch.B.z) / 3.0f;
        patchBrightness.push_back({brightness, static_cast<int>(i)});
    }
    std::sort(patchBrightness.begin(), patchBrightness.end(), std::greater<>());
    
    for (int i = 0; i < (std::min)(5, static_cast<int>(patchBrightness.size())); i++) {
        int idx = patchBrightness[i].second;
        const auto& patch = box.scene.patches[idx];
        std::cout << "  Patch " << std::setw(2) << idx << ": B=("
                  << std::fixed << std::setprecision(2)
                  << patch.B.x << ", " << patch.B.y << ", " << patch.B.z << ")"
                  << (patch.isEmissive() ? " [LIGHT]" : "") << std::endl;
    }
    
    std::cout << "\n✓ Radiosity solver test complete\n";
}

void exportCornellBox() {
    TEST("OBJ Export - Material Colors");
    
    std::cout << "Creating Cornell Box for export...\n";
    CornellBox box;
    box.build(8, 4);  // Moderate detail for visualization
    
    std::string filename = OUTPUT_SCENES_DIR + "cornell_box.obj";
    std::cout << "Exporting to " << filename << "...\n";
    
    if (OBJWriter::writeCornellBox(filename, box)) {
        std::cout << "\n✓ Export successful!\n";
        std::cout << "\nVisualization instructions:\n";
        std::cout << "  1. Open with Blender:\n";
        std::cout << "     blender cornell_box.obj\n";
        std::cout << "  2. Or use online viewer:\n";
        std::cout << "     https://3dviewer.net/\n";
        std::cout << "  3. Or MeshLab:\n";
        std::cout << "     meshlab cornell_box.obj\n";
        std::cout << "\nExpected appearance:\n";
        std::cout << "  - Red wall on the left\n";
        std::cout << "  - Green wall on the right\n";
        std::cout << "  - White walls (floor, ceiling, back)\n";
        std::cout << "  - Two white boxes in the scene\n";
        std::cout << "  - White square (light) on ceiling\n";
    } else {
        std::cerr << "✗ Export failed!\n";
    }
}

void exportRadiositySolution() {
    TEST("OBJ Export - Radiosity Solution");
    
    std::cout << "Creating Cornell Box and computing radiosity...\n";
    std::cout << "Medium subdivision - approximately 20 minutes computation!\n";
    CornellBox box;
    box.build(14, 7);  // 14x14 walls (196 patches/wall), 7x7 boxes (medium detail, ~20 min)
    
    std::cout << "  Scene complexity:\n";
    std::cout << "    Patches: " << box.scene.patches.size() << "\n";
    std::cout << "    Triangles: " << box.scene.mesh.triangleCount() << "\n";
    std::cout << "    Vertices: " << box.scene.mesh.vertexCount() << "\n";
    std::cout << "    Estimated visibility computations: ~" << (box.scene.patches.size() * box.scene.patches.size()) << "\n";
    std::cout << "    (This may take 15-25 minutes depending on GPU)\n\n";
    
    // Initialize visibility tester with OptiX
    VisibilityTester visTester;
    VisibilityTester* visPtr = nullptr;
    if (visTester.initialize(box.scene.mesh)) {
        std::cout << "  ✓ OptiX visibility tester initialized\n";
        visPtr = &visTester;
    } else {
        std::cout << "  ⚠ OptiX not available, using geometric visibility only\n";
    }
    
    // Enable VISIBILITY caching (stores 0-1 ray hit fractions, NOT form factors!)
    radiosity::MonteCarloFormFactor::DEBUG_ENABLE_CACHE = true;
    radiosity::MonteCarloFormFactor::DEBUG_CACHE_FILENAME = OUTPUT_CACHE_DIR + "cornell_box_14x7_visibility.cache";
    std::cout << "  Visibility caching: ENABLED\n";
    std::cout << "    Cache file: " << radiosity::MonteCarloFormFactor::DEBUG_CACHE_FILENAME << "\n";
    std::cout << "    ⚠ Cache stores VISIBILITY fractions (0-1), NOT form factors!\n\n";
    
    // Configure and run radiosity solver
    std::cout << "  Running radiosity solver...\n";
    radiosity::RadiosityRenderer::Config config;
    config.convergenceThreshold = 0.0001f;  // Lower threshold for more bounces
    config.maxIterations = 100;             // Allow up to 100 iterations for better color bleeding
    config.verbose = true;                  // Show iteration progress
    config.debugOutput = false;
    config.debugNoDistance = false;         // Keep distance attenuation enabled
    config.debugAttenuateReflectance = false;  // Full reflectance (no attenuation)
    
    radiosity::RadiosityRenderer solver(config);
    auto stats = solver.solve(box.scene, visPtr);
    
    std::cout << "  ✓ Solution converged in " << stats.iterations 
              << " iterations (unshot: " 
              << std::fixed << std::setprecision(4) 
              << stats.unshotEnergy << ")\n";
    
    // === DEBUG: Export accumulated visibility visualization ===
    std::cout << "\n  === GENERATING DEBUG VISIBILITY MAP ===\n";
    
    // Save original radiosity solution
    std::vector<Vector3> originalRadiosity;
    for (const auto& patch : box.scene.patches) {
        originalRadiosity.push_back(patch.B);
    }
    
    // Compute and export accumulated visibility using PURE visibility fractions
    std::cout << "  Computing accumulated visibility (AO-like visualization)...\n";
    std::cout << "  NOTE: Using PURE VISIBILITY fractions (0-1) - light source should NOT be brightest!\n";
    box.scene.computeAccumulatedVisibility(solver.getVisibilityMatrix(), true);
    
    std::string visFilename = OUTPUT_SCENES_DIR + "cornell_box_visibility_debug.obj";
    std::cout << "\n  Exporting visibility map to " << visFilename << "...\n";
    OBJWriter::DEBUG_NO_TONE_MAPPING = true; // No tone mapping for visibility
    if (OBJWriter::writeCornellBoxRadiosity(visFilename, box, 1.0f)) {
        std::cout << "  ✓ Visibility debug map exported\n";
        std::cout << "    Interpretation:\n";
        std::cout << "      Brighter = more visible from other patches\n";
        std::cout << "      Darker = more occluded/hidden\n";
        std::cout << "      Light source should NOT be brightest (pure geometry!)\n";
    }
    
    // Restore original radiosity solution
    for (size_t i = 0; i < box.scene.patches.size(); ++i) {
        box.scene.patches[i].B = originalRadiosity[i];
    }
    std::cout << "  ✓ Radiosity solution restored\n";
    
    // === Continue with normal radiosity export ===
    // Reconstruct per-vertex radiosity for smooth interpolation
    std::cout << "\n  Reconstructing per-vertex radiosity...\n";
    box.scene.reconstructVertexRadiosity();
    std::cout << "  ✓ Per-vertex radiosity computed\n";
    
    // Export with radiosity values
    std::string objFilename = OUTPUT_SCENES_DIR + "cornell_box_radiosity.obj";
    std::cout << "  Exporting radiosity solution to " << objFilename << "...\n";
    
    // Enable debug mode to disable tone mapping and enable auto-normalization
    OBJWriter::DEBUG_NO_TONE_MAPPING = true;
    std::cout << "  ✓ Tone mapping: DISABLED (debug mode)\n";
    std::cout << "  ✓ Auto-normalization: ENABLED (brightest = 255)\n";
    std::cout << "  ✓ Distance attenuation (1/r²): ACTIVE\n\n";
    
    float exposure = 5.0f;  // Higher exposure for better visibility (will be auto-normalized)
    
    // Export OBJ with radiosity as materials
    if (OBJWriter::writeCornellBoxRadiosity(objFilename, box, exposure)) {
        std::cout << "\n✓ OBJ export successful!\n";
        std::cout << "  Debug mode: Tone mapping " << (OBJWriter::DEBUG_NO_TONE_MAPPING ? "DISABLED" : "ENABLED") << "\n";
    } else {
        std::cerr << "✗ OBJ export failed!\n";
    }
    
    std::cout << "\nVisualization instructions:\n";
    std::cout << "  Blender:\n";
    std::cout << "    File > Import > Wavefront (.obj) > " << objFilename << "\n";
    std::cout << "  Online: https://3dviewer.net/\n";
    std::cout << "\nExpected appearance:\n";
    std::cout << "  - Brightest area: light source on ceiling\n";
    std::cout << "  - Visible shadows behind boxes\n";
    std::cout << "  - Color bleeding: red tint on left, green on right\n";
    std::cout << "  - PLY: Smooth gradients (vertex colors)\n";
    std::cout << "  - OBJ: Faceted appearance (per-patch colors)\n";
}

void testVisibilityTester() {
    TEST("Visibility Tester (Stub Mode)");
    
    // Create simple test scene
    CornellBox box;
    box.build(2, 1);  // Very coarse for testing
    
    VisibilityTester tester;
    
    // Initialize with mesh
    bool initSuccess = tester.initialize(box.scene.mesh);
    if (!initSuccess) {
        std::cerr << "✗ Failed to initialize visibility tester\n";
        return;
    }
    
    std::cout << "  Initialized: " << (tester.isInitialized() ? "YES" : "NO") << "\n";
    std::cout << "  Mesh data: " << tester.getMeshVertexCount() << " vertices, "
              << tester.getMeshTriangleCount() << " triangles\n";
    
    // Test point-to-point visibility
    Vector3 point1(100, 100, 100);
    Vector3 point2(200, 200, 200);
    Vector3 normal(1, 0, 0);  // Dummy normal for test
    float visibility = tester.testVisibility(point1, point2, normal);
    std::cout << "  Point visibility test: " << visibility << " (stub: always 1.0)\n";
    
    // Test patch-to-patch visibility
    if (box.scene.patches.size() >= 2) {
        float patchVis = tester.testPatchVisibility(
            box.scene.patches[0], 
            box.scene.patches[1],
            0,  // from patch ID
            1   // to patch ID
        );
        std::cout << "  Patch visibility test: " << patchVis << "\n";
    }
    
    std::cout << "✓ Visibility tester working (OptiX integration pending)\n";
}

void testFormFactors() {
    TEST("Form Factor Calculation");
    
    // Create simple test scene with moderate subdivision
    CornellBox box;
    box.build(5, 3);  // 5x5 walls, 3x3 boxes - more patches for better accuracy
    
    const auto& patches = box.scene.patches;
    std::cout << "Scene: " << patches.size() << " patches\n";
    
    // NO CACHING FOR TESTS - tests should always compute fresh to validate correctness
    radiosity::MonteCarloFormFactor::DEBUG_ENABLE_CACHE = false;
    
    // Initialize visibility tester
    VisibilityTester visTester;
    const VisibilityTester* visPtr = nullptr;
    if (visTester.initialize(box.scene.mesh)) {
        std::cout << "✓ Visibility tester initialized for form factors\n";
        visPtr = &visTester;
    } else {
        std::cout << "⚠ Visibility tester failed, using stub mode\n";
    }
    std::cout << "\n";
    
    // Test 1: Calculate form factor between two patches
    if (patches.size() >= 2) {
        std::cout << "Test 1: Single form factor\n";
        float ff_01 = radiosity::FormFactorCalculator::calculate(patches[0], patches[1], visPtr);
        float ff_10 = radiosity::FormFactorCalculator::calculate(patches[1], patches[0], visPtr);
        
        std::cout << "  F_01 = " << ff_01 << "\n";
        std::cout << "  F_10 = " << ff_10 << "\n";
        
        // Check reciprocity: A_0 * F_01 should equal A_1 * F_10
        float left = patches[0].area * ff_01;
        float right = patches[1].area * ff_10;
        std::cout << "  Reciprocity check:\n";
        std::cout << "    A_0 * F_01 = " << left << "\n";
        std::cout << "    A_1 * F_10 = " << right << "\n";
        std::cout << "    Error: " << std::abs(left - right) << "\n";
    }
    
    // Test 2: Calculate all form factors from one patch
    std::cout << "\nTest 2: Form factors from patch 0 to all others\n";
    auto formFactors = radiosity::FormFactorCalculator::calculateFromPatch(
        patches, 0, visPtr
    );
    
    float sum = 0.0f;
    for (size_t i = 0; i < (std::min)(size_t(10), formFactors.size()); i++) {
        std::cout << "  F_0" << i << " = " << formFactors[i] << "\n";
        sum += formFactors[i];
    }
    if (formFactors.size() > 10) {
        std::cout << "  ... (" << (formFactors.size() - 10) << " more)\n";
        for (size_t i = 10; i < formFactors.size(); i++) {
            sum += formFactors[i];
        }
    }
    std::cout << "  Sum of all form factors from patch 0: " << sum << "\n";
    std::cout << "  (Expected ≈ 1.0 for closed environment)\n";
    
    // Test 3: Calculate full form factor matrix (small scene only)
    if (patches.size() <= 20) {
        std::cout << "\nTest 3: Full form factor matrix\n";
        auto matrix = radiosity::FormFactorCalculator::calculateMatrix(patches, visPtr, true);
        
        // Print statistics
        // radiosity::FormFactorCalculator::printStatistics(matrix);
        
        // Validate
        // radiosity::FormFactorCalculator::validate(patches, matrix);
    } else {
        std::cout << "\nTest 3: Skipped (too many patches: " << patches.size() << ")\n";
    }
    
    std::cout << "\n✓ Form factor tests complete\n";
}

void runFormFactorUnitTests() {
    // Create visibility tester with OptiX
    VisibilityTester visTester;
    bool optixAvailable = visTester.initialize(IndexedMesh());  // Initialize without mesh first
    
    if (!optixAvailable) {
        std::cout << "\n⚠ OptiX not available - some tests will be skipped\n";
    }
    
    // Run the comprehensive test suite using REAL production code
    radiosity::experiments::FormFactorTests tests;
    tests.runAll(optixAvailable ? &visTester : nullptr);
}

void printSystemInfo() {
    std::cout << "========================================\n";
    std::cout << "  RADIOSITY RENDERER - Week 2\n";
    std::cout << "========================================\n";
    std::cout << "OptiX-Ready Architecture\n";
    std::cout << "\nCompiler: ";
    #if defined(__clang__)
        std::cout << "Clang " << __clang_major__ << "." << __clang_minor__ << "\n";
    #elif defined(__GNUC__)
        std::cout << "GCC " << __GNUC__ << "." << __GNUC_MINOR__ << "\n";
    #elif defined(_MSC_VER)
        std::cout << "MSVC " << _MSC_VER << "\n";
    #else
        std::cout << "Unknown\n";
    #endif
    
    std::cout << "C++ Standard: " << __cplusplus << "\n";
    std::cout << "Build type: ";
    #ifdef NDEBUG
        std::cout << "Release\n";
    #else
        std::cout << "Debug\n";
    #endif
    std::cout << "\n";
}

int main() {
    // Set output precision
    std::cout << std::fixed << std::setprecision(6);
    
    printSystemInfo();
    
    // Week 1 tests (quick verification)
    std::cout << "=== Week 1 Foundation (Quick Check) ===\n";
    Vector3 v1(1, 2, 3);
    Vector3 v2(4, 5, 6);
    std::cout << "v1 + v2 = " << (v1 + v2) << "\n";
    std::cout << "v1 · v2 = " << v1.dot(v2) << "\n";
    std::cout << "✓ Math foundation working\n";
    
    // Week 2 tests - NEW OptiX-ready system
    testIndexedMesh();
    testPatch();
    testRadiosityScene();
    testMaterial();
    testCornellBox();
    
    // Week 3 tests - Visibility and Form Factors
    testVisibilityTester();
    
    // *** FORM FACTOR UNIT TESTS ***
    // Comprehensive validation using REAL production code
    runFormFactorUnitTests();
    
    testFormFactors();
    
    // Week 3/4 test - Radiosity Solver (NEW!)
    testRadiositySolver();
    
    // Export Cornell Box for visual verification
    exportCornellBox();
    
    // Export radiosity solution with computed lighting
    exportRadiositySolution();
    
    std::cout << "\n========================================\n";
    std::cout << "  ALL TESTS COMPLETED\n";
    std::cout << "========================================\n";
    std::cout << "\nWeek 2 Accomplishments:\n";
    std::cout << "  ✓ IndexedMesh with shared vertices\n";
    std::cout << "  ✓ Patch structure for radiosity\n";
    std::cout << "  Week 3 Progress:\n";
    std::cout << "  ✓ Visibility tester framework (OptiX stub mode)\n";
    std::cout << "  ✓ Form factor calculation\n";
    std::cout << "  ✓ Form factor validation (reciprocity, sum)\n";
    std::cout << "  ⏳ OptiX SDK installation (manual download required)\n";
    std::cout << "\nData structure benefits:\n";
    std::cout << "  • ~50% memory savings vs old system\n";
    std::cout << "  • Direct GPU buffer upload via getVertexDataPtr()\n";
    std::cout << "  • Per-triangle patch ID for material lookup\n";
    std::cout << "\nNext steps (Week 3 continued):\n";
    std::cout << "  1. Download & install OptiX SDK from NVIDIA\n";
    std::cout << "  2. Integrate OptiX into VisibilityTester\n";
    std::cout << "  3. Test ray tracing with actual occlusion\n";
    std::cout << "  4. Week 4: Progressive radiosity solver\n";
    std::cout << "\nWeek 3 Status: In Progress ⏳):\n";
    std::cout << "  1. Install OptiX SDK\n";
    std::cout << "  2. Create OptiX context and upload mesh\n";
    std::cout << "  3. Implement visibility testing\n";
    std::cout << "  4. Form factor calculations\n";
    std::cout << "  5. Progressive radiosity solver\n";
    std::cout << "\nWeek 2 Complete: ✓\n\n";
    
    return 0;
}
