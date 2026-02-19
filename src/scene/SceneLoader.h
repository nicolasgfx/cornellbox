#pragma once
#include "../mesh/MeshData.h"
#include "../mesh/AdaptiveRefinement.h"
#include "../solver/FormFactorRefinement.h"
#include "../scene/OBJLoader.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include "../app/Config.h"
#include <string>
#include <vector>
#include <iostream>
#include <cmath>
#include <algorithm>

namespace SceneLoader {

// --------------------------------------------------------------------------
// Scene data: mesh + material table + camera suggestion
// --------------------------------------------------------------------------
struct Scene {
    Mesh mesh;
    Mesh preRefinementMesh;  // mesh after uniform subdivision but before FF refinement
    std::vector<OBJLoader::MTLMaterial> materials;
    Vec3 bboxMin, bboxMax;

    // Suggested camera placement (inside the model, at centroid).
    Vec3 cameraEye{0.0f, 0.0f, 0.0f};
    Vec3 cameraLookAt{0.0f, 0.0f, -1.0f};
    float cameraFovY = 60.0f;

    // Light emission: which materials emit?
    // If no Ke found, we'll add a synthetic light.
    bool hasEmissiveMaterials = false;
    uint32_t emissiveTriCount = 0;
};

// --------------------------------------------------------------------------
// Assign per-triangle material properties from the MTL material table.
// Called after subdivision when triangle_material_id is populated.
// --------------------------------------------------------------------------
inline void buildMaterialData(Mesh& mesh,
                               const std::vector<OBJLoader::MTLMaterial>& materials) {
    size_t N = mesh.numTriangles();
    mesh.triangle_reflectance.resize(N);
    mesh.triangle_emission.resize(N);

    for (size_t i = 0; i < N; ++i) {
        uint32_t matId = (i < mesh.triangle_material_id.size())
                         ? mesh.triangle_material_id[i] : 0u;
        if (matId < materials.size()) {
            const auto& mat = materials[matId];
            // Use Kd for diffuse reflectance.
            // If Kd is black but Ka is non-zero, use Ka as fallback.
            Vec3 kd = mat.Kd;
            if (kd.lengthSq() < 1e-6f && mat.Ka.lengthSq() > 1e-6f) {
                kd = mat.Ka;
            }
            // Clamp reflectance to [0,1] range.
            kd.x = std::min(std::max(kd.x, 0.0f), 1.0f);
            kd.y = std::min(std::max(kd.y, 0.0f), 1.0f);
            kd.z = std::min(std::max(kd.z, 0.0f), 1.0f);
            mesh.triangle_reflectance[i] = kd;
            mesh.triangle_emission[i] = mat.Ke * kLightBrightnessScale;
        } else {
            mesh.triangle_reflectance[i] = Vec3(0.8f);
            mesh.triangle_emission[i] = Vec3(0.0f);
        }
    }
}

// --------------------------------------------------------------------------
// Compute per-triangle geometry (area, normal, centroid).
// Standalone version that doesn't depend on CornellBox.
// --------------------------------------------------------------------------
inline void buildTriangleGeometry(Mesh& mesh) {
    size_t N = mesh.numTriangles();
    mesh.triangle_area.resize(N);
    mesh.triangle_normal.resize(N);
    mesh.triangle_centroid.resize(N);

    for (size_t i = 0; i < N; ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();
        mesh.triangle_area[i]     = MathUtils::triangleArea(v0, v1, v2);
        mesh.triangle_centroid[i] = MathUtils::triangleCentroid(v0, v1, v2);
        mesh.triangle_normal[i]   = MathUtils::triangleNormal(v0, v1, v2).normalized();
    }
}

// --------------------------------------------------------------------------
// Validate mesh (standalone).
// --------------------------------------------------------------------------
inline bool validateMesh(const Mesh& mesh) {
    size_t badN = 0, badA = 0;
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const Vec3& n = mesh.triangle_normal[i];
        if (n.isZero() || n.hasNaN()) badN++;
        if (mesh.triangle_area[i] <= 0.0f) badA++;
    }
    if (badN || badA)
        std::cerr << "Mesh validation: " << badN << " bad normals, " << badA << " bad areas\n";
    return badN == 0 && badA == 0;
}

// --------------------------------------------------------------------------
// Suggest camera placement: center of bounding box, looking along -Z.
// For interiors, place camera slightly above center looking forward.
// --------------------------------------------------------------------------
inline void suggestCamera(Scene& scene) {
    Vec3 extent = scene.bboxMax - scene.bboxMin;
    float maxExtent = std::max({extent.x, extent.y, extent.z});

    // Place eye at center, slightly above midpoint for interior scenes.
    scene.cameraEye = Vec3(0.0f, extent.y * 0.05f, 0.0f);

    // Look along the shorter horizontal axis (likely interior depth).
    if (extent.z > extent.x) {
        scene.cameraLookAt = Vec3(0.0f, scene.cameraEye.y, -1.0f);
    } else {
        scene.cameraLookAt = Vec3(-1.0f, scene.cameraEye.y, 0.0f);
    }

    // FOV to show reasonable interior view.
    scene.cameraFovY = 60.0f;
}

// --------------------------------------------------------------------------
// Add a synthetic ceiling light if no emissive materials exist.
// Creates a small quad emitter at the top center of the bounding box.
// --------------------------------------------------------------------------
inline void ensureLight(Scene& scene) {
    // Check if any triangle already emits light.
    for (size_t i = 0; i < scene.mesh.numTriangles(); ++i) {
        if (scene.mesh.triangle_emission[i].lengthSq() > 1e-6f) {
            scene.hasEmissiveMaterials = true;
            scene.emissiveTriCount++;
        }
    }

    if (scene.hasEmissiveMaterials) {
        std::cout << "  Emissive triangles: " << scene.emissiveTriCount << "\n";
        return;
    }

    // No lights found: add a synthetic ceiling light.
    std::cout << "  No emissive materials — adding synthetic ceiling light\n";

    Vec3 extent = scene.bboxMax - scene.bboxMin;
    float lightSize = std::min(extent.x, extent.z) * 0.15f;
    float ceilingY = scene.bboxMax.y - extent.y * 0.01f; // just below ceiling

    uint32_t base = static_cast<uint32_t>(scene.mesh.vertices.size());
    scene.mesh.vertices.push_back(Vertex(Vec3(-lightSize, ceilingY, -lightSize)));
    scene.mesh.vertices.push_back(Vertex(Vec3( lightSize, ceilingY, -lightSize)));
    scene.mesh.vertices.push_back(Vertex(Vec3( lightSize, ceilingY,  lightSize)));
    scene.mesh.vertices.push_back(Vertex(Vec3(-lightSize, ceilingY,  lightSize)));

    // Two triangles forming the light quad.
    scene.mesh.indices.push_back(TriIdx(base, base + 2, base + 1));
    scene.mesh.indices.push_back(TriIdx(base, base + 3, base + 2));

    // Add a light material.
    uint32_t lightMatId = static_cast<uint32_t>(scene.materials.size());
    OBJLoader::MTLMaterial lightMat;
    lightMat.name = "_synthetic_light";
    lightMat.Kd = Vec3(0.78f, 0.78f, 0.78f);
    lightMat.Ke = Vec3(18.4f, 15.6f, 8.0f);
    scene.materials.push_back(lightMat);

    scene.mesh.triangle_material_id.push_back(lightMatId);
    scene.mesh.triangle_material_id.push_back(lightMatId);

    scene.emissiveTriCount = 2;
    scene.hasEmissiveMaterials = true;
}

// --------------------------------------------------------------------------
// Full pipeline: load OBJ → center → subdivide → build patch data.
// --------------------------------------------------------------------------
inline Scene loadScene(const std::string& objPath) {
    Scene scene;

    std::cout << "\n=== Loading OBJ scene ===\n";
    std::cout << "  File: " << objPath << "\n";

    // 1) Load OBJ + MTL.
    auto loaded = OBJLoader::loadOBJ(objPath);
    if (loaded.mesh.numTriangles() == 0) {
        std::cerr << "Error: failed to load scene\n";
        return scene;
    }
    OBJLoader::printLoadSummary(loaded);

    scene.mesh = std::move(loaded.mesh);
    scene.materials = std::move(loaded.materialTable);
    scene.bboxMin = loaded.bboxMin;
    scene.bboxMax = loaded.bboxMax;

    // 2) Uniform area subdivision.
    Vec3 extent = scene.bboxMax - scene.bboxMin;
    float maxExtent = std::max({extent.x, extent.y, extent.z});

    // Scale target area relative to scene size.
    // kSubdivisionTargetArea is designed for 1×1×1 Cornell Box.
    // Scale proportionally to scene extent².
    float scaleRatio = maxExtent; // scene extent vs 1.0 Cornell
    float scaledTargetArea = kSubdivisionTargetArea * scaleRatio * scaleRatio;

    // Cap with absolute maximum so large-scene patches stay fine enough.
    if (kMaxAbsoluteTriangleArea > 0.0f)
        scaledTargetArea = std::min(scaledTargetArea, kMaxAbsoluteTriangleArea);

    AdaptiveRefinement::Options refOpts;
    refOpts.targetArea    = scaledTargetArea;
    refOpts.minEdgeLength = maxExtent * 0.001f;
    refOpts.maxTriangles  = kMaxSubdivisionTriangles;
    refOpts.maxDepth      = 20;
    refOpts.maxEdgeRatio  = kMaxTriangleEdgeRatio;

    std::cout << "\n=== Subdivision ===\n";
    std::cout << "  Target area: " << refOpts.targetArea
              << " (scene-scaled " << (kSubdivisionTargetArea * scaleRatio * scaleRatio)
              << ", abs cap " << kMaxAbsoluteTriangleArea << ")\n";
    if (refOpts.maxEdgeRatio > 0.0f)
        std::cout << "  Edge ratio cap: " << refOpts.maxEdgeRatio << "\n";
    std::cout << "  Triangle budget: " << refOpts.maxTriangles << "\n";

    scene.mesh = AdaptiveRefinement::subdivideUniform(scene.mesh, refOpts);
    std::cout << "  Final (uniform): " << scene.mesh.numTriangles() << " triangles, "
              << scene.mesh.numVertices() << " vertices\n";

    // Save a copy before FF refinement so the viewer can toggle between them.
    scene.preRefinementMesh = scene.mesh;
    // Build triangle data for the pre-refinement mesh so it can be displayed.
    buildTriangleGeometry(scene.preRefinementMesh);
    buildMaterialData(scene.preRefinementMesh, scene.materials);

    // 2b) Form-factor-driven adaptive refinement.
    //     Iteratively subdivides patches where interacting triangles are
    //     too coarse relative to their distance (near-contact, corners).
    if (kEnableFFRefinement) {
        // Build triangle geometry now so FF refinement can use normals/centroids.
        buildTriangleGeometry(scene.mesh);

        std::cout << "\n=== Form-factor-driven refinement ===\n";
        FormFactorRefinement::Options ffOpts;
        ffOpts.accuracyRatio          = kFFRefinementAccuracyRatio;
        ffOpts.minFormFactor          = kFFRefinementMinFormFactor;
        ffOpts.minEdgeLength          = maxExtent * 0.001f;
        ffOpts.minArea                = scaledTargetArea * 0.01f;
        ffOpts.maxTriangles           = refOpts.maxTriangles;
        ffOpts.maxPasses              = kFFRefinementMaxPasses;
        ffOpts.splitAll3Edges         = kFFRefinementSplitAll3Edges;
        ffOpts.refineBoth             = true;
        ffOpts.searchRadiusMultiplier = 4.0f;

        scene.mesh = FormFactorRefinement::refineByFormFactor(scene.mesh, ffOpts);
        std::cout << "  Final (FF-refined): " << scene.mesh.numTriangles() << " triangles, "
                  << scene.mesh.numVertices() << " vertices\n";
    }

    // 3) Build triangle geometry.
    buildTriangleGeometry(scene.mesh);

    // 4) Build material data (reflectance + emission from MTL).
    buildMaterialData(scene.mesh, scene.materials);

    // 5) Ensure scene has light sources.
    ensureLight(scene);

    // If synthetic light was added, rebuild geometry for the new triangles.
    if (scene.mesh.triangle_area.size() < scene.mesh.numTriangles()) {
        buildTriangleGeometry(scene.mesh);
        buildMaterialData(scene.mesh, scene.materials);
    }

    // 6) Validate mesh.
    if (!validateMesh(scene.mesh)) {
        std::cerr << "Warning: mesh has invalid triangles\n";
    }

    // 7) Suggest camera placement.
    suggestCamera(scene);

    std::cout << "\n=== Scene ready ===\n";
    std::cout << "  Triangles : " << scene.mesh.numTriangles() << "\n";
    std::cout << "  Vertices  : " << scene.mesh.numVertices() << "\n";
    std::cout << "  Materials : " << scene.materials.size() << "\n";
    std::cout << "  Camera eye: ("
              << scene.cameraEye.x << ", " << scene.cameraEye.y << ", "
              << scene.cameraEye.z << ")\n";

    return scene;
}

} // namespace SceneLoader
