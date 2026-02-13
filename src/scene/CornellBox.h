#pragma once
#include "../mesh/MeshData.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include "../app/Config.h"
#include <cmath>
#include <algorithm>

namespace CornellBox {

// Front inner (short) box tuning.
inline constexpr float kShortBoxBaseHeight = 0.33f;
inline constexpr float kShortBoxBaseHalfSize = 0.165f;
inline constexpr float kShortBoxScale = 0.88f; // 12% smaller than base
inline constexpr float kShortBoxHeight = kShortBoxBaseHeight * kShortBoxScale;
inline constexpr float kShortBoxHalfSize = kShortBoxBaseHalfSize * kShortBoxScale;
inline constexpr float kShortBoxCenterX = 0.19f;
inline constexpr float kShortBoxCenterZ = 0.15f; // moved slightly backward
inline constexpr float kShortBoxRotYDeg = 18.0f;

// Rear inner (tall) box tuning.
inline constexpr float kTallBoxHeight = 0.6f;
inline constexpr float kTallBoxHalfSize = 0.1485f;
inline constexpr float kTallBoxCenterX = -0.14f;
inline constexpr float kTallBoxCenterZ = -0.2f;
inline constexpr float kTallBoxRotYDeg = -18.0f;

struct Material {
    Vec3 diffuse;
    Vec3 emission;
    
    Material() : diffuse(0.8f, 0.8f, 0.8f), emission(0, 0, 0) {}
    Material(const Vec3& d, const Vec3& e = Vec3(0, 0, 0)) : diffuse(d), emission(e) {}
};

// Material IDs
enum MaterialID : uint32_t {
    MAT_WHITE = 0,
    MAT_RED_WALL = 1,
    MAT_GREEN_WALL = 2,
    MAT_LIGHT = 3
};

// Assign material based on material ID
inline Material getMaterialForID(uint32_t matID) {
    // Cornell data-derived diffuse RGB (first-pass spectrum to RGB constants).
    // Reference RGB approximations (Cornell spectral data → sRGB).
    // Source: graphics.cornell.edu/online/box/data.html
    const Vec3 kWhite(0.73f, 0.73f, 0.73f);
    const Vec3 kRed(0.63f, 0.065f, 0.05f);
    const Vec3 kGreen(0.15f, 0.45f, 0.09f);
    const Vec3 kLightReflectance(0.780f, 0.780f, 0.780f);
    const Vec3 kLightEmission = Vec3(18.4f, 15.6f, 8.0f) * kLightBrightnessScale;

    switch (matID) {
        case MAT_LIGHT:
            return Material(kLightReflectance, kLightEmission);
        case MAT_RED_WALL:
            return Material(kRed, Vec3(0, 0, 0));
        case MAT_GREEN_WALL:
            return Material(kGreen, Vec3(0, 0, 0));
        default:
            // White surfaces (floor, ceiling, back wall, and inner boxes)
            return Material(kWhite, Vec3(0, 0, 0));
    }
}

// Determine material ID based on triangle centroid position
inline uint32_t getMaterialIDFromPosition(const Vec3& centroid) {
    const float eps = 0.01f;
    
    // Light area: center of ceiling, approximately 0.3x0.3 region
    // After subdivision, identify triangles in this region
    if (centroid.y > 0.48f && std::abs(centroid.x) < 0.15f && std::abs(centroid.z) < 0.15f) {
        return MAT_LIGHT;
    }
    
    // Check if it's the left wall (x ≈ -0.5)
    if (centroid.x < -0.49f) {
        return MAT_RED_WALL;
    }
    
    // Check if it's the right wall (x ≈ 0.5)
    if (centroid.x > 0.49f) {
        return MAT_GREEN_WALL;
    }
    
    // Everything else is white
    return MAT_WHITE;
}

// Create a procedural Cornell box (each wall is separate, no shared vertices)
inline Mesh createCornellBox() {
    Mesh mesh;
    
    // Cornell box dimensions (classic 1x1x1 box)
    const float size = 1.0f;
    const float half = size * 0.5f;
    
    std::vector<Vec3> positions;
    
    // Back wall (z = -half) - separate vertices
    uint32_t backStart = static_cast<uint32_t>(positions.size());
    positions.push_back(Vec3(-half, -half, -half)); // 0
    positions.push_back(Vec3( half, -half, -half)); // 1
    positions.push_back(Vec3( half,  half, -half)); // 2
    positions.push_back(Vec3(-half,  half, -half)); // 3
    mesh.indices.push_back(TriIdx(backStart+0, backStart+1, backStart+2));
    mesh.indices.push_back(TriIdx(backStart+0, backStart+2, backStart+3));
    
    // Floor (y = -half) - separate vertices
    uint32_t floorStart = static_cast<uint32_t>(positions.size());
    positions.push_back(Vec3(-half, -half,  half)); // 4
    positions.push_back(Vec3( half, -half,  half)); // 5
    positions.push_back(Vec3( half, -half, -half)); // 6
    positions.push_back(Vec3(-half, -half, -half)); // 7
    mesh.indices.push_back(TriIdx(floorStart+0, floorStart+1, floorStart+2));
    mesh.indices.push_back(TriIdx(floorStart+0, floorStart+2, floorStart+3));
    
    // Ceiling (y = +half) - separate vertices
    uint32_t ceilingStart = static_cast<uint32_t>(positions.size());
    positions.push_back(Vec3(-half,  half, -half)); // 8
    positions.push_back(Vec3( half,  half, -half)); // 9
    positions.push_back(Vec3( half,  half,  half)); // 10
    positions.push_back(Vec3(-half,  half,  half)); // 11
    mesh.indices.push_back(TriIdx(ceilingStart+0, ceilingStart+1, ceilingStart+2));
    mesh.indices.push_back(TriIdx(ceilingStart+0, ceilingStart+2, ceilingStart+3));
    
    // Left wall (x = -half) - RED - separate vertices
    uint32_t leftStart = static_cast<uint32_t>(positions.size());
    positions.push_back(Vec3(-half, -half,  half)); // 12
    positions.push_back(Vec3(-half, -half, -half)); // 13
    positions.push_back(Vec3(-half,  half, -half)); // 14
    positions.push_back(Vec3(-half,  half,  half)); // 15
    mesh.indices.push_back(TriIdx(leftStart+0, leftStart+1, leftStart+2));
    mesh.indices.push_back(TriIdx(leftStart+0, leftStart+2, leftStart+3));
    
    // Right wall (x = +half) - GREEN - separate vertices
    uint32_t rightStart = static_cast<uint32_t>(positions.size());
    positions.push_back(Vec3( half, -half, -half)); // 16
    positions.push_back(Vec3( half, -half,  half)); // 17
    positions.push_back(Vec3( half,  half,  half)); // 18
    positions.push_back(Vec3( half,  half, -half)); // 19
    mesh.indices.push_back(TriIdx(rightStart+0, rightStart+1, rightStart+2));
    mesh.indices.push_back(TriIdx(rightStart+0, rightStart+2, rightStart+3));
    
    
    // Short box (right side, rotated ~18 degrees clockwise) - each face separate
    const float shortHeight = kShortBoxHeight;
    const float shortSize = kShortBoxHalfSize;
    const float hover = -0.002f; // sink slightly below floor so OptiX occludes hidden surfaces
    const Vec3 shortCenter(kShortBoxCenterX, -half + shortHeight * 0.5f + hover, kShortBoxCenterZ);
    const float shortRotY = kShortBoxRotYDeg * 3.14159265f / 180.0f;
    
    auto rotatePoint = [](const Vec3& center, float x, float y, float z, float rotY) -> Vec3 {
        float x_rot = x * std::cos(rotY) - z * std::sin(rotY);
        float z_rot = x * std::sin(rotY) + z * std::cos(rotY);
        return Vec3(center.x + x_rot, center.y + y, center.z + z_rot);
    };
    
    // Short box - front face (separate vertices)
    uint32_t shortFrontStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortFrontStart+0, shortFrontStart+2, shortFrontStart+1));
    mesh.indices.push_back(TriIdx(shortFrontStart+0, shortFrontStart+3, shortFrontStart+2));
    
    // Short box - right face (separate vertices)
    uint32_t shortRightStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortRightStart+0, shortRightStart+2, shortRightStart+1));
    mesh.indices.push_back(TriIdx(shortRightStart+0, shortRightStart+3, shortRightStart+2));
    
    // Short box - back face (separate vertices)
    uint32_t shortBackStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortBackStart+0, shortBackStart+2, shortBackStart+1));
    mesh.indices.push_back(TriIdx(shortBackStart+0, shortBackStart+3, shortBackStart+2));
    
    // Short box - left face (separate vertices)
    uint32_t shortLeftStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortLeftStart+0, shortLeftStart+2, shortLeftStart+1));
    mesh.indices.push_back(TriIdx(shortLeftStart+0, shortLeftStart+3, shortLeftStart+2));
    
    // Short box - top face (separate vertices)
    uint32_t shortTopStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortTopStart+0, shortTopStart+2, shortTopStart+1));
    mesh.indices.push_back(TriIdx(shortTopStart+0, shortTopStart+3, shortTopStart+2));
    
    // Tall box (left side, rotated ~18 degrees counter-clockwise) - each face separate
    const float tallHeight = kTallBoxHeight;
    const float tallSize = kTallBoxHalfSize;
    const Vec3 tallCenter(kTallBoxCenterX, -half + tallHeight * 0.5f + hover, kTallBoxCenterZ);
    const float tallRotY = kTallBoxRotYDeg * 3.14159265f / 180.0f;
    
    // Tall box - front face (separate vertices)
    uint32_t tallFrontStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallFrontStart+0, tallFrontStart+2, tallFrontStart+1));
    mesh.indices.push_back(TriIdx(tallFrontStart+0, tallFrontStart+3, tallFrontStart+2));
    
    // Tall box - right face (separate vertices)
    uint32_t tallRightStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallRightStart+0, tallRightStart+2, tallRightStart+1));
    mesh.indices.push_back(TriIdx(tallRightStart+0, tallRightStart+3, tallRightStart+2));
    
    // Tall box - back face (separate vertices)
    uint32_t tallBackStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallBackStart+0, tallBackStart+2, tallBackStart+1));
    mesh.indices.push_back(TriIdx(tallBackStart+0, tallBackStart+3, tallBackStart+2));
    
    // Tall box - left face (separate vertices)
    uint32_t tallLeftStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallLeftStart+0, tallLeftStart+2, tallLeftStart+1));
    mesh.indices.push_back(TriIdx(tallLeftStart+0, tallLeftStart+3, tallLeftStart+2));
    
    // Tall box - top face (separate vertices)
    uint32_t tallTopStart = static_cast<uint32_t>(positions.size());
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallTopStart+0, tallTopStart+2, tallTopStart+1));
    mesh.indices.push_back(TriIdx(tallTopStart+0, tallTopStart+3, tallTopStart+2));
    
    // Convert to mesh vertices
    mesh.vertices.reserve(positions.size());
    for (const auto& p : positions) {
        mesh.vertices.push_back(Vertex(p));
    }
    
    // Assign material IDs based on triangle position
    mesh.triangle_material_id.resize(mesh.numTriangles());
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();
        Vec3 centroid = MathUtils::triangleCentroid(v0, v1, v2);
        mesh.triangle_material_id[i] = getMaterialIDFromPosition(centroid);
    }
    
    return mesh;
}

inline void fixNormalsOrientation(Mesh& mesh) {
    // Enforce consistent orientation after any procedural edits:
    // room surfaces face inward, inner box surfaces face outward.
    const float half = 0.5f;
    const Vec3 shortCenter(kShortBoxCenterX, -half + kShortBoxHeight * 0.5f, kShortBoxCenterZ);
    const Vec3 tallCenter(kTallBoxCenterX, -half + kTallBoxHeight * 0.5f, kTallBoxCenterZ);

    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        TriIdx& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();

        Vec3 centroid = MathUtils::triangleCentroid(v0, v1, v2);
        Vec3 normal = MathUtils::triangleNormal(v0, v1, v2).normalized();

        Vec3 desired(0.0f);
        bool isRoomSurface = false;
        if (centroid.x < -0.49f) { isRoomSurface = true; desired = Vec3(1, 0, 0); }
        else if (centroid.x > 0.49f) { isRoomSurface = true; desired = Vec3(-1, 0, 0); }
        else if (centroid.y < -0.49f) { isRoomSurface = true; desired = Vec3(0, 1, 0); }
        else if (centroid.y > 0.49f) { isRoomSurface = true; desired = Vec3(0, -1, 0); }
        else if (centroid.z < -0.49f) { isRoomSurface = true; desired = Vec3(0, 0, 1); }

        if (!isRoomSurface) {
            Vec3 toShort = centroid - shortCenter;
            Vec3 toTall = centroid - tallCenter;
            desired = (toShort.lengthSq() < toTall.lengthSq()) ? toShort.normalized() : toTall.normalized();
        }

        if (normal.dot(desired) < 0.0f) {
            std::swap(tri.i1, tri.i2);
        }
    }
}

} // namespace CornellBox
