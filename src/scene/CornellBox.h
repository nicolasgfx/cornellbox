#pragma once
#include "../mesh/MeshData.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include <iostream>
#include <cmath>

namespace CornellBox {

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
    switch (matID) {
        case MAT_LIGHT:
            // Increased emission for proper illumination
            return Material(Vec3(0.78f, 0.78f, 0.78f), Vec3(15.0f, 15.0f, 15.0f));
        case MAT_RED_WALL:
            return Material(Vec3(0.75f, 0.25f, 0.25f), Vec3(0, 0, 0));
        case MAT_GREEN_WALL:
            return Material(Vec3(0.25f, 0.75f, 0.25f), Vec3(0, 0, 0));
        default:
            // White surfaces (floor, ceiling, back wall, and inner boxes)
            return Material(Vec3(0.7f, 0.7f, 0.7f), Vec3(0, 0, 0));
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
    size_t backStart = positions.size();
    positions.push_back(Vec3(-half, -half, -half)); // 0
    positions.push_back(Vec3( half, -half, -half)); // 1
    positions.push_back(Vec3( half,  half, -half)); // 2
    positions.push_back(Vec3(-half,  half, -half)); // 3
    mesh.indices.push_back(TriIdx(backStart+0, backStart+1, backStart+2));
    mesh.indices.push_back(TriIdx(backStart+0, backStart+2, backStart+3));
    
    // Floor (y = -half) - separate vertices
    size_t floorStart = positions.size();
    positions.push_back(Vec3(-half, -half,  half)); // 4
    positions.push_back(Vec3( half, -half,  half)); // 5
    positions.push_back(Vec3( half, -half, -half)); // 6
    positions.push_back(Vec3(-half, -half, -half)); // 7
    mesh.indices.push_back(TriIdx(floorStart+0, floorStart+1, floorStart+2));
    mesh.indices.push_back(TriIdx(floorStart+0, floorStart+2, floorStart+3));
    
    // Ceiling (y = +half) - separate vertices
    size_t ceilingStart = positions.size();
    positions.push_back(Vec3(-half,  half, -half)); // 8
    positions.push_back(Vec3( half,  half, -half)); // 9
    positions.push_back(Vec3( half,  half,  half)); // 10
    positions.push_back(Vec3(-half,  half,  half)); // 11
    mesh.indices.push_back(TriIdx(ceilingStart+0, ceilingStart+1, ceilingStart+2));
    mesh.indices.push_back(TriIdx(ceilingStart+0, ceilingStart+2, ceilingStart+3));
    
    // Left wall (x = -half) - RED - separate vertices
    size_t leftStart = positions.size();
    positions.push_back(Vec3(-half, -half,  half)); // 12
    positions.push_back(Vec3(-half, -half, -half)); // 13
    positions.push_back(Vec3(-half,  half, -half)); // 14
    positions.push_back(Vec3(-half,  half,  half)); // 15
    mesh.indices.push_back(TriIdx(leftStart+0, leftStart+1, leftStart+2));
    mesh.indices.push_back(TriIdx(leftStart+0, leftStart+2, leftStart+3));
    
    // Right wall (x = +half) - GREEN - separate vertices
    size_t rightStart = positions.size();
    positions.push_back(Vec3( half, -half, -half)); // 16
    positions.push_back(Vec3( half, -half,  half)); // 17
    positions.push_back(Vec3( half,  half,  half)); // 18
    positions.push_back(Vec3( half,  half, -half)); // 19
    mesh.indices.push_back(TriIdx(rightStart+0, rightStart+1, rightStart+2));
    mesh.indices.push_back(TriIdx(rightStart+0, rightStart+2, rightStart+3));
    
    
    // Short box (right side, rotated ~18 degrees clockwise) - each face separate
    const float shortHeight = 0.33f;
    const float shortSize = 0.165f;
    const Vec3 shortCenter(0.19f, -half + shortHeight * 0.5f, 0.2f);
    const float shortRotY = -18.0f * 3.14159265f / 180.0f;
    
    auto rotatePoint = [](const Vec3& center, float x, float y, float z, float rotY) -> Vec3 {
        float x_rot = x * std::cos(rotY) - z * std::sin(rotY);
        float z_rot = x * std::sin(rotY) + z * std::cos(rotY);
        return Vec3(center.x + x_rot, center.y + y, center.z + z_rot);
    };
    
    // Short box - front face (separate vertices)
    size_t shortFrontStart = positions.size();
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortFrontStart+1, shortFrontStart+3, shortFrontStart+0));
    mesh.indices.push_back(TriIdx(shortFrontStart+1, shortFrontStart+2, shortFrontStart+3));
    
    // Short box - right face (separate vertices)
    size_t shortRightStart = positions.size();
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortRightStart+1, shortRightStart+3, shortRightStart+0));
    mesh.indices.push_back(TriIdx(shortRightStart+1, shortRightStart+2, shortRightStart+3));
    
    // Short box - back face (separate vertices)
    size_t shortBackStart = positions.size();
    positions.push_back(rotatePoint(shortCenter,  shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortBackStart+1, shortBackStart+3, shortBackStart+0));
    mesh.indices.push_back(TriIdx(shortBackStart+1, shortBackStart+2, shortBackStart+3));
    
    // Short box - left face (separate vertices)
    size_t shortLeftStart = positions.size();
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize, -shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortLeftStart+1, shortLeftStart+3, shortLeftStart+0));
    mesh.indices.push_back(TriIdx(shortLeftStart+1, shortLeftStart+2, shortLeftStart+3));
    
    // Short box - top face (separate vertices)
    size_t shortTopStart = positions.size();
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f, -shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter,  shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    positions.push_back(rotatePoint(shortCenter, -shortSize,  shortHeight * 0.5f,  shortSize, shortRotY));
    mesh.indices.push_back(TriIdx(shortTopStart+0, shortTopStart+2, shortTopStart+3));
    mesh.indices.push_back(TriIdx(shortTopStart+0, shortTopStart+1, shortTopStart+2));
    
    // Tall box (left side, rotated ~18 degrees counter-clockwise) - each face separate
    const float tallHeight = 0.6f;
    const float tallSize = 0.165f;
    const Vec3 tallCenter(-0.19f, -half + tallHeight * 0.5f, -0.2f);
    const float tallRotY = 18.0f * 3.14159265f / 180.0f;
    
    // Tall box - front face (separate vertices)
    size_t tallFrontStart = positions.size();
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallFrontStart+1, tallFrontStart+3, tallFrontStart+0));
    mesh.indices.push_back(TriIdx(tallFrontStart+1, tallFrontStart+2, tallFrontStart+3));
    
    // Tall box - right face (separate vertices)
    size_t tallRightStart = positions.size();
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallRightStart+1, tallRightStart+3, tallRightStart+0));
    mesh.indices.push_back(TriIdx(tallRightStart+1, tallRightStart+2, tallRightStart+3));
    
    // Tall box - back face (separate vertices)
    size_t tallBackStart = positions.size();
    positions.push_back(rotatePoint(tallCenter,  tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallBackStart+1, tallBackStart+3, tallBackStart+0));
    mesh.indices.push_back(TriIdx(tallBackStart+1, tallBackStart+2, tallBackStart+3));
    
    // Tall box - left face (separate vertices)
    size_t tallLeftStart = positions.size();
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize, -tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallLeftStart+1, tallLeftStart+3, tallLeftStart+0));
    mesh.indices.push_back(TriIdx(tallLeftStart+1, tallLeftStart+2, tallLeftStart+3));
    
    // Tall box - top face (separate vertices)
    size_t tallTopStart = positions.size();
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f, -tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter,  tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    positions.push_back(rotatePoint(tallCenter, -tallSize,  tallHeight * 0.5f,  tallSize, tallRotY));
    mesh.indices.push_back(TriIdx(tallTopStart+0, tallTopStart+2, tallTopStart+3));
    mesh.indices.push_back(TriIdx(tallTopStart+0, tallTopStart+1, tallTopStart+2));
    
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
    // Normals are already correct by construction
}

} // namespace CornellBox
