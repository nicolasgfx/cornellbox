#pragma once
#include "Vec3.h"
#include <cmath>
#include <algorithm>

namespace MathUtils {

inline float clamp(float v, float minVal, float maxVal) {
    return std::max(minVal, std::min(maxVal, v));
}

inline Vec3 clamp(const Vec3& v, float minVal, float maxVal) {
    return Vec3(
        clamp(v.x, minVal, maxVal),
        clamp(v.y, minVal, maxVal),
        clamp(v.z, minVal, maxVal)
    );
}

// Triangle area from 3 vertices
inline float triangleArea(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    return e1.cross(e2).length() * 0.5f;
}

// Triangle normal (not normalized)
inline Vec3 triangleNormal(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    return e1.cross(e2);
}

// Triangle centroid
inline Vec3 triangleCentroid(const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    return (v0 + v1 + v2) / 3.0f;
}

// Check if triangle is degenerate
inline bool isTriangleDegenerate(const Vec3& v0, const Vec3& v1, const Vec3& v2, float eps = 1e-8f) {
    return triangleArea(v0, v1, v2) < eps;
}

} // namespace MathUtils
