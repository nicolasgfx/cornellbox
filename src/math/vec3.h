#pragma once
#include <cmath>
#include <algorithm>

struct Vec3 {
    float x, y, z;

    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vec3(float v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& v) const { return Vec3(x + v.x, y + v.y, z + v.z); }
    Vec3 operator-(const Vec3& v) const { return Vec3(x - v.x, y - v.y, z - v.z); }
    Vec3 operator*(float s) const { return Vec3(x * s, y * s, z * s); }
    Vec3 operator/(float s) const { return Vec3(x / s, y / s, z / s); }
    
    Vec3& operator+=(const Vec3& v) { x += v.x; y += v.y; z += v.z; return *this; }
    Vec3& operator-=(const Vec3& v) { x -= v.x; y -= v.y; z -= v.z; return *this; }
    Vec3& operator*=(float s) { x *= s; y *= s; z *= s; return *this; }

    float dot(const Vec3& v) const { return x * v.x + y * v.y + z * v.z; }
    
    Vec3 cross(const Vec3& v) const {
        return Vec3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    float length() const { return std::sqrt(x * x + y * y + z * z); }
    float lengthSq() const { return x * x + y * y + z * z; }
    
    Vec3 normalized() const {
        float len = length();
        return len > 1e-8f ? (*this / len) : Vec3(0, 0, 0);
    }

    void normalize() {
        float len = length();
        if (len > 1e-8f) {
            x /= len; y /= len; z /= len;
        }
    }

    bool isZero() const { return std::abs(x) < 1e-8f && std::abs(y) < 1e-8f && std::abs(z) < 1e-8f; }
    bool hasNaN() const { return std::isnan(x) || std::isnan(y) || std::isnan(z); }
};

inline Vec3 operator*(float s, const Vec3& v) { return v * s; }
