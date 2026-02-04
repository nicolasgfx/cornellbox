#pragma once

#include <cmath>
#include <iostream>

namespace radiosity {
namespace math {

/**
 * 3D Vector class for geometric computations
 * Simple, debuggable implementation with all basic operations
 */
class Vector3 {
public:
    float x, y, z;

    // Constructors
    Vector3() : x(0), y(0), z(0) {}
    Vector3(float x, float y, float z) : x(x), y(y), z(z) {}
    Vector3(float scalar) : x(scalar), y(scalar), z(scalar) {}

    // Accessors
    float& operator[](int i) {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    const float& operator[](int i) const {
        if (i == 0) return x;
        if (i == 1) return y;
        return z;
    }

    // Vector addition
    Vector3 operator+(const Vector3& v) const {
        return Vector3(x + v.x, y + v.y, z + v.z);
    }

    Vector3& operator+=(const Vector3& v) {
        x += v.x; y += v.y; z += v.z;
        return *this;
    }

    // Vector subtraction
    Vector3 operator-(const Vector3& v) const {
        return Vector3(x - v.x, y - v.y, z - v.z);
    }

    Vector3& operator-=(const Vector3& v) {
        x -= v.x; y -= v.y; z -= v.z;
        return *this;
    }

    // Scalar multiplication
    Vector3 operator*(float s) const {
        return Vector3(x * s, y * s, z * s);
    }

    Vector3& operator*=(float s) {
        x *= s; y *= s; z *= s;
        return *this;
    }

    // Component-wise multiplication
    Vector3 operator*(const Vector3& v) const {
        return Vector3(x * v.x, y * v.y, z * v.z);
    }

    Vector3& operator*=(const Vector3& v) {
        x *= v.x; y *= v.y; z *= v.z;
        return *this;
    }

    // Scalar division
    Vector3 operator/(float s) const {
        float inv = 1.0f / s;
        return Vector3(x * inv, y * inv, z * inv);
    }

    Vector3& operator/=(float s) {
        float inv = 1.0f / s;
        x *= inv; y *= inv; z *= inv;
        return *this;
    }

    // Component-wise division
    Vector3 operator/(const Vector3& v) const {
        return Vector3(x / v.x, y / v.y, z / v.z);
    }

    // Negation
    Vector3 operator-() const {
        return Vector3(-x, -y, -z);
    }

    // Comparison
    bool operator==(const Vector3& v) const {
        return x == v.x && y == v.y && z == v.z;
    }

    bool operator!=(const Vector3& v) const {
        return !(*this == v);
    }

    // Dot product
    float dot(const Vector3& v) const {
        return x * v.x + y * v.y + z * v.z;
    }

    // Cross product
    Vector3 cross(const Vector3& v) const {
        return Vector3(
            y * v.z - z * v.y,
            z * v.x - x * v.z,
            x * v.y - y * v.x
        );
    }

    // Length operations
    float lengthSquared() const {
        return x * x + y * y + z * z;
    }

    float length() const {
        return std::sqrt(lengthSquared());
    }

    // Normalization
    Vector3 normalized() const {
        float len = length();
        if (len < 1e-8f) {
            // DEBUG: Warn about zero-length vector
            std::cerr << "Warning: Attempting to normalize zero-length vector\n";
            return Vector3(0, 0, 0);
        }
        return *this / len;
    }

    void normalize() {
        float len = length();
        if (len < 1e-8f) {
            std::cerr << "Warning: Attempting to normalize zero-length vector\n";
            x = y = z = 0;
            return;
        }
        *this /= len;
    }

    // Distance operations
    float distanceSquared(const Vector3& v) const {
        return (*this - v).lengthSquared();
    }

    float distance(const Vector3& v) const {
        return (*this - v).length();
    }

    // Utility: Check if vector is zero
    bool isZero() const {
        return x == 0 && y == 0 && z == 0;
    }

    // Utility: Check if vector is approximately zero
    bool isNearZero(float epsilon = 1e-6f) const {
        return std::abs(x) < epsilon && std::abs(y) < epsilon && std::abs(z) < epsilon;
    }

    // Utility: Clamp components
    Vector3 clamped(float minVal, float maxVal) const {
        return Vector3(
            std::max(minVal, std::min(maxVal, x)),
            std::max(minVal, std::min(maxVal, y)),
            std::max(minVal, std::min(maxVal, z))
        );
    }

    // Debug output
    void print(const char* label = nullptr) const {
        if (label) {
            std::cout << label << ": ";
        }
        std::cout << "(" << x << ", " << y << ", " << z << ")\n";
    }

    // Common vector constants
    static Vector3 zero() { return Vector3(0, 0, 0); }
    static Vector3 one() { return Vector3(1, 1, 1); }
    static Vector3 unitX() { return Vector3(1, 0, 0); }
    static Vector3 unitY() { return Vector3(0, 1, 0); }
    static Vector3 unitZ() { return Vector3(0, 0, 1); }
};

// External operator for scalar * vector
inline Vector3 operator*(float s, const Vector3& v) {
    return v * s;
}

// Stream output
inline std::ostream& operator<<(std::ostream& os, const Vector3& v) {
    os << "(" << v.x << ", " << v.y << ", " << v.z << ")";
    return os;
}

// Common operations as free functions (for clarity)
inline float dot(const Vector3& a, const Vector3& b) {
    return a.dot(b);
}

inline Vector3 cross(const Vector3& a, const Vector3& b) {
    return a.cross(b);
}

inline Vector3 normalize(const Vector3& v) {
    return v.normalized();
}

// Linear interpolation
inline Vector3 lerp(const Vector3& a, const Vector3& b, float t) {
    return a * (1.0f - t) + b * t;
}

// Reflect vector v across normal n (assumes n is normalized)
inline Vector3 reflect(const Vector3& v, const Vector3& n) {
    return v - 2.0f * dot(v, n) * n;
}

// Component-wise min/max
inline Vector3 min(const Vector3& a, const Vector3& b) {
    return Vector3(
        std::min(a.x, b.x),
        std::min(a.y, b.y),
        std::min(a.z, b.z)
    );
}

inline Vector3 max(const Vector3& a, const Vector3& b) {
    return Vector3(
        std::max(a.x, b.x),
        std::max(a.y, b.y),
        std::max(a.z, b.z)
    );
}

} // namespace math
} // namespace radiosity
