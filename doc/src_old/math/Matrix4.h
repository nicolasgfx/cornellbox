#pragma once

#include "Vector3.h"
#include <cmath>
#include <iostream>

namespace radiosity {
namespace math {

/**
 * 4x4 Matrix for transformations
 * Column-major order (OpenGL style)
 */
class Matrix4 {
public:
    float m[16];  // Column-major: m[column * 4 + row]

    // Constructors
    Matrix4() {
        // Identity matrix
        for (int i = 0; i < 16; i++) m[i] = 0.0f;
        m[0] = m[5] = m[10] = m[15] = 1.0f;
    }

    // Access element at (row, col)
    float& at(int row, int col) {
        return m[col * 4 + row];
    }

    const float& at(int row, int col) const {
        return m[col * 4 + row];
    }

    // Array access (for passing to OpenGL)
    float* data() { return m; }
    const float* data() const { return m; }

    // Matrix multiplication
    Matrix4 operator*(const Matrix4& other) const {
        Matrix4 result;
        for (int col = 0; col < 4; col++) {
            for (int row = 0; row < 4; row++) {
                float sum = 0.0f;
                for (int k = 0; k < 4; k++) {
                    sum += at(row, k) * other.at(k, col);
                }
                result.at(row, col) = sum;
            }
        }
        return result;
    }

    Matrix4& operator*=(const Matrix4& other) {
        *this = *this * other;
        return *this;
    }

    // Transform a point (w=1)
    Vector3 transformPoint(const Vector3& p) const {
        float x = at(0, 0) * p.x + at(0, 1) * p.y + at(0, 2) * p.z + at(0, 3);
        float y = at(1, 0) * p.x + at(1, 1) * p.y + at(1, 2) * p.z + at(1, 3);
        float z = at(2, 0) * p.x + at(2, 1) * p.y + at(2, 2) * p.z + at(2, 3);
        float w = at(3, 0) * p.x + at(3, 1) * p.y + at(3, 2) * p.z + at(3, 3);
        
        if (std::abs(w) > 1e-6f) {
            return Vector3(x / w, y / w, z / w);
        }
        return Vector3(x, y, z);
    }

    // Transform a direction (w=0)
    Vector3 transformDirection(const Vector3& d) const {
        float x = at(0, 0) * d.x + at(0, 1) * d.y + at(0, 2) * d.z;
        float y = at(1, 0) * d.x + at(1, 1) * d.y + at(1, 2) * d.z;
        float z = at(2, 0) * d.x + at(2, 1) * d.y + at(2, 2) * d.z;
        return Vector3(x, y, z);
    }

    // Static factory methods
    static Matrix4 identity() {
        return Matrix4();
    }

    static Matrix4 translation(float x, float y, float z) {
        Matrix4 mat;
        mat.at(0, 3) = x;
        mat.at(1, 3) = y;
        mat.at(2, 3) = z;
        return mat;
    }

    static Matrix4 translation(const Vector3& v) {
        return translation(v.x, v.y, v.z);
    }

    static Matrix4 scale(float x, float y, float z) {
        Matrix4 mat;
        mat.at(0, 0) = x;
        mat.at(1, 1) = y;
        mat.at(2, 2) = z;
        return mat;
    }

    static Matrix4 scale(float s) {
        return scale(s, s, s);
    }

    static Matrix4 scale(const Vector3& v) {
        return scale(v.x, v.y, v.z);
    }

    // Rotation around X axis
    static Matrix4 rotationX(float angleRadians) {
        Matrix4 mat;
        float c = std::cos(angleRadians);
        float s = std::sin(angleRadians);
        mat.at(1, 1) = c;
        mat.at(1, 2) = -s;
        mat.at(2, 1) = s;
        mat.at(2, 2) = c;
        return mat;
    }

    // Rotation around Y axis
    static Matrix4 rotationY(float angleRadians) {
        Matrix4 mat;
        float c = std::cos(angleRadians);
        float s = std::sin(angleRadians);
        mat.at(0, 0) = c;
        mat.at(0, 2) = s;
        mat.at(2, 0) = -s;
        mat.at(2, 2) = c;
        return mat;
    }

    // Rotation around Z axis
    static Matrix4 rotationZ(float angleRadians) {
        Matrix4 mat;
        float c = std::cos(angleRadians);
        float s = std::sin(angleRadians);
        mat.at(0, 0) = c;
        mat.at(0, 1) = -s;
        mat.at(1, 0) = s;
        mat.at(1, 1) = c;
        return mat;
    }

    // Rotation around arbitrary axis (assumes axis is normalized)
    static Matrix4 rotation(const Vector3& axis, float angleRadians) {
        Matrix4 mat;
        float c = std::cos(angleRadians);
        float s = std::sin(angleRadians);
        float t = 1.0f - c;
        
        float x = axis.x, y = axis.y, z = axis.z;
        
        mat.at(0, 0) = t * x * x + c;
        mat.at(0, 1) = t * x * y - s * z;
        mat.at(0, 2) = t * x * z + s * y;
        
        mat.at(1, 0) = t * x * y + s * z;
        mat.at(1, 1) = t * y * y + c;
        mat.at(1, 2) = t * y * z - s * x;
        
        mat.at(2, 0) = t * x * z - s * y;
        mat.at(2, 1) = t * y * z + s * x;
        mat.at(2, 2) = t * z * z + c;
        
        return mat;
    }

    // Look-at matrix (for camera/view transforms)
    static Matrix4 lookAt(const Vector3& eye, const Vector3& target, const Vector3& up) {
        Vector3 zAxis = (eye - target).normalized();  // Forward
        Vector3 xAxis = cross(up, zAxis).normalized(); // Right
        Vector3 yAxis = cross(zAxis, xAxis);           // Up

        Matrix4 mat;
        mat.at(0, 0) = xAxis.x;
        mat.at(0, 1) = xAxis.y;
        mat.at(0, 2) = xAxis.z;
        mat.at(0, 3) = -dot(xAxis, eye);

        mat.at(1, 0) = yAxis.x;
        mat.at(1, 1) = yAxis.y;
        mat.at(1, 2) = yAxis.z;
        mat.at(1, 3) = -dot(yAxis, eye);

        mat.at(2, 0) = zAxis.x;
        mat.at(2, 1) = zAxis.y;
        mat.at(2, 2) = zAxis.z;
        mat.at(2, 3) = -dot(zAxis, eye);

        mat.at(3, 0) = 0.0f;
        mat.at(3, 1) = 0.0f;
        mat.at(3, 2) = 0.0f;
        mat.at(3, 3) = 1.0f;

        return mat;
    }

    // Perspective projection matrix
    static Matrix4 perspective(float fovYRadians, float aspect, float near, float far) {
        Matrix4 mat;
        for (int i = 0; i < 16; i++) mat.m[i] = 0.0f;

        float tanHalfFovy = std::tan(fovYRadians / 2.0f);
        
        mat.at(0, 0) = 1.0f / (aspect * tanHalfFovy);
        mat.at(1, 1) = 1.0f / tanHalfFovy;
        mat.at(2, 2) = -(far + near) / (far - near);
        mat.at(2, 3) = -(2.0f * far * near) / (far - near);
        mat.at(3, 2) = -1.0f;

        return mat;
    }

    // Orthographic projection matrix
    static Matrix4 ortho(float left, float right, float bottom, float top, float near, float far) {
        Matrix4 mat;
        
        mat.at(0, 0) = 2.0f / (right - left);
        mat.at(1, 1) = 2.0f / (top - bottom);
        mat.at(2, 2) = -2.0f / (far - near);
        
        mat.at(0, 3) = -(right + left) / (right - left);
        mat.at(1, 3) = -(top + bottom) / (top - bottom);
        mat.at(2, 3) = -(far + near) / (far - near);

        return mat;
    }

    // Transpose
    Matrix4 transposed() const {
        Matrix4 result;
        for (int row = 0; row < 4; row++) {
            for (int col = 0; col < 4; col++) {
                result.at(row, col) = at(col, row);
            }
        }
        return result;
    }

    // Debug output
    void print(const char* label = nullptr) const {
        if (label) {
            std::cout << label << ":\n";
        }
        for (int row = 0; row < 4; row++) {
            std::cout << "  [ ";
            for (int col = 0; col < 4; col++) {
                printf("%8.4f ", at(row, col));
            }
            std::cout << "]\n";
        }
    }
};

} // namespace math
} // namespace radiosity
