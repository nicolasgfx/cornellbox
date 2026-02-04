#pragma once

#include <cmath>
#include <random>
#include <algorithm>

namespace radiosity {
namespace math {

// Mathematical constants
constexpr float PI = 3.14159265358979323846f;
constexpr float TWO_PI = 2.0f * PI;
constexpr float HALF_PI = 0.5f * PI;
constexpr float INV_PI = 1.0f / PI;
constexpr float INV_TWO_PI = 1.0f / TWO_PI;
constexpr float INV_FOUR_PI = 1.0f / (4.0f * PI);

constexpr float EPSILON = 1e-6f;
constexpr float SHADOW_EPSILON = 1e-4f;  // For ray offsetting

constexpr float DEG_TO_RAD = PI / 180.0f;
constexpr float RAD_TO_DEG = 180.0f / PI;

// Clamp a value between min and max
template<typename T>
inline T clamp(T value, T minVal, T maxVal) {
    return std::max(minVal, std::min(maxVal, value));
}

// Linear interpolation
template<typename T>
inline T lerp(T a, T b, float t) {
    return a * (1.0f - t) + b * t;
}

// Smooth interpolation (cubic Hermite)
inline float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

// Smoother interpolation (quintic)
inline float smootherstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

// Angle conversions
inline float radians(float degrees) {
    return degrees * DEG_TO_RAD;
}

inline float degrees(float radians) {
    return radians * RAD_TO_DEG;
}

// Floating point comparison with epsilon
inline bool nearEqual(float a, float b, float epsilon = EPSILON) {
    return std::abs(a - b) < epsilon;
}

inline bool nearZero(float value, float epsilon = EPSILON) {
    return std::abs(value) < epsilon;
}

// Safe division (returns 0 if denominator is near zero)
inline float safeDivide(float numerator, float denominator, float epsilon = EPSILON) {
    if (std::abs(denominator) < epsilon) {
        return 0.0f;
    }
    return numerator / denominator;
}

// Square function
template<typename T>
inline T sqr(T x) {
    return x * x;
}

// Sign function (-1, 0, or 1)
template<typename T>
inline T sign(T x) {
    if (x > 0) return T(1);
    if (x < 0) return T(-1);
    return T(0);
}

// Random number generation utilities
class Random {
private:
    static std::mt19937& getGenerator() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        return gen;
    }

public:
    // Random float in [0, 1)
    static float uniform01() {
        static std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        return dist(getGenerator());
    }

    // Random float in [min, max)
    static float uniformRange(float minVal, float maxVal) {
        return minVal + (maxVal - minVal) * uniform01();
    }

    // Random integer in [min, max]
    static int uniformInt(int minVal, int maxVal) {
        std::uniform_int_distribution<int> dist(minVal, maxVal);
        return dist(getGenerator());
    }

    // Set seed for reproducibility (useful for debugging)
    static void seed(unsigned int s) {
        getGenerator().seed(s);
    }

    // Random point on unit hemisphere (cosine-weighted for radiosity)
    // Returns direction in local coordinate system (z-up)
    static void cosineWeightedHemisphere(float& x, float& y, float& z) {
        float r1 = uniform01();
        float r2 = uniform01();
        
        float phi = TWO_PI * r1;
        float cosTheta = std::sqrt(r2);
        float sinTheta = std::sqrt(1.0f - r2);
        
        x = std::cos(phi) * sinTheta;
        y = std::sin(phi) * sinTheta;
        z = cosTheta;
    }

    // Random point on unit sphere (uniform)
    static void uniformSphere(float& x, float& y, float& z) {
        float r1 = uniform01();
        float r2 = uniform01();
        
        float phi = TWO_PI * r1;
        float cosTheta = 2.0f * r2 - 1.0f;
        float sinTheta = std::sqrt(1.0f - cosTheta * cosTheta);
        
        x = std::cos(phi) * sinTheta;
        y = std::sin(phi) * sinTheta;
        z = cosTheta;
    }

    // Random point in unit disk
    static void uniformDisk(float& x, float& y) {
        float r1 = uniform01();
        float r2 = uniform01();
        
        float r = std::sqrt(r1);
        float theta = TWO_PI * r2;
        
        x = r * std::cos(theta);
        y = r * std::sin(theta);
    }
};

// Color space conversions (linear to sRGB gamma)
inline float linearToSRGB(float linear) {
    if (linear <= 0.0031308f) {
        return 12.92f * linear;
    } else {
        return 1.055f * std::pow(linear, 1.0f / 2.4f) - 0.055f;
    }
}

inline float sRGBToLinear(float srgb) {
    if (srgb <= 0.04045f) {
        return srgb / 12.92f;
    } else {
        return std::pow((srgb + 0.055f) / 1.055f, 2.4f);
    }
}

// Simple gamma correction
inline float gammaCorrect(float linear, float gamma = 2.2f) {
    return std::pow(linear, 1.0f / gamma);
}

// Tone mapping operators
inline float reinhardToneMap(float hdr) {
    return hdr / (1.0f + hdr);
}

inline float reinhardToneMapExtended(float hdr, float maxWhite) {
    float numerator = hdr * (1.0f + (hdr / (maxWhite * maxWhite)));
    return numerator / (1.0f + hdr);
}

// ACES filmic tone mapping (approximation)
inline float acesToneMap(float x) {
    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0f, 1.0f);
}

// Solve quadratic equation ax^2 + bx + c = 0
// Returns true if real solutions exist, and fills t0, t1 (t0 <= t1)
inline bool solveQuadratic(float a, float b, float c, float& t0, float& t1) {
    float discriminant = b * b - 4.0f * a * c;
    if (discriminant < 0.0f) {
        return false;
    }
    
    float sqrtDisc = std::sqrt(discriminant);
    float q = (b < 0.0f) ? -0.5f * (b - sqrtDisc) : -0.5f * (b + sqrtDisc);
    
    t0 = q / a;
    t1 = c / q;
    
    if (t0 > t1) {
        std::swap(t0, t1);
    }
    
    return true;
}

// Power heuristic for multiple importance sampling (MIS)
inline float powerHeuristic(int nf, float fPdf, int ng, float gPdf) {
    float f = nf * fPdf;
    float g = ng * gPdf;
    return (f * f) / (f * f + g * g);
}

// Balance heuristic for MIS
inline float balanceHeuristic(int nf, float fPdf, int ng, float gPdf) {
    return (nf * fPdf) / (nf * fPdf + ng * gPdf);
}

} // namespace math
} // namespace radiosity
