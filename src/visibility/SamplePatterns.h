#pragma once
#include <vector>
#include <cmath>

namespace SamplePatterns {

// Barycentric coordinate pair (u, v) for triangle sampling
struct BaryCoord {
    float u, v;
};

// Generate stratified barycentric sample pattern
// Uses uniform grid + reflection for u+v>1
inline std::vector<BaryCoord> generateStratified(int sqrtSamples) {
    std::vector<BaryCoord> samples;
    samples.reserve(sqrtSamples * sqrtSamples);
    
    float inv = 1.0f / sqrtSamples;
    for (int i = 0; i < sqrtSamples; ++i) {
        for (int j = 0; j < sqrtSamples; ++j) {
            float u = (i + 0.5f) * inv;
            float v = (j + 0.5f) * inv;
            
            // Reflect if outside triangle
            if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
            }
            
            samples.push_back({u, v});
        }
    }
    
    return samples;
}

// Precomputed patterns for common sample counts
inline std::vector<BaryCoord> getPattern4() {
    return generateStratified(2);
}

inline std::vector<BaryCoord> getPattern16() {
    return generateStratified(4);
}

inline std::vector<BaryCoord> getPattern64() {
    return generateStratified(8);
}

// Get pattern by sample count
inline std::vector<BaryCoord> getPattern(int samples) {
    if (samples <= 4) return getPattern4();
    if (samples <= 16) return getPattern16();
    return getPattern64();
}

// Print pattern as CUDA constant array (for code generation)
inline void printCUDAConstant(const std::vector<BaryCoord>& pattern, const char* name) {
    printf("__constant__ float2 %s[%zu] = {\n", name, pattern.size());
    for (size_t i = 0; i < pattern.size(); ++i) {
        printf("    {%.6ff, %.6ff}%s\n", 
               pattern[i].u, pattern[i].v,
               (i + 1 < pattern.size()) ? "," : "");
    }
    printf("};\n");
}

} // namespace SamplePatterns
