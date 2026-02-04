#pragma once
#include <cstdint>
#include <vector>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>

namespace VisibilityCache {

// Magic number for file validation
constexpr uint32_t MAGIC = 0x56495330; // "VIS0"
constexpr uint32_t VERSION = 1;

// Cache file header
struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t numTriangles;
    uint32_t samplesPerPair;
    uint32_t padding; // alignment
    
    bool isValid() const {
        return magic == MAGIC && version == VERSION;
    }
};

// Build cache filename
inline std::string getCacheFilename(const std::string& profile, int samples) {
    return "output/cache/visibility_" + profile + "_s" + std::to_string(samples) + ".bin";
}

// Save visibility matrix to disk as float32
// Matrix is symmetric but we only store upper triangle (i < j)
inline bool save(const std::string& filename, 
                 const std::vector<float>& visibility,
                 uint32_t numTriangles,
                 uint32_t samplesPerPair) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open cache file for writing: " << filename << "\n";
        return false;
    }
    
    Header header;
    header.magic = MAGIC;
    header.version = VERSION;
    header.numTriangles = numTriangles;
    header.samplesPerPair = samplesPerPair;
    header.padding = 0;
    
    file.write(reinterpret_cast<const char*>(&header), sizeof(Header));
    file.write(reinterpret_cast<const char*>(visibility.data()), visibility.size() * sizeof(float));
    
    bool success = file.good();
    if (success) {
        std::cout << "Successfully saved visibility cache: " << filename << " (" << visibility.size() << " pairs)\n";
    } else {
        std::cerr << "Failed to write visibility cache: " << filename << "\n";
    }
    return success;
}

// Load visibility matrix from disk as float32
inline bool load(const std::string& filename,
                 std::vector<float>& visibility,
                 uint32_t& numTriangles,
                 uint32_t& samplesPerPair) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cout << "Cache file not found: " << filename << "\n";
        return false;
    }
    
    Header header;
    file.read(reinterpret_cast<char*>(&header), sizeof(Header));
    
    if (!header.isValid()) {
        std::cerr << "Invalid cache file header: " << filename << "\n";
        return false;
    }
    
    numTriangles = header.numTriangles;
    samplesPerPair = header.samplesPerPair;
    
    // Calculate expected size for upper triangle matrix
    size_t numPairs = (size_t(numTriangles) * (numTriangles - 1)) / 2;
    visibility.resize(numPairs);
    
    file.read(reinterpret_cast<char*>(visibility.data()), numPairs * sizeof(float));
    
    bool success = file.good();
    if (success) {
        std::cout << "Successfully loaded visibility cache: " << filename << " (" << numPairs << " pairs)\n";
    } else {
        std::cerr << "Failed to read visibility data from: " << filename << "\n";
    }
    return success;
}

// Get index for pair (i, j) where i < j
inline size_t getPairIndex(uint32_t i, uint32_t j, uint32_t numTriangles) {
    if (i >= j) std::swap(i, j); // ensure i < j
    // Index in upper triangle: sum of previous rows + offset in current row
    return (size_t(i) * (2 * numTriangles - i - 1)) / 2 + (j - i - 1);
}

// Get visibility value for pair (i, j)
inline float getVisibility(const std::vector<float>& visibility, 
                           uint32_t i, uint32_t j, 
                           uint32_t numTriangles) {
    if (i == j) return 1.0f; // self-visibility
    size_t idx = getPairIndex(i, j, numTriangles);
    return visibility[idx];
}

// Set visibility value for pair (i, j)
inline void setVisibility(std::vector<float>& visibility,
                         uint32_t i, uint32_t j,
                         uint32_t numTriangles,
                         float value) {
    if (i == j) return; // skip self
    size_t idx = getPairIndex(i, j, numTriangles);
    visibility[idx] = value;
}

} // namespace VisibilityCache
