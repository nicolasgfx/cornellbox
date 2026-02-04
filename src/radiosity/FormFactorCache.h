#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <cstring>

namespace FormFactorCache {

// Compressed Sparse Row (CSR) format for form factors
// Stores only non-zero F[i,j] values efficiently
struct FormFactorCSR {
    uint32_t numPatches;           // N (number of patches)
    uint32_t raysPerPatch;         // R (sampling density)
    std::string profile;           // "low", "medium", "high"
    
    // CSR data
    std::vector<uint32_t> rowPtr;  // [N+1] - row start indices
    std::vector<uint32_t> colIdx;  // [nnz] - column indices
    std::vector<float> values;     // [nnz] - form factor values
    
    FormFactorCSR() : numPatches(0), raysPerPatch(0) {}
    
    size_t nonZeros() const { return values.size(); }
    
    // Get F[i,j] (0 if not stored)
    float get(uint32_t i, uint32_t j) const {
        if (i >= numPatches) return 0.0f;
        
        uint32_t start = rowPtr[i];
        uint32_t end = rowPtr[i + 1];
        
        for (uint32_t k = start; k < end; ++k) {
            if (colIdx[k] == j) {
                return values[k];
            }
        }
        return 0.0f;
    }
    
    // Get entire row i as sparse pairs (j, F[i,j])
    void getRow(uint32_t i, std::vector<std::pair<uint32_t, float>>& row) const {
        row.clear();
        if (i >= numPatches) return;
        
        uint32_t start = rowPtr[i];
        uint32_t end = rowPtr[i + 1];
        
        row.reserve(end - start);
        for (uint32_t k = start; k < end; ++k) {
            row.emplace_back(colIdx[k], values[k]);
        }
    }
    
    // Compute row sums for validation
    std::vector<float> computeRowSums() const {
        std::vector<float> sums(numPatches, 0.0f);
        for (uint32_t i = 0; i < numPatches; ++i) {
            uint32_t start = rowPtr[i];
            uint32_t end = rowPtr[i + 1];
            for (uint32_t k = start; k < end; ++k) {
                sums[i] += values[k];
            }
        }
        return sums;
    }
};

// Cache file format version
constexpr uint32_t CACHE_VERSION = 1;
constexpr char CACHE_MAGIC[8] = "FFCSR01";

// Build cache filename
inline std::string getCacheFilename(const std::string& profile, 
                                    uint32_t originSamples, 
                                    uint32_t dirSamples) {
    uint32_t totalRays = originSamples * dirSamples;
    return "output/cache/ff_hemi_" + profile + 
           "_O" + std::to_string(originSamples) +
           "_D" + std::to_string(dirSamples) +
           "_R" + std::to_string(totalRays) + ".csr.bin";
}

// Save CSR to binary file
inline bool saveCSR(const std::string& filepath, const FormFactorCSR& csr) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open file for writing: " << filepath << "\n";
        return false;
    }
    
    // Header
    file.write(CACHE_MAGIC, 8);
    file.write(reinterpret_cast<const char*>(&CACHE_VERSION), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&csr.numPatches), sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(&csr.raysPerPatch), sizeof(uint32_t));
    
    uint32_t profileLen = static_cast<uint32_t>(csr.profile.size());
    file.write(reinterpret_cast<const char*>(&profileLen), sizeof(uint32_t));
    file.write(csr.profile.c_str(), profileLen);
    
    uint32_t nnz = static_cast<uint32_t>(csr.values.size());
    file.write(reinterpret_cast<const char*>(&nnz), sizeof(uint32_t));
    
    // CSR data
    file.write(reinterpret_cast<const char*>(csr.rowPtr.data()), 
               csr.rowPtr.size() * sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(csr.colIdx.data()), 
               csr.colIdx.size() * sizeof(uint32_t));
    file.write(reinterpret_cast<const char*>(csr.values.data()), 
               csr.values.size() * sizeof(float));
    
    file.close();
    
    std::cout << "Saved form factor cache: " << filepath << "\n";
    std::cout << "  Patches: " << csr.numPatches << "\n";
    std::cout << "  Rays/patch: " << csr.raysPerPatch << "\n";
    std::cout << "  Non-zeros: " << nnz << " (" 
              << (100.0 * nnz / (uint64_t(csr.numPatches) * csr.numPatches)) 
              << "% density)\n";
    
    return true;
}

// Load CSR from binary file
inline bool loadCSR(const std::string& filepath, FormFactorCSR& csr) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        std::cerr << "Cache file not found: " << filepath << "\n";
        return false;
    }
    
    // Verify header
    char magic[8];
    file.read(magic, 8);
    if (std::memcmp(magic, CACHE_MAGIC, 8) != 0) {
        std::cerr << "Invalid cache file magic\n";
        return false;
    }
    
    uint32_t version;
    file.read(reinterpret_cast<char*>(&version), sizeof(uint32_t));
    if (version != CACHE_VERSION) {
        std::cerr << "Cache file version mismatch\n";
        return false;
    }
    
    file.read(reinterpret_cast<char*>(&csr.numPatches), sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(&csr.raysPerPatch), sizeof(uint32_t));
    
    uint32_t profileLen;
    file.read(reinterpret_cast<char*>(&profileLen), sizeof(uint32_t));
    csr.profile.resize(profileLen);
    file.read(&csr.profile[0], profileLen);
    
    uint32_t nnz;
    file.read(reinterpret_cast<char*>(&nnz), sizeof(uint32_t));
    
    // CSR data
    csr.rowPtr.resize(csr.numPatches + 1);
    csr.colIdx.resize(nnz);
    csr.values.resize(nnz);
    
    file.read(reinterpret_cast<char*>(csr.rowPtr.data()), 
              csr.rowPtr.size() * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(csr.colIdx.data()), 
              csr.colIdx.size() * sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(csr.values.data()), 
              csr.values.size() * sizeof(float));
    
    file.close();
    
    std::cout << "Loaded form factor cache: " << filepath << "\n";
    std::cout << "  Patches: " << csr.numPatches << "\n";
    std::cout << "  Rays/patch: " << csr.raysPerPatch << "\n";
    std::cout << "  Non-zeros: " << nnz << "\n";
    
    return true;
}

// Convert dense matrix to CSR (for LOW profile or debugging)
inline FormFactorCSR denseToCSR(const std::vector<std::vector<float>>& dense,
                                 const std::string& profile,
                                 uint32_t raysPerPatch,
                                 float threshold = 1e-8f) {
    FormFactorCSR csr;
    csr.numPatches = static_cast<uint32_t>(dense.size());
    csr.raysPerPatch = raysPerPatch;
    csr.profile = profile;
    
    csr.rowPtr.push_back(0);
    
    for (uint32_t i = 0; i < csr.numPatches; ++i) {
        for (uint32_t j = 0; j < csr.numPatches; ++j) {
            if (i != j && dense[i][j] > threshold) {
                csr.colIdx.push_back(j);
                csr.values.push_back(dense[i][j]);
            }
        }
        csr.rowPtr.push_back(static_cast<uint32_t>(csr.values.size()));
    }
    
    return csr;
}

// Convert dense flat array to CSR (for OptiX output)
inline FormFactorCSR denseFlatToCSR(const float* denseFlat,
                                     uint32_t numPatches,
                                     const std::string& profile,
                                     uint32_t raysPerPatch,
                                     float threshold = 1e-8f) {
    FormFactorCSR csr;
    csr.numPatches = numPatches;
    csr.raysPerPatch = raysPerPatch;
    csr.profile = profile;
    
    csr.rowPtr.push_back(0);
    
    for (uint32_t i = 0; i < numPatches; ++i) {
        for (uint32_t j = 0; j < numPatches; ++j) {
            float val = denseFlat[i * numPatches + j];
            if (i != j && val > threshold) {
                csr.colIdx.push_back(j);
                csr.values.push_back(val);
            }
        }
        csr.rowPtr.push_back(static_cast<uint32_t>(csr.values.size()));
    }
    
    return csr;
}

// Validation: check row sums and report statistics
inline void validateCSR(const FormFactorCSR& csr) {
    std::cout << "\n=== Form Factor Validation ===\n";
    
    auto rowSums = csr.computeRowSums();
    
    float minSum = 1e10f, maxSum = -1e10f;
    float avgSum = 0.0f;
    uint32_t countLow = 0, countHigh = 0;
    
    for (float s : rowSums) {
        avgSum += s;
        minSum = std::min(minSum, s);
        maxSum = std::max(maxSum, s);
        
        if (s < 0.5f) countLow++;
        if (s > 1.01f) countHigh++;
    }
    avgSum /= csr.numPatches;
    
    std::cout << "Row sums:\n";
    std::cout << "  Min: " << minSum << "\n";
    std::cout << "  Max: " << maxSum << "\n";
    std::cout << "  Avg: " << avgSum << "\n";
    std::cout << "  Patches with sum < 0.5: " << countLow << "\n";
    std::cout << "  Patches with sum > 1.01: " << countHigh << "\n";
    
    if (countHigh > 0) {
        std::cout << "WARNING: Some row sums exceed 1.0 (energy conservation violated)\n";
    }
    
    // Check for self-hits
    uint32_t selfHits = 0;
    for (uint32_t i = 0; i < csr.numPatches; ++i) {
        if (csr.get(i, i) > 1e-8f) {
            selfHits++;
        }
    }
    
    if (selfHits > 0) {
        std::cout << "WARNING: " << selfHits << " patches have F[i,i] > 0 (self-hits not filtered)\n";
    } else {
        std::cout << "✓ No self-hits detected\n";
    }
    
    std::cout << "✓ Validation complete\n\n";
}

} // namespace FormFactorCache
