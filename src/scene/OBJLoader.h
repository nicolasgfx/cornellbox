#pragma once
#include "../mesh/MeshData.h"
#include "../math/MathUtils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace OBJLoader {

inline bool loadOBJ(const std::string& filepath, Mesh& mesh) {
    mesh.clear();
    
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "Failed to open OBJ file: " << filepath << std::endl;
        return false;
    }

    std::vector<Vec3> positions;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        std::string prefix;
        iss >> prefix;
        
        if (prefix == "v") {
            // Vertex position
            float x, y, z;
            iss >> x >> y >> z;
            positions.push_back(Vec3(x, y, z));
        }
        else if (prefix == "f") {
            // Face (triangle or quad)
            std::vector<uint32_t> face_indices;
            std::string vertex_str;
            
            while (iss >> vertex_str) {
                // Parse vertex index (format: v or v/vt or v/vt/vn or v//vn)
                size_t slash_pos = vertex_str.find('/');
                std::string index_str = (slash_pos != std::string::npos) 
                    ? vertex_str.substr(0, slash_pos) 
                    : vertex_str;
                
                int idx = std::stoi(index_str);
                // OBJ indices are 1-based
                if (idx > 0) {
                    face_indices.push_back(idx - 1);
                } else if (idx < 0) {
                    // Negative indices count from the end
                    face_indices.push_back(positions.size() + idx);
                }
            }
            
            // Triangulate if needed
            if (face_indices.size() >= 3) {
                // Triangle fan triangulation for n-gons
                for (size_t i = 1; i + 1 < face_indices.size(); ++i) {
                    mesh.indices.push_back(TriIdx(
                        face_indices[0],
                        face_indices[i],
                        face_indices[i + 1]
                    ));
                }
            }
        }
    }
    
    // Convert positions to Vertex format
    mesh.vertices.reserve(positions.size());
    for (const auto& p : positions) {
        mesh.vertices.push_back(Vertex(p));
    }
    
    std::cout << "Loaded OBJ: " << mesh.numVertices() << " vertices, " 
              << mesh.numTriangles() << " triangles" << std::endl;
    
    return true;
}

// Validate mesh integrity
inline bool validateMesh(const Mesh& mesh) {
    size_t numVerts = mesh.numVertices();
    size_t numTris = mesh.numTriangles();
    
    if (numVerts == 0 || numTris == 0) {
        std::cerr << "Empty mesh" << std::endl;
        return false;
    }
    
    // Check index bounds
    for (size_t i = 0; i < numTris; ++i) {
        const auto& tri = mesh.indices[i];
        if (tri.i0 >= numVerts || tri.i1 >= numVerts || tri.i2 >= numVerts) {
            std::cerr << "Triangle " << i << " has out-of-bounds index" << std::endl;
            return false;
        }
    }
    
    // Check for degenerate triangles
    size_t degenerateCount = 0;
    for (size_t i = 0; i < numTris; ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();
        
        if (MathUtils::isTriangleDegenerate(v0, v1, v2)) {
            degenerateCount++;
        }
    }
    
    if (degenerateCount > 0) {
        std::cerr << "Warning: " << degenerateCount << " degenerate triangles found" << std::endl;
    }
    
    return true;
}

} // namespace OBJLoader
