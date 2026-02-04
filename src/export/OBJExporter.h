#pragma once
#include "../mesh/MeshData.h"
#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace OBJExporter {

inline bool exportOBJ(const std::string& filepath,
                      const Mesh& mesh,
                      const std::vector<Vec3>& vertexColors) 
{
    // Generate MTL filename
    size_t lastSlash = filepath.find_last_of("/\\");
    size_t lastDot = filepath.find_last_of('.');
    std::string baseName = (lastDot != std::string::npos) 
        ? filepath.substr(lastSlash + 1, lastDot - lastSlash - 1)
        : filepath.substr(lastSlash + 1);
    
    std::string mtlFilename = baseName + ".mtl";
    std::string mtlPath = filepath.substr(0, lastSlash + 1) + mtlFilename;
    
    // Write OBJ file
    std::ofstream objFile(filepath);
    if (!objFile.is_open()) {
        std::cerr << "Failed to create OBJ file: " << filepath << std::endl;
        return false;
    }
    
    objFile << "# Radiosity Cornell Box\n";
    objFile << "mtllib " << mtlFilename << "\n\n";
    
    // Write vertices with colors
    for (size_t i = 0; i < mesh.numVertices(); ++i) {
        const Vertex& v = mesh.vertices[i];
        const Vec3& c = (i < vertexColors.size()) ? vertexColors[i] : Vec3(0.8f);
        
        objFile << "v " << v.x << " " << v.y << " " << v.z << " "
                << MathUtils::clamp(c.x, 0.0f, 1.0f) << " "
                << MathUtils::clamp(c.y, 0.0f, 1.0f) << " "
                << MathUtils::clamp(c.z, 0.0f, 1.0f) << "\n";
    }
    
    objFile << "\n";
    
    // Write material reference
    objFile << "usemtl default\n\n";
    
    // Write faces
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        // OBJ indices are 1-based
        objFile << "f " << (tri.i0 + 1) << " " 
                << (tri.i1 + 1) << " " 
                << (tri.i2 + 1) << "\n";
    }
    
    objFile.close();
    
    // Write MTL file
    std::ofstream mtlFile(mtlPath);
    if (!mtlFile.is_open()) {
        std::cerr << "Failed to create MTL file: " << mtlPath << std::endl;
        return false;
    }
    
    mtlFile << "# Radiosity Cornell Box Material\n\n";
    mtlFile << "newmtl default\n";
    mtlFile << "Ka 1.0 1.0 1.0\n";
    mtlFile << "Kd 1.0 1.0 1.0\n";
    mtlFile << "Ks 0.0 0.0 0.0\n";
    mtlFile << "Ns 0.0\n";
    mtlFile << "illum 1\n";
    
    mtlFile.close();
    
    std::cout << "Exported OBJ: " << filepath << " (" 
              << mesh.numVertices() << " vertices, " 
              << mesh.numTriangles() << " triangles)" << std::endl;
    
    return true;
}

// Simplified export with default colors
inline bool exportOBJ(const std::string& filepath, const Mesh& mesh) {
    std::vector<Vec3> defaultColors(mesh.numVertices(), Vec3(0.8f, 0.8f, 0.8f));
    return exportOBJ(filepath, mesh, defaultColors);
}

} // namespace OBJExporter
