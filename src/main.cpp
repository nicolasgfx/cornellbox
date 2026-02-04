#define NOMINMAX
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "app/Config.h"
#include "scene/CornellBox.h"
#include "mesh/Subdivision.h"
#include "mesh/PatchBuilder.h"
#include "export/VertexColor.h"
#include "export/OBJExporter.h"
#include <iostream>
#include <filesystem>
#include <algorithm>

int main(int argc, char** argv) {
    Config config;
    if (!config.parseArgs(argc, argv)) return 0;
    
    std::cout << "Profile: " << config.getProfileName() << ", Phase: " << config.phase << "\n\n";
    
    Mesh mesh = CornellBox::createCornellBox();
    
    CornellBox::fixNormalsOrientation(mesh);
    
    mesh = Subdivision::subdivideByArea(mesh, config.getTargetArea());
    
    mesh.triangle_material_id.resize(mesh.numTriangles());
    uint32_t lightCount = 0;
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 centroid = MathUtils::triangleCentroid(
            mesh.vertices[tri.i0].toVec3(),
            mesh.vertices[tri.i1].toVec3(),
            mesh.vertices[tri.i2].toVec3());
        mesh.triangle_material_id[i] = CornellBox::getMaterialIDFromPosition(centroid);
        if (mesh.triangle_material_id[i] == CornellBox::MAT_LIGHT) lightCount++;
    }
    
    PatchBuilder::buildVertexAdjacency(mesh);
    PatchBuilder::buildTriangleData(mesh);
    
    if (config.validate && !PatchBuilder::validateMesh(mesh)) return 1;
    
    std::cout << "Mesh: " << mesh.numTriangles() << " triangles, " << lightCount << " lights\n";
    
    std::string profilePath = config.outputPath + "/" + config.getProfileName();
    std::filesystem::create_directories(profilePath);
    
    std::vector<Vec3> vertexColors = VertexColor::bakeTriangleColorsToVertices(mesh, mesh.triangle_reflectance);
    std::string filename = profilePath + "/cornell_phase1.obj";
    if (!OBJExporter::exportOBJ(filename, mesh, vertexColors)) return 1;
    std::cout << "Phase 1: " << filename << "\n";
    
    std::vector<Vec3> normalColors(mesh.numTriangles());
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        Vec3 n = mesh.triangle_normal[i];
        normalColors[i] = Vec3((n.x + 1.0f) * 0.5f, (n.y + 1.0f) * 0.5f, (n.z + 1.0f) * 0.5f);
    }
    std::string phase2File = profilePath + "/cornell_phase2.obj";
    if (!OBJExporter::exportOBJ(phase2File, mesh, VertexColor::bakeTriangleColorsToVertices(mesh, normalColors))) return 1;
    std::cout << "Phase 2: " << phase2File << "\n";
    
    if (config.phase < 2) return 0;
    
    std::cout << "\nPhase 3: Radiosity\n";
    
    uint32_t numTriangles = mesh.numTriangles();
    std::vector<float> formFactors(numTriangles * numTriangles, 0.0f);
    
    for (uint32_t i = 0; i < numTriangles; ++i) {
        Vec3 ci = mesh.triangle_centroid[i];
        Vec3 ni = mesh.triangle_normal[i];
        
        for (uint32_t j = 0; j < numTriangles; ++j) {
            if (i == j) continue;
            
            Vec3 r = mesh.triangle_centroid[j] - ci;
            float dist = r.length();
            if (dist < 1e-6f) continue;
            
            Vec3 rNorm = r / dist;
            float cosI = ni.dot(rNorm);
            float cosJ = mesh.triangle_normal[j].dot(rNorm * (-1.0f));
            
            if (cosI > 0.0f && cosJ > 0.0f) {
                formFactors[i * numTriangles + j] = (cosI * cosJ * mesh.triangle_area[j]) / (float(M_PI) * dist * dist);
            }
        }
    }
    
    std::vector<Vec3> radiosity1Step(numTriangles);
    std::vector<Vec3> radiosityNSteps(numTriangles);
    std::vector<Vec3> unshot(numTriangles);
    
    for (uint32_t i = 0; i < numTriangles; ++i) {
        radiosity1Step[i] = radiosityNSteps[i] = unshot[i] = mesh.triangle_emission[i];
    }
    
    for (uint32_t i = 0; i < numTriangles; ++i) {
        for (uint32_t j = 0; j < numTriangles; ++j) {
            float ff = formFactors[i * numTriangles + j];
            if (ff < 1e-8f) continue;
            Vec3 incoming = mesh.triangle_emission[j] * ff;
            radiosity1Step[i] = radiosity1Step[i] + Vec3(
                incoming.x * mesh.triangle_reflectance[i].x,
                incoming.y * mesh.triangle_reflectance[i].y,
                incoming.z * mesh.triangle_reflectance[i].z);
        }
    }
    
    const uint32_t maxIter = 100;
    const float convergence = 1e-4f;
    
    for (uint32_t iter = 0; iter < maxIter; ++iter) {
        float maxUnshot = 0.0f;
        uint32_t shootTri = 0;
        
        for (uint32_t i = 0; i < numTriangles; ++i) {
            float unshotMag = unshot[i].length();
            if (unshotMag > maxUnshot) {
                maxUnshot = unshotMag;
                shootTri = i;
            }
        }
        
        if (maxUnshot < convergence) break;
        
        Vec3 shootEnergy = unshot[shootTri];
        unshot[shootTri] = Vec3(0, 0, 0);
        
        for (uint32_t j = 0; j < numTriangles; ++j) {
            float ff = formFactors[j * numTriangles + shootTri];
            if (ff < 1e-8f) continue;
            
            Vec3 incoming = shootEnergy * ff;
            Vec3 reflected = Vec3(
                incoming.x * mesh.triangle_reflectance[j].x,
                incoming.y * mesh.triangle_reflectance[j].y,
                incoming.z * mesh.triangle_reflectance[j].z);
            
            radiosityNSteps[j] = radiosityNSteps[j] + reflected;
            unshot[j] = unshot[j] + reflected;
        }
    }
    
    std::string file1Step = profilePath + "/cornell_1step.obj";
    std::string fileNSteps = profilePath + "/cornell_nsteps.obj";
    
    if (!OBJExporter::exportOBJ(file1Step, mesh, VertexColor::bakeTriangleColorsToVertices(mesh, radiosity1Step))) return 1;
    if (!OBJExporter::exportOBJ(fileNSteps, mesh, VertexColor::bakeTriangleColorsToVertices(mesh, radiosityNSteps))) return 1;
    
    std::cout << "Phase 3: " << file1Step << ", " << fileNSteps << "\n";
    
    return 0;
}
