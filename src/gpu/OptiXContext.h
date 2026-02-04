#pragma once

//
// OptiX Context Wrapper
//
// ARCHITECTURE NOTE:
//
// OptiX uses vertices/indices for ray tracing acceleration structures (GAS).
// Triangle geometry data (area, normal, centroid, reflectance, emission, radiosity)
// is stored directly in the Mesh structure for simplicity.
//
// For GPU kernels that need per-triangle data, we extract and upload as needed.
//

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

#include "../mesh/MeshData.h"

namespace OptiXContext {

// Error checking macros
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(error) \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(1); \
        } \
    } while(0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX error: " << optixGetErrorName(res) \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n"; \
            std::exit(1); \
        } \
    } while(0)

// OptiX context wrapper
class Context {
private:
    OptixDeviceContext context = nullptr;
    CUstream stream = nullptr;
    
    OptixModule module = nullptr;
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    
    OptixShaderBindingTable sbt = {};
    
    // World A (Traversal): vertices/indices for OptiX GAS only
    CUdeviceptr d_vertices = 0;
    CUdeviceptr d_indices = 0;
    
    // Triangle data for CUDA kernels
    CUdeviceptr d_cx = 0, d_cy = 0, d_cz = 0;  // centroids
    CUdeviceptr d_nx = 0, d_ny = 0, d_nz = 0;  // normals
    
    CUdeviceptr d_visibility = 0;       // float visibility buffer [0,1]
    CUdeviceptr d_geometricKernel = 0;  // float geometric kernel buffer
    CUdeviceptr d_launchParams = 0;
    
    OptixTraversableHandle gasHandle = 0;
    CUdeviceptr d_gasOutput = 0;
    
public:
    Context() {
        // Initialize CUDA
        CUDA_CHECK(cudaFree(0));
        
        // Initialize OptiX
        OPTIX_CHECK(optixInit());
        
        // Create OptiX context
        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void*) {
            std::cerr << "[OptiX][" << level << "][" << tag << "]: " << message << "\n";
        };
        options.logCallbackLevel = 4;
        
        CUcontext cuCtx = 0;  // Use current CUDA context
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
        
        // Create CUDA stream
        CUDA_CHECK(cudaStreamCreate(&stream));
    }
    
    ~Context() {
        cleanup();
        
        if (stream) cudaStreamDestroy(stream);
        if (context) optixDeviceContextDestroy(context);
    }
    
    void cleanup() {
        if (d_geometricKernel) cudaFree((void*)d_geometricKernel);
        if (d_visibility) cudaFree((void*)d_visibility);
        
        // World B cleanup
        if (d_nz) cudaFree((void*)d_nz);
        if (d_ny) cudaFree((void*)d_ny);
        if (d_nx) cudaFree((void*)d_nx);
        if (d_cz) cudaFree((void*)d_cz);
        if (d_cy) cudaFree((void*)d_cy);
        if (d_cx) cudaFree((void*)d_cx);
        
        // World A cleanup
        if (d_indices) cudaFree((void*)d_indices);
        if (d_vertices) cudaFree((void*)d_vertices);
        
        if (d_launchParams) cudaFree((void*)d_launchParams);
        if (d_gasOutput) cudaFree((void*)d_gasOutput);
        
        if (sbt.raygenRecord) cudaFree((void*)sbt.raygenRecord);
        if (sbt.missRecordBase) cudaFree((void*)sbt.missRecordBase);
        if (sbt.hitgroupRecordBase) cudaFree((void*)sbt.hitgroupRecordBase);
        
        if (pipeline) optixPipelineDestroy(pipeline);
        if (hitgroupPG) optixProgramGroupDestroy(hitgroupPG);
        if (missPG) optixProgramGroupDestroy(missPG);
        if (raygenPG) optixProgramGroupDestroy(raygenPG);
        if (module) optixModuleDestroy(module);
        
        d_geometricKernel = 0;
        d_visibility = 0;
        
        // World B reset
        d_nz = 0;
        d_ny = 0;
        d_nx = 0;
        d_cz = 0;
        d_cy = 0;
        d_cx = 0;
        
        // World A reset
        d_indices = 0;
        d_vertices = 0;
        
        d_launchParams = 0;
        d_gasOutput = 0;
        gasHandle = 0;
        
        sbt = {};
        pipeline = nullptr;
        hitgroupPG = nullptr;
        missPG = nullptr;
        raygenPG = nullptr;
        module = nullptr;
    }
    
    // Load PTX and create module
    void createModule(const std::string& ptxPath) {
        // Read PTX file
        std::ifstream ptxFile(ptxPath);
        if (!ptxFile) {
            std::cerr << "Failed to open PTX file: " << ptxPath << "\n";
            std::exit(1);
        }
        
        std::string ptx((std::istreambuf_iterator<char>(ptxFile)),
                        std::istreambuf_iterator<char>());
        
        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;
        
        pipelineCompileOptions = {};
        pipelineCompileOptions.usesMotionBlur = false;
        pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions.numPayloadValues = 2;  // hitPatchId + sourcePatchId
        pipelineCompileOptions.numAttributeValues = 0;
        pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
        
        char log[2048];
        size_t sizeof_log = sizeof(log);
        
        OPTIX_CHECK(optixModuleCreate(
            context,
            &moduleCompileOptions,
            &pipelineCompileOptions,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &module
        ));
    }
    
    // Create program groups
    void createProgramGroups() {
        char log[2048];
        size_t sizeof_log = sizeof(log);
        
        // Raygen program group
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc raygenPGDesc = {};
        raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenPGDesc.raygen.module = module;
        raygenPGDesc.raygen.entryFunctionName = "__raygen__hemisphere";
        
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &raygenPGDesc, 1, &pgOptions,
            log, &sizeof_log, &raygenPG
        ));
        
        // Miss program group
        OptixProgramGroupDesc missPGDesc = {};
        missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missPGDesc.miss.module = module;
        missPGDesc.miss.entryFunctionName = "__miss__hemisphere";
        
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &missPGDesc, 1, &pgOptions,
            log, &sizeof_log, &missPG
        ));
        
        // Hit group program group
        OptixProgramGroupDesc hitgroupPGDesc = {};
        hitgroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroupPGDesc.hitgroup.moduleCH = module;
        hitgroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__hemisphere";
        hitgroupPGDesc.hitgroup.moduleAH = module;
        hitgroupPGDesc.hitgroup.entryFunctionNameAH = "__anyhit__hemisphere";
        
        OPTIX_CHECK(optixProgramGroupCreate(
            context, &hitgroupPGDesc, 1, &pgOptions,
            log, &sizeof_log, &hitgroupPG
        ));
    }
    
    // Create pipeline
    void createPipeline() {
        OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };
        
        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 1;
        
        char log[2048];
        size_t sizeof_log = sizeof(log);
        
        OPTIX_CHECK(optixPipelineCreate(
            context,
            &pipelineCompileOptions,
            &pipelineLinkOptions,
            programGroups,
            sizeof(programGroups) / sizeof(programGroups[0]),
            log,
            &sizeof_log,
            &pipeline
        ));
    }
    
    // Create SBT
    void createSBT() {
        // Raygen record
        struct RaygenRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
        RaygenRecord rg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG, &rg_sbt));
        CUDA_CHECK(cudaMalloc((void**)&sbt.raygenRecord, sizeof(RaygenRecord)));
        CUDA_CHECK(cudaMemcpy((void*)sbt.raygenRecord, &rg_sbt, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
        
        // Miss record
        struct MissRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
        MissRecord ms_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG, &ms_sbt));
        CUDA_CHECK(cudaMalloc((void**)&sbt.missRecordBase, sizeof(MissRecord)));
        CUDA_CHECK(cudaMemcpy((void*)sbt.missRecordBase, &ms_sbt, sizeof(MissRecord), cudaMemcpyHostToDevice));
        sbt.missRecordStrideInBytes = sizeof(MissRecord);
        sbt.missRecordCount = 1;
        
        // Hit group record
        struct HitGroupRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
        HitGroupRecord hg_sbt;
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG, &hg_sbt));
        CUDA_CHECK(cudaMalloc((void**)&sbt.hitgroupRecordBase, sizeof(HitGroupRecord)));
        CUDA_CHECK(cudaMemcpy((void*)sbt.hitgroupRecordBase, &hg_sbt, sizeof(HitGroupRecord), cudaMemcpyHostToDevice));
        sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt.hitgroupRecordCount = 1;
    }
    
    // Build GAS from mesh
    void buildGAS(const Mesh& mesh) {
        // Upload vertices
        size_t vertexBytes = mesh.vertices.size() * sizeof(Vertex);
        CUDA_CHECK(cudaMalloc((void**)&d_vertices, vertexBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_vertices, mesh.vertices.data(), vertexBytes, cudaMemcpyHostToDevice));
        
        // Upload indices
        size_t indexBytes = mesh.indices.size() * sizeof(TriIdx);
        CUDA_CHECK(cudaMalloc((void**)&d_indices, indexBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_indices, mesh.indices.data(), indexBytes, cudaMemcpyHostToDevice));
        
        // Build input
        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
        buildInput.triangleArray.numVertices = static_cast<unsigned int>(mesh.vertices.size());
        buildInput.triangleArray.vertexBuffers = &d_vertices;
        
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(TriIdx);
        buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.indices.size());
        buildInput.triangleArray.indexBuffer = d_indices;
        
        uint32_t buildInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
        buildInput.triangleArray.flags = &buildInputFlags;
        buildInput.triangleArray.numSbtRecords = 1;
        
        // Build options
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        
        // Query memory requirements
        OptixAccelBufferSizes gasBufferSizes;
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accelOptions, &buildInput, 1, &gasBufferSizes));
        
        // Allocate GAS buffer
        CUDA_CHECK(cudaMalloc((void**)&d_gasOutput, gasBufferSizes.outputSizeInBytes));
        
        // Allocate temp buffer
        CUdeviceptr d_temp;
        CUDA_CHECK(cudaMalloc((void**)&d_temp, gasBufferSizes.tempSizeInBytes));
        
        // Build GAS
        OPTIX_CHECK(optixAccelBuild(
            context,
            stream,
            &accelOptions,
            &buildInput,
            1,
            d_temp,
            gasBufferSizes.tempSizeInBytes,
            d_gasOutput,
            gasBufferSizes.outputSizeInBytes,
            &gasHandle,
            nullptr,
            0
        ));
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        CUDA_CHECK(cudaFree((void*)d_temp));
    }
    
    // Compute visibility and geometric kernels
    void computeVisibility(const Mesh& mesh,
                          std::vector<float>& visibility, 
                          std::vector<float>& geometricKernel,
                          uint32_t samplesPerPair) {
        uint32_t numTriangles = static_cast<uint32_t>(mesh.indices.size());
        size_t numPairs = (size_t(numTriangles) * (numTriangles - 1)) / 2;
        
        visibility.resize(numPairs);
        geometricKernel.resize(numPairs);
        
        // Allocate device memory for visibility and geometric kernel as float
        CUDA_CHECK(cudaMalloc((void**)&d_visibility, numPairs * sizeof(float)));
        CUDA_CHECK(cudaMemset((void*)d_visibility, 0, numPairs * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc((void**)&d_geometricKernel, numPairs * sizeof(float)));
        CUDA_CHECK(cudaMemset((void*)d_geometricKernel, 0, numPairs * sizeof(float)));
        
        // Upload triangle data
        size_t floatBytes = numTriangles * sizeof(float);
        std::vector<float> cx(numTriangles), cy(numTriangles), cz(numTriangles);
        std::vector<float> nx(numTriangles), ny(numTriangles), nz(numTriangles);
        
        for (uint32_t i = 0; i < numTriangles; ++i) {
            cx[i] = mesh.triangle_centroid[i].x;
            cy[i] = mesh.triangle_centroid[i].y;
            cz[i] = mesh.triangle_centroid[i].z;
            nx[i] = mesh.triangle_normal[i].x;
            ny[i] = mesh.triangle_normal[i].y;
            nz[i] = mesh.triangle_normal[i].z;
        }
        
        // Centroids
        CUDA_CHECK(cudaMalloc((void**)&d_cx, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_cy, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_cz, floatBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_cx, cx.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_cy, cy.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_cz, cz.data(), floatBytes, cudaMemcpyHostToDevice));
        
        // Normals
        CUDA_CHECK(cudaMalloc((void**)&d_nx, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_ny, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_nz, floatBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_nx, nx.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_ny, ny.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_nz, nz.data(), floatBytes, cudaMemcpyHostToDevice));
        
        // Prepare launch params (hybrid: vertices for sampling, SoA for geometry)
        struct LaunchParams {
            float3* vertices;
            uint3* indices;
            float* cx;
            float* cy;
            float* cz;
            float* nx;
            float* ny;
            float* nz;
            float* visibility;
            float* geometricKernel;
            uint32_t numTriangles;
            uint32_t samplesPerPair;
            uint32_t iOffset;
            OptixTraversableHandle gasHandle;
        };
        
        LaunchParams params;
        params.vertices = (float3*)d_vertices;
        params.indices = (uint3*)d_indices;
        params.cx = (float*)d_cx;
        params.cy = (float*)d_cy;
        params.cz = (float*)d_cz;
        params.nx = (float*)d_nx;
        params.ny = (float*)d_ny;
        params.nz = (float*)d_nz;
        params.visibility = (float*)d_visibility;
        params.geometricKernel = (float*)d_geometricKernel;
        params.cy = (float*)d_cy;
        params.cz = (float*)d_cz;
        params.nx = (float*)d_nx;
        params.ny = (float*)d_ny;
        params.nz = (float*)d_nz;
        params.visibility = (float*)d_visibility;  // cast to float*
        params.numTriangles = numTriangles;
        params.samplesPerPair = samplesPerPair;
        params.iOffset = 0;  // Will be updated per batch
        params.gasHandle = gasHandle;
        
        CUDA_CHECK(cudaMalloc((void**)&d_launchParams, sizeof(LaunchParams)));
        CUDA_CHECK(cudaMemcpy((void*)d_launchParams, &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));
        
        // Launch OptiX kernel in batches to avoid exceeding launch limits
        // OptiX max launch size is typically 2^31-1, but we batch to be safe
        const uint32_t maxBatchSize = 16384; // 16k x 16k = 256M rays per batch
        
        for (uint32_t iStart = 0; iStart < numTriangles; iStart += maxBatchSize) {
            uint32_t iEnd = std::min(iStart + maxBatchSize, numTriangles);
            uint32_t batchWidth = iEnd - iStart;
            
            std::cout << "  Batch " << (iStart / maxBatchSize + 1) 
                      << ": triangles " << iStart << "-" << iEnd << "..." << std::flush;
            
            // Update launch params for this batch
            params.iOffset = iStart;
            CUDA_CHECK(cudaMemcpy((void*)d_launchParams, &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));
            
            OPTIX_CHECK(optixLaunch(
                pipeline,
                stream,
                d_launchParams,
                sizeof(LaunchParams),
                &sbt,
                batchWidth,    // width (subset of i)
                numTriangles,  // height (full j range)
                1              // depth
            ));
            
            std::cout << " done.\n" << std::flush;
        }
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Download results as float
        CUDA_CHECK(cudaMemcpy(visibility.data(), (void*)d_visibility, numPairs * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(geometricKernel.data(), (void*)d_geometricKernel, numPairs * sizeof(float), cudaMemcpyDeviceToHost));
    }
    
    // Compute form factors directly via PER-VERTEX hemisphere ray casting
    void computeFormFactors(const Mesh& mesh,
                            std::vector<float>& formFactors,
                            uint32_t originSamples = 1,
                            uint32_t dirSamples = 256) {
        uint32_t numVertices = static_cast<uint32_t>(mesh.vertices.size());
        uint32_t numTriangles = static_cast<uint32_t>(mesh.indices.size());
        size_t matrixSize = size_t(numVertices) * numTriangles;
        formFactors.resize(matrixSize, 0.0f);
        
        std::cout << "\n=== Computing Form Factors (Per-Vertex Hemisphere Sampling) ===\n";
        std::cout << "Vertices: " << numVertices << "\n";
        std::cout << "Triangles: " << numTriangles << "\n";
        std::cout << "Direction samples per vertex: " << dirSamples << "\n";
        std::cout << "Total rays: " << (uint64_t(numVertices) * dirSamples) << "\n\n";
        
        // Compute vertex normals from incident triangles
        std::vector<Vec3> vertexNormals(numVertices, Vec3(0,0,0));
        for (uint32_t ti = 0; ti < numTriangles; ++ti) {
            const auto& tri = mesh.indices[ti];
            Vec3 triNormal = mesh.triangle_normal[ti];
            vertexNormals[tri.i0] = vertexNormals[tri.i0] + triNormal;
            vertexNormals[tri.i1] = vertexNormals[tri.i1] + triNormal;
            vertexNormals[tri.i2] = vertexNormals[tri.i2] + triNormal;
        }
        for (auto& n : vertexNormals) {
            n = n.normalized();
        }
        
        // Allocate device memory
        CUdeviceptr d_formFactors;
        CUDA_CHECK(cudaMalloc((void**)&d_formFactors, matrixSize * sizeof(float)));
        CUDA_CHECK(cudaMemset((void*)d_formFactors, 0, matrixSize * sizeof(float)));
        
        CUdeviceptr d_vertexNormals;
        CUDA_CHECK(cudaMalloc((void**)&d_vertexNormals, numVertices * sizeof(float3)));
        CUDA_CHECK(cudaMemcpy((void*)d_vertexNormals, vertexNormals.data(), 
                              numVertices * sizeof(float3), cudaMemcpyHostToDevice));
        
        // Compute scene epsilon
        float sceneSize = 0.0f;
        for (const auto& v : mesh.vertices) {
            sceneSize = std::max(sceneSize, std::abs(v.x));
            sceneSize = std::max(sceneSize, std::abs(v.y));
            sceneSize = std::max(sceneSize, std::abs(v.z));
        }
        float sceneEpsilon = sceneSize * 1e-4f;
        
        // Prepare launch params
        struct LaunchParams {
            float3* vertices;
            uint3* indices;
            float3* vertexNormals;
            uint32_t numVertices;
            uint32_t numTriangles;
            uint32_t dirSamples;
            float* formFactors;
            float sceneEpsilon;
            OptixTraversableHandle gasHandle;
        };
        
        LaunchParams params;
        params.vertices = (float3*)d_vertices;
        params.indices = (uint3*)d_indices;
        params.vertexNormals = (float3*)d_vertexNormals;
        params.numVertices = numVertices;
        params.numTriangles = numTriangles;
        params.dirSamples = dirSamples;
        params.formFactors = (float*)d_formFactors;
        params.sceneEpsilon = sceneEpsilon;
        params.gasHandle = gasHandle;
        
        CUDA_CHECK(cudaMalloc((void**)&d_launchParams, sizeof(LaunchParams)));
        CUDA_CHECK(cudaMemcpy((void*)d_launchParams, &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));
        
        // Launch: [numVertices, dirSamples]
        std::cout << "Launching OptiX...\n" << std::flush;
        auto t0 = std::chrono::high_resolution_clock::now();
        
        OPTIX_CHECK(optixLaunch(
            pipeline,
            stream,
            d_launchParams,
            sizeof(LaunchParams),
            &sbt,
            numVertices,   // width (vertices)
            dirSamples,    // height (rays per vertex)
            1              // depth
        ));
        
        CUDA_CHECK(cudaStreamSynchronize(stream));
        
        auto t1 = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "OptiX completed in " << elapsed << "s\n";
        std::cout << "Rays/sec: " << (uint64_t(numVertices) * dirSamples / elapsed / 1e6) << " million\n\n";
        
        // Download results
        std::cout << "Downloading vertex-to-triangle form factors...\n" << std::flush;
        CUDA_CHECK(cudaMemcpy(formFactors.data(), (void*)d_formFactors, 
                              matrixSize * sizeof(float), cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaFree((void*)d_formFactors));
        CUDA_CHECK(cudaFree((void*)d_vertexNormals));
        
        std::cout << "Form factors computed!\n";
    }
};

} // namespace OptiXContext
