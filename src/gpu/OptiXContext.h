#pragma once

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../mesh/MeshData.h"

namespace OptiXContext {

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

class Context {
private:
    OptixDeviceContext context_ = nullptr;
    CUstream stream_ = nullptr;

    OptixModule module_ = nullptr;
    OptixProgramGroup raygenPG_ = nullptr;
    OptixProgramGroup missPG_ = nullptr;
    OptixProgramGroup hitgroupPG_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixPipelineCompileOptions pipelineCompileOptions_ = {};
    OptixShaderBindingTable sbt_ = {};

    CUdeviceptr d_vertices_ = 0;
    CUdeviceptr d_indices_ = 0;
    CUdeviceptr d_cx_ = 0, d_cy_ = 0, d_cz_ = 0;
    CUdeviceptr d_nx_ = 0, d_ny_ = 0, d_nz_ = 0;
    CUdeviceptr d_area_ = 0;
    CUdeviceptr d_launchParams_ = 0;
    CUdeviceptr d_gasOutput_ = 0;

    OptixTraversableHandle gasHandle_ = 0;
    uint32_t cachedNumTriangles_ = 0;

    static float computeSceneEpsilon(const Mesh& mesh) {
        float sceneSize = 0.0f;
        for (const auto& v : mesh.vertices) {
            sceneSize = std::max(sceneSize, std::abs(v.x));
            sceneSize = std::max(sceneSize, std::abs(v.y));
            sceneSize = std::max(sceneSize, std::abs(v.z));
        }
        return sceneSize * 1e-4f;
    }

public:
    Context() {
        CUDA_CHECK(cudaFree(0));
        OPTIX_CHECK(optixInit());

        OptixDeviceContextOptions options = {};
        options.logCallbackFunction = [](unsigned int level, const char*, const char* message, void*) {
            if (level <= 2) std::cerr << "[OptiX] " << message << "\n";
        };
        options.logCallbackLevel = 2;

        CUcontext cuCtx = 0;
        OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context_));
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }

    ~Context() {
        cleanup();
        if (stream_) cudaStreamDestroy(stream_);
        if (context_) optixDeviceContextDestroy(context_);
    }

    void cleanup() {
        if (d_area_) CUDA_CHECK(cudaFree((void*)d_area_));
        if (d_nz_) CUDA_CHECK(cudaFree((void*)d_nz_));
        if (d_ny_) CUDA_CHECK(cudaFree((void*)d_ny_));
        if (d_nx_) CUDA_CHECK(cudaFree((void*)d_nx_));
        if (d_cz_) CUDA_CHECK(cudaFree((void*)d_cz_));
        if (d_cy_) CUDA_CHECK(cudaFree((void*)d_cy_));
        if (d_cx_) CUDA_CHECK(cudaFree((void*)d_cx_));

        if (d_indices_) CUDA_CHECK(cudaFree((void*)d_indices_));
        if (d_vertices_) CUDA_CHECK(cudaFree((void*)d_vertices_));

        if (d_launchParams_) CUDA_CHECK(cudaFree((void*)d_launchParams_));
        if (d_gasOutput_) CUDA_CHECK(cudaFree((void*)d_gasOutput_));

        if (sbt_.raygenRecord) CUDA_CHECK(cudaFree((void*)sbt_.raygenRecord));
        if (sbt_.missRecordBase) CUDA_CHECK(cudaFree((void*)sbt_.missRecordBase));
        if (sbt_.hitgroupRecordBase) CUDA_CHECK(cudaFree((void*)sbt_.hitgroupRecordBase));

        if (pipeline_) optixPipelineDestroy(pipeline_);
        if (hitgroupPG_) optixProgramGroupDestroy(hitgroupPG_);
        if (missPG_) optixProgramGroupDestroy(missPG_);
        if (raygenPG_) optixProgramGroupDestroy(raygenPG_);
        if (module_) optixModuleDestroy(module_);

        d_nz_ = d_ny_ = d_nx_ = d_cz_ = d_cy_ = d_cx_ = 0;
        d_area_ = 0;
        d_indices_ = d_vertices_ = d_launchParams_ = d_gasOutput_ = 0;
        sbt_ = {};

        pipeline_ = nullptr;
        hitgroupPG_ = nullptr;
        missPG_ = nullptr;
        raygenPG_ = nullptr;
        module_ = nullptr;

        gasHandle_ = 0;
        cachedNumTriangles_ = 0;
    }

    void createModule(const std::string& ptxPath) {
        std::ifstream ptxFile(ptxPath);
        if (!ptxFile) {
            std::cerr << "Failed to open PTX file: " << ptxPath << "\n";
            std::exit(1);
        }

        std::string ptx((std::istreambuf_iterator<char>(ptxFile)), std::istreambuf_iterator<char>());

        OptixModuleCompileOptions moduleCompileOptions = {};
        moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pipelineCompileOptions_ = {};
        pipelineCompileOptions_.usesMotionBlur = false;
        pipelineCompileOptions_.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pipelineCompileOptions_.numPayloadValues = 2;
        pipelineCompileOptions_.numAttributeValues = 0;
        pipelineCompileOptions_.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
        pipelineCompileOptions_.pipelineLaunchParamsVariableName = "params";

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixModuleCreate(
            context_,
            &moduleCompileOptions,
            &pipelineCompileOptions_,
            ptx.c_str(),
            ptx.size(),
            log,
            &sizeof_log,
            &module_
        ));
    }

    void createProgramGroups(const char* raygen = "__raygen__triangle_visibility_row",
                             const char* miss = "__miss__hemisphere",
                             const char* closesthit = "__closesthit__hemisphere",
                             const char* anyhit = "__anyhit__hemisphere") {
        char log[2048];
        size_t sizeof_log = sizeof(log);

        OptixProgramGroupOptions pgOptions = {};

        OptixProgramGroupDesc raygenPGDesc = {};
        raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        raygenPGDesc.raygen.module = module_;
        raygenPGDesc.raygen.entryFunctionName = raygen;
        OPTIX_CHECK(optixProgramGroupCreate(context_, &raygenPGDesc, 1, &pgOptions, log, &sizeof_log, &raygenPG_));

        OptixProgramGroupDesc missPGDesc = {};
        missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        missPGDesc.miss.module = module_;
        missPGDesc.miss.entryFunctionName = miss;
        OPTIX_CHECK(optixProgramGroupCreate(context_, &missPGDesc, 1, &pgOptions, log, &sizeof_log, &missPG_));

        OptixProgramGroupDesc hitgroupPGDesc = {};
        hitgroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hitgroupPGDesc.hitgroup.moduleCH = module_;
        hitgroupPGDesc.hitgroup.entryFunctionNameCH = closesthit;
        hitgroupPGDesc.hitgroup.moduleAH = module_;
        hitgroupPGDesc.hitgroup.entryFunctionNameAH = anyhit;
        OPTIX_CHECK(optixProgramGroupCreate(context_, &hitgroupPGDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPG_));
    }

    void createPipeline() {
        OptixProgramGroup programGroups[] = { raygenPG_, missPG_, hitgroupPG_ };

        OptixPipelineLinkOptions pipelineLinkOptions = {};
        pipelineLinkOptions.maxTraceDepth = 1;

        char log[2048];
        size_t sizeof_log = sizeof(log);
        OPTIX_CHECK(optixPipelineCreate(
            context_,
            &pipelineCompileOptions_,
            &pipelineLinkOptions,
            programGroups,
            sizeof(programGroups) / sizeof(programGroups[0]),
            log,
            &sizeof_log,
            &pipeline_
        ));
    }

    void createSBT() {
        struct RaygenRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
        struct MissRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
        struct HitGroupRecord { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };

        RaygenRecord rg = {};
        MissRecord ms = {};
        HitGroupRecord hg = {};

        OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_, &rg));
        OPTIX_CHECK(optixSbtRecordPackHeader(missPG_, &ms));
        OPTIX_CHECK(optixSbtRecordPackHeader(hitgroupPG_, &hg));

        CUDA_CHECK(cudaMalloc((void**)&sbt_.raygenRecord, sizeof(RaygenRecord)));
        CUDA_CHECK(cudaMemcpy((void*)sbt_.raygenRecord, &rg, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&sbt_.missRecordBase, sizeof(MissRecord)));
        CUDA_CHECK(cudaMemcpy((void*)sbt_.missRecordBase, &ms, sizeof(MissRecord), cudaMemcpyHostToDevice));
        sbt_.missRecordStrideInBytes = sizeof(MissRecord);
        sbt_.missRecordCount = 1;

        CUDA_CHECK(cudaMalloc((void**)&sbt_.hitgroupRecordBase, sizeof(HitGroupRecord)));
        CUDA_CHECK(cudaMemcpy((void*)sbt_.hitgroupRecordBase, &hg, sizeof(HitGroupRecord), cudaMemcpyHostToDevice));
        sbt_.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
        sbt_.hitgroupRecordCount = 1;
    }

    void buildGAS(const Mesh& mesh) {
        if (d_vertices_) CUDA_CHECK(cudaFree((void*)d_vertices_));
        if (d_indices_) CUDA_CHECK(cudaFree((void*)d_indices_));
        if (d_gasOutput_) CUDA_CHECK(cudaFree((void*)d_gasOutput_));

        const size_t vertexBytes = mesh.vertices.size() * sizeof(Vertex);
        const size_t indexBytes = mesh.indices.size() * sizeof(TriIdx);

        CUDA_CHECK(cudaMalloc((void**)&d_vertices_, vertexBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_vertices_, mesh.vertices.data(), vertexBytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&d_indices_, indexBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_indices_, mesh.indices.data(), indexBytes, cudaMemcpyHostToDevice));

        OptixBuildInput buildInput = {};
        buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        buildInput.triangleArray.vertexStrideInBytes = sizeof(Vertex);
        buildInput.triangleArray.numVertices = static_cast<unsigned int>(mesh.vertices.size());
        buildInput.triangleArray.vertexBuffers = &d_vertices_;
        buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        buildInput.triangleArray.indexStrideInBytes = sizeof(TriIdx);
        buildInput.triangleArray.numIndexTriplets = static_cast<unsigned int>(mesh.indices.size());
        buildInput.triangleArray.indexBuffer = d_indices_;
        uint32_t buildInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
        buildInput.triangleArray.flags = &buildInputFlags;
        buildInput.triangleArray.numSbtRecords = 1;

        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes sizes = {};
        OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &accelOptions, &buildInput, 1, &sizes));

        CUDA_CHECK(cudaMalloc((void**)&d_gasOutput_, sizes.outputSizeInBytes));
        CUdeviceptr d_temp = 0;
        CUDA_CHECK(cudaMalloc((void**)&d_temp, sizes.tempSizeInBytes));

        OPTIX_CHECK(optixAccelBuild(
            context_, stream_, &accelOptions, &buildInput, 1,
            d_temp, sizes.tempSizeInBytes,
            d_gasOutput_, sizes.outputSizeInBytes,
            &gasHandle_, nullptr, 0
        ));

        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaFree((void*)d_temp));
    }

    void ensureTriangleSoA(const Mesh& mesh) {
        const uint32_t numTriangles = static_cast<uint32_t>(mesh.indices.size());
        if (cachedNumTriangles_ == numTriangles && d_cx_ && d_nx_ && d_area_) return;

        if (d_area_) CUDA_CHECK(cudaFree((void*)d_area_));
        if (d_nz_) CUDA_CHECK(cudaFree((void*)d_nz_));
        if (d_ny_) CUDA_CHECK(cudaFree((void*)d_ny_));
        if (d_nx_) CUDA_CHECK(cudaFree((void*)d_nx_));
        if (d_cz_) CUDA_CHECK(cudaFree((void*)d_cz_));
        if (d_cy_) CUDA_CHECK(cudaFree((void*)d_cy_));
        if (d_cx_) CUDA_CHECK(cudaFree((void*)d_cx_));

        const size_t floatBytes = numTriangles * sizeof(float);
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

        CUDA_CHECK(cudaMalloc((void**)&d_cx_, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_cy_, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_cz_, floatBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_cx_, cx.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_cy_, cy.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_cz_, cz.data(), floatBytes, cudaMemcpyHostToDevice));

        CUDA_CHECK(cudaMalloc((void**)&d_nx_, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_ny_, floatBytes));
        CUDA_CHECK(cudaMalloc((void**)&d_nz_, floatBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_nx_, nx.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_ny_, ny.data(), floatBytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy((void*)d_nz_, nz.data(), floatBytes, cudaMemcpyHostToDevice));

        // Upload per-triangle area.
        std::vector<float> area(numTriangles);
        for (uint32_t i = 0; i < numTriangles; ++i) {
            area[i] = mesh.triangle_area[i];
        }
        CUDA_CHECK(cudaMalloc((void**)&d_area_, floatBytes));
        CUDA_CHECK(cudaMemcpy((void*)d_area_, area.data(), floatBytes, cudaMemcpyHostToDevice));

        cachedNumTriangles_ = numTriangles;
    }

    void computeFormFactorRow(const Mesh& mesh,
                              uint32_t patchId,
                              std::vector<float>& row,
                              uint32_t originSamples = 1,
                              uint32_t dirSamples = 256) {
        const uint32_t numTriangles = static_cast<uint32_t>(mesh.indices.size());
        row.assign(numTriangles, 0.0f);

        if (d_launchParams_) CUDA_CHECK(cudaFree((void*)d_launchParams_));
        ensureTriangleSoA(mesh);

        CUdeviceptr d_row = 0;
        CUDA_CHECK(cudaMalloc((void**)&d_row, numTriangles * sizeof(float)));
        CUDA_CHECK(cudaMemset((void*)d_row, 0, numTriangles * sizeof(float)));

        struct LaunchParams {
            float3* vertices;
            uint3* indices;
            float* cx;
            float* cy;
            float* cz;
            float* nx;
            float* ny;
            float* nz;
            float* area;
            uint32_t numTriangles;
            uint32_t originSamples;
            uint32_t dirSamples;
            float* formFactors;
            float sceneEpsilon;
            float distanceSoftening;
            uint32_t basePatchId;
            float* rowOutput;
            OptixTraversableHandle gasHandle;
        };

        LaunchParams params = {};
        params.vertices = (float3*)d_vertices_;
        params.indices = (uint3*)d_indices_;
        params.cx = (float*)d_cx_;
        params.cy = (float*)d_cy_;
        params.cz = (float*)d_cz_;
        params.nx = (float*)d_nx_;
        params.ny = (float*)d_ny_;
        params.nz = (float*)d_nz_;
        params.area = (float*)d_area_;
        params.numTriangles = numTriangles;
        params.originSamples = originSamples;
        params.dirSamples = dirSamples;
        params.formFactors = nullptr;
        params.sceneEpsilon = computeSceneEpsilon(mesh);
        params.distanceSoftening = kDistanceSoftening;
        params.basePatchId = patchId;
        params.rowOutput = (float*)d_row;
        params.gasHandle = gasHandle_;

        CUDA_CHECK(cudaMalloc((void**)&d_launchParams_, sizeof(LaunchParams)));
        CUDA_CHECK(cudaMemcpy((void*)d_launchParams_, &params, sizeof(LaunchParams), cudaMemcpyHostToDevice));

        OPTIX_CHECK(optixLaunch(
            pipeline_,
            stream_,
            d_launchParams_,
            sizeof(LaunchParams),
            &sbt_,
            numTriangles,
            dirSamples,
            1
        ));

        CUDA_CHECK(cudaStreamSynchronize(stream_));
        CUDA_CHECK(cudaMemcpy(row.data(), (void*)d_row, numTriangles * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree((void*)d_row));
    }
};

} // namespace OptiXContext
