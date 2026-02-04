#pragma once

#ifdef USE_OPTIX

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <cuda.h>
#include <iostream>
#include <vector>
#include <cstring>

#include "OptiXLaunchParams.h"

namespace radiosity {
namespace visibility {

// Helper to check CUDA driver API errors
#define CUDA_CHECK(call) \
    do { \
        CUresult result = call; \
        if (result != CUDA_SUCCESS) { \
            const char* errorStr; \
            cuGetErrorString(result, &errorStr); \
            std::cerr << "CUDA error: " << errorStr << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
            return false; \
        } \
    } while(0)

/**
 * OptiX Context Manager
 * 
 * Handles OptiX initialization and provides ray tracing capabilities
 * Week 3: Basic visibility testing implementation
 */
class OptiXContext {
public:
    using TriangleData = radiosity::visibility::TriangleData;
    using RayGenParams = radiosity::visibility::RayGenParams;

    OptiXContext() : context(nullptr), gasHandle(0), 
                     d_vertexBuffer(0), d_indexBuffer(0),
                     pipeline(nullptr), module(nullptr),
                     d_sbtBuffer(0), d_paramsBuffer(0),
                     d_resultsBuffer(0), paramsBufferSize(0),
                     resultsBufferSize(0) {
    }
    
    ~OptiXContext() {
        cleanup();
    }
    
    /**
     * Initialize OptiX and create context
     */
    bool initialize() {
        try {
            // Initialize CUDA driver API
            std::cout << "  Initializing CUDA driver API...\n";
            CUresult cuRes = cuInit(0);
            if (cuRes != CUDA_SUCCESS) {
                const char* errorStr = "unknown";
                cuGetErrorString(cuRes, &errorStr);
                std::cerr << "CUDA initialization failed: " << errorStr << "\n";
                return false;
            }
            std::cout << "  ✓ CUDA driver initialized\n";
            
            // Create CUDA device and context
            std::cout << "  Creating CUDA context...\n";
            CUdevice cuDevice;
            CUDA_CHECK(cuDeviceGet(&cuDevice, 0)); // Use device 0
            
            CUcontext cuContext;
            CUDA_CHECK(cuCtxCreate(&cuContext, 0, cuDevice));
            std::cout << "  ✓ CUDA context created\n";
            
            // Initialize OptiX function table
            std::cout << "  Initializing OptiX...\n";
            OptixResult optixRes = optixInit();
            if (optixRes != OPTIX_SUCCESS) {
                std::cerr << "OptiX initialization failed with code: " << optixRes << "\n";
                std::cerr << "Note: OptiX requires NVIDIA GPU with driver 465.84+ (you have " << "581.57" << ")\n";
                std::cerr << "OptiX SDK at: " << (getenv("OPTIX_ROOT") ? getenv("OPTIX_ROOT") : "not set") << "\n";
                return false;
            }
            std::cout << "  ✓ OptiX initialized\n";
            
            // Create OptiX device context
            OptixDeviceContextOptions options = {};
            options.logCallbackFunction = &OptiXContext::logCallback;
            options.logCallbackLevel = 4; // Warnings and errors
            
            OptixResult res = optixDeviceContextCreate(cuContext, &options, &context);
            
            if (res != OPTIX_SUCCESS) {
                std::cerr << "Failed to create OptiX context: " << res << "\n";
                return false;
            }
            
            std::cout << "✓ OptiX context created successfully\n";
            
            // Create pipeline
            if (!createPipeline()) {
                std::cerr << "Failed to create OptiX pipeline\n";
                return false;
            }
            
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Exception during OptiX initialization: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * Upload mesh geometry to GPU and build acceleration structure
     */
    bool uploadMesh(const float* vertices, size_t vertexCount,
                   const uint32_t* indices, size_t triangleCount) {
        try {
            if (vertexCount == 0 || triangleCount == 0) {
                std::cerr << "OptiX upload skipped: empty mesh data\n";
                gasHandle = 0;
                return true;
            }

            if (d_vertexBuffer) {
                CUDA_CHECK(cuMemFree(d_vertexBuffer));
                d_vertexBuffer = 0;
            }
            if (d_indexBuffer) {
                CUDA_CHECK(cuMemFree(d_indexBuffer));
                d_indexBuffer = 0;
            }

            // Upload vertices
            size_t vertexBytes = vertexCount * 3 * sizeof(float);
            CUDA_CHECK(cuMemAlloc(&d_vertexBuffer, vertexBytes));
            CUDA_CHECK(cuMemcpyHtoD(d_vertexBuffer, vertices, vertexBytes));
            
            // Upload indices
            size_t indexBytes = triangleCount * 3 * sizeof(uint32_t);
            CUDA_CHECK(cuMemAlloc(&d_indexBuffer, indexBytes));
            CUDA_CHECK(cuMemcpyHtoD(d_indexBuffer, indices, indexBytes));
            
            // Build acceleration structure
            if (!buildGAS(vertexCount, triangleCount)) {
                return false;
            }
            
            std::cout << "✓ Mesh uploaded to GPU (" << vertexCount << " vertices, " 
                      << triangleCount << " triangles)\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Exception during mesh upload: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * Test visibility between two points using OptiX ray tracing
     * Returns 1.0 if visible, 0.0 if occluded
     * @param source_normal Normal of the source patch (for hemisphere check)
     * @param target_patch_id ID of target patch to ignore in hit test
     */
    float traceRay(const float origin[3], const float direction[3], const float source_normal[3], float tmax, int target_patch_id) const {
        static int call_count = 0;
        static bool first_call = true;
        bool is_first = first_call;
        call_count++;
        
        if (first_call) {
            std::cout << "[OptiX Ray Tracing] First call to traceRay\n";
            std::cout << "  Pipeline: " << (pipeline ? "valid" : "NULL") << "\n";
            std::cout << "  SBT Buffer: " << (d_sbtBuffer ? "valid" : "NULL") << "\n";
            std::cout << "  GAS Handle: " << gasHandle << "\n";
            std::cout << "  Ray origin: (" << origin[0] << ", " << origin[1] << ", " << origin[2] << ")\n";
            std::cout << "  Ray direction: (" << direction[0] << ", " << direction[1] << ", " << direction[2] << ")\n";
            std::cout << "  Ray tmax: " << tmax << "\n";
            first_call = false;
        }
        
        if (call_count <= 3) {
            std::cout << "  traceRay call #" << call_count 
                      << " tmax=" << tmax << " dir=(" << direction[0] << "," << direction[1] << "," << direction[2] << ")\n";
        }
        
        if (!pipeline || !d_sbtBuffer) {
            // Pipeline not ready, return stub value
            if (is_first) std::cout << "  Pipeline not ready, using stub\n";
            return 1.0f;
        }
        
        // Allocate result buffer on GPU (use unsigned int)
        unsigned int hostResult = 0;
        CUdeviceptr d_result = 0;
        CUresult cuRes = cuMemAlloc(&d_result, sizeof(unsigned int));
        if (cuRes != CUDA_SUCCESS) {
            std::cerr << "Failed to allocate result buffer\n";
            return 1.0f;
        }
        
        // Initialize result to 0
        cuMemsetD32(d_result, 0, 1);
        
        // Setup launch parameters (must match CUDA kernel struct)
        struct RayGenParams {
            float origin[3];
            float direction[3];
            float source_normal[3];
            float tmax;
            int target_patch_id;
            OptixTraversableHandle traversable;
            unsigned int* result;
        };
        
        RayGenParams params;
        params.origin[0] = origin[0];
        params.origin[1] = origin[1];
        params.origin[2] = origin[2];
        params.direction[0] = direction[0];
        params.direction[1] = direction[1];
        params.direction[2] = direction[2];
        params.source_normal[0] = source_normal[0];
        params.source_normal[1] = source_normal[1];
        params.source_normal[2] = source_normal[2];
        params.tmax = tmax;
        params.target_patch_id = target_patch_id;
        params.traversable = gasHandle;
        params.result = reinterpret_cast<unsigned int*>(d_result);
        
        // Upload params to GPU
        CUdeviceptr d_params = 0;
        cuRes = cuMemAlloc(&d_params, sizeof(RayGenParams));
        if (cuRes != CUDA_SUCCESS) {
            cuMemFree(d_result);
            return 1.0f;
        }
        
        cuMemcpyHtoD(d_params, &params, sizeof(RayGenParams));
        
        // Launch ray tracing
        OptixResult res = optixLaunch(
            pipeline,
            0, // CUDA stream
            d_params,
            sizeof(RayGenParams),
            &sbt,
            1, // width (single ray)
            1, // height
            1  // depth
        );
        
        if (res != OPTIX_SUCCESS) {
            std::cerr << "optixLaunch failed with code: " << res << "\n";
            cuMemFree(d_result);
            cuMemFree(d_params);
            return 1.0f; // Assume visible on error
        }
        
        // Synchronize and read result
        cuCtxSynchronize();
        cuMemcpyDtoH(&hostResult, d_result, sizeof(unsigned int));
        
        if (is_first) {
            std::cout << "  optixLaunch succeeded\n";
            std::cout << "  Result from GPU: " << hostResult << " (1=visible, 0=occluded)\n";
        }
        
        if (call_count <= 3) {
            std::cout << "    -> Result: " << hostResult << "\n";
        }
        
        // Cleanup
        cuMemFree(d_result);
        cuMemFree(d_params);
        
        // Convert to float (1 = visible, 0 = occluded)
        float visibility = (hostResult == 1u) ? 1.0f : 0.0f;
        if (is_first) {
            std::cout << "  Final visibility: " << visibility << "\n";
        }
        return visibility;
    }

    /**
     * Launch a single visibility ray between sampled points on two triangles.
     * Returns 1 if the path is unobstructed, 0 otherwise.
     */
    unsigned int traceTriangleSample(
        const TriangleData& sourceTriangle,
        const TriangleData& targetTriangle,
        const float sourceUV[2],
        const float targetUV[2]) const {
        if (!pipeline || !d_sbtBuffer) {
            return 1u;
        }

        if (!ensureParamsBuffer(sizeof(RayGenParams)) || !ensureResultsBuffer(sizeof(unsigned int))) {
            return 1u;
        }

        RayGenParams params = {};
        memcpy(&params.source, &sourceTriangle, sizeof(TriangleData));
        memcpy(&params.target, &targetTriangle, sizeof(TriangleData));
        params.source_uv[0] = sourceUV[0];
        params.source_uv[1] = sourceUV[1];
        params.target_uv[0] = targetUV[0];
        params.target_uv[1] = targetUV[1];
        params.sample_count = 1;
        params.result_offset = 0;
        params.traversable = gasHandle;
        params.results = reinterpret_cast<unsigned int*>(d_resultsBuffer);

        if (cuMemcpyHtoD(d_paramsBuffer, &params, sizeof(RayGenParams)) != CUDA_SUCCESS) {
            std::cerr << "Failed to upload ray generation parameters\n";
            return 1u;
        }
        if (cuMemsetD32(d_resultsBuffer, 0, 1) != CUDA_SUCCESS) {
            std::cerr << "Failed to clear results buffer\n";
            return 1u;
        }

        OptixResult res = optixLaunch(
            pipeline,
            0,
            d_paramsBuffer,
            sizeof(RayGenParams),
            &sbt,
            1, 1, 1
        );

        if (res != OPTIX_SUCCESS) {
            std::cerr << "optixLaunch failed with code: " << res << "\n";
            return 1u;
        }

        cuCtxSynchronize();

        unsigned int hostResult = 1u;
        if (cuMemcpyDtoH(&hostResult, d_resultsBuffer, sizeof(unsigned int)) != CUDA_SUCCESS) {
            std::cerr << "Failed to read ray result from device\n";
            return 1u;
        }
        return hostResult;
    }
    
    bool isInitialized() const {
        return context != nullptr && gasHandle != 0;
    }
    
private:
    OptixDeviceContext context;
    OptixTraversableHandle gasHandle;
    CUdeviceptr d_vertexBuffer;
    CUdeviceptr d_indexBuffer;
    CUdeviceptr d_gasBuffer;
    
    // Pipeline components (mutable for const traceRay)
    mutable OptixPipeline pipeline;
    mutable OptixModule module;
    mutable OptixShaderBindingTable sbt;
    mutable CUdeviceptr d_sbtBuffer;
    mutable CUdeviceptr d_paramsBuffer;
    mutable CUdeviceptr d_resultsBuffer;
    mutable size_t paramsBufferSize;
    mutable size_t resultsBufferSize;

    bool ensureParamsBuffer(size_t requiredSize) const {
        return ensureBuffer(d_paramsBuffer, requiredSize, paramsBufferSize);
    }

    bool ensureResultsBuffer(size_t requiredSize) const {
        return ensureBuffer(d_resultsBuffer, requiredSize, resultsBufferSize);
    }

    bool ensureBuffer(CUdeviceptr& buffer, size_t requiredSize, size_t& currentSize) const {
        if (buffer != 0 && currentSize >= requiredSize) {
            return true;
        }
        if (buffer != 0) {
            cuMemFree(buffer);
            buffer = 0;
            currentSize = 0;
        }
        if (cuMemAlloc(&buffer, requiredSize) != CUDA_SUCCESS) {
            std::cerr << "Failed to allocate device buffer of size " << requiredSize << " bytes\n";
            return false;
        }
        currentSize = requiredSize;
        return true;
    }
    
    /**
     * Create OptiX pipeline with ray tracing programs
     */
    bool createPipeline() {
        try {
            // Load PTX from file
            const char* ptxPath = "optix_kernels.ptx";
            std::vector<char> ptxCode = loadPTXFile(ptxPath);
            
            if (ptxCode.empty()) {
                std::cerr << "Failed to load PTX file: " << ptxPath << "\n";
                return false;
            }
            
            // Create module from PTX
            OptixModuleCompileOptions moduleCompileOptions = {};
            moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
            moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
            moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
            
            OptixPipelineCompileOptions pipelineCompileOptions = {};
            pipelineCompileOptions.usesMotionBlur = false;
            pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
            pipelineCompileOptions.numPayloadValues = 1;
            pipelineCompileOptions.numAttributeValues = 0;
            pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
            pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";
            
            char log[2048];
            size_t logSize = sizeof(log);
            
            OptixResult res = optixModuleCreate(
                context,
                &moduleCompileOptions,
                &pipelineCompileOptions,
                ptxCode.data(),
                ptxCode.size(),
                log, &logSize,
                &module
            );
            
            if (res != OPTIX_SUCCESS) {
                std::cerr << "Failed to create module: " << res << "\n";
                if (logSize > 0) std::cerr << "Log: " << log << "\n";
                return false;
            }
            
            // Create program groups
            OptixProgramGroup raygenPG, missPG, hitgroupPG;
            
            // Raygen program
            OptixProgramGroupOptions pgOptions = {};
            OptixProgramGroupDesc raygenPGDesc = {};
            raygenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
            raygenPGDesc.raygen.module = module;
            raygenPGDesc.raygen.entryFunctionName = "__raygen__visibility";
            
            logSize = sizeof(log);
            res = optixProgramGroupCreate(context, &raygenPGDesc, 1, &pgOptions, log, &logSize, &raygenPG);
            if (res != OPTIX_SUCCESS) {
                std::cerr << "Failed to create raygen program group: " << res << "\n";
                return false;
            }
            
            // Miss program
            OptixProgramGroupDesc missPGDesc = {};
            missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
            missPGDesc.miss.module = module;
            missPGDesc.miss.entryFunctionName = "__miss__visibility";
            
            logSize = sizeof(log);
            res = optixProgramGroupCreate(context, &missPGDesc, 1, &pgOptions, log, &logSize, &missPG);
            if (res != OPTIX_SUCCESS) {
                std::cerr << "Failed to create miss program group: " << res << "\n";
                return false;
            }
            
            // Hit group
            OptixProgramGroupDesc hitgroupPGDesc = {};
            hitgroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            hitgroupPGDesc.hitgroup.moduleCH = module;
            hitgroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__occlusion";
            
            logSize = sizeof(log);
            res = optixProgramGroupCreate(context, &hitgroupPGDesc, 1, &pgOptions, log, &logSize, &hitgroupPG);
            if (res != OPTIX_SUCCESS) {
                std::cerr << "Failed to create hitgroup program group: " << res << "\n";
                return false;
            }
            
            // Link pipeline
            OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };
            
            OptixPipelineLinkOptions pipelineLinkOptions = {};
            pipelineLinkOptions.maxTraceDepth = 1;
            
            logSize = sizeof(log);
            res = optixPipelineCreate(
                context,
                &pipelineCompileOptions,
                &pipelineLinkOptions,
                programGroups, 3,
                log, &logSize,
                &pipeline
            );
            
            if (res != OPTIX_SUCCESS) {
                std::cerr << "Failed to create pipeline: " << res << "\n";
                if (logSize > 0) std::cerr << "Log: " << log << "\n";
                return false;
            }
            
            // Build SBT (Shader Binding Table)
            if (!buildSBT(raygenPG, missPG, hitgroupPG)) {
                return false;
            }
            
            std::cout << "✓ OptiX pipeline created\n";
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "Exception creating pipeline: " << e.what() << "\n";
            return false;
        }
    }
    
    /**
     * Build Shader Binding Table
     */
    bool buildSBT(OptixProgramGroup raygenPG, OptixProgramGroup missPG, OptixProgramGroup hitgroupPG) {
        // SBT record structure
        struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
            char header[OPTIX_SBT_RECORD_HEADER_SIZE];
        };
        
        // Raygen record
        SbtRecord raygenRecord;
        optixSbtRecordPackHeader(raygenPG, &raygenRecord);
        
        // Miss record
        SbtRecord missRecord;
        optixSbtRecordPackHeader(missPG, &missRecord);
        
        // Hitgroup record
        SbtRecord hitgroupRecord;
        optixSbtRecordPackHeader(hitgroupPG, &hitgroupRecord);
        
        // Allocate SBT buffer
        size_t sbtSize = sizeof(SbtRecord) * 3;
        CUDA_CHECK(cuMemAlloc(&d_sbtBuffer, sbtSize));
        
        // Upload records
        CUDA_CHECK(cuMemcpyHtoD(d_sbtBuffer, &raygenRecord, sizeof(SbtRecord)));
        CUDA_CHECK(cuMemcpyHtoD(d_sbtBuffer + sizeof(SbtRecord), &missRecord, sizeof(SbtRecord)));
        CUDA_CHECK(cuMemcpyHtoD(d_sbtBuffer + sizeof(SbtRecord) * 2, &hitgroupRecord, sizeof(SbtRecord)));
        
        // Setup SBT
        sbt = {};
        sbt.raygenRecord = d_sbtBuffer;
        sbt.missRecordBase = d_sbtBuffer + sizeof(SbtRecord);
        sbt.missRecordStrideInBytes = sizeof(SbtRecord);
        sbt.missRecordCount = 1;
        sbt.hitgroupRecordBase = d_sbtBuffer + sizeof(SbtRecord) * 2;
        sbt.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
        sbt.hitgroupRecordCount = 1;
        
        return true;
    }
    
    /**
     * Load PTX file
     */
    std::vector<char> loadPTXFile(const char* filename) {
        std::vector<char> ptxCode;
        
        FILE* fp = fopen(filename, "rb");
        if (!fp) {
            std::cerr << "Could not open PTX file: " << filename << "\n";
            return ptxCode;
        }
        
        fseek(fp, 0, SEEK_END);
        size_t fileSize = ftell(fp);
        fseek(fp, 0, SEEK_SET);
        
        ptxCode.resize(fileSize);
        fread(ptxCode.data(), 1, fileSize, fp);
        fclose(fp);
        
        return ptxCode;
    }
    
    /**
     * Build Geometry Acceleration Structure (BVH)
     */
    bool buildGAS(size_t vertexCount, size_t triangleCount) {
        // Configure triangle input
        OptixBuildInput triangleInput = {};
        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        
        uint32_t triangleFlags = OPTIX_GEOMETRY_FLAG_NONE;
        
        triangleInput.triangleArray.vertexBuffers = &d_vertexBuffer;
        triangleInput.triangleArray.numVertices = static_cast<uint32_t>(vertexCount);
        triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = 0; // Tightly packed
        
        triangleInput.triangleArray.indexBuffer = d_indexBuffer;
        triangleInput.triangleArray.numIndexTriplets = static_cast<uint32_t>(triangleCount);
        triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes = 0; // Tightly packed
        
        triangleInput.triangleArray.flags = &triangleFlags;
        triangleInput.triangleArray.numSbtRecords = 1;
        
        // Configure build options
        OptixAccelBuildOptions accelOptions = {};
        accelOptions.buildFlags = 
            OPTIX_BUILD_FLAG_ALLOW_COMPACTION |         // Enable memory compaction
            OPTIX_BUILD_FLAG_PREFER_FAST_TRACE |        // Optimize for ray tracing speed (radiosity needs many rays)
            OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS; // Allow vertex access in closest-hit (for future features)
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        
        // Query memory requirements
        OptixAccelBufferSizes gasBufferSizes;
        OptixResult res = optixAccelComputeMemoryUsage(
            context, &accelOptions, &triangleInput, 1, &gasBufferSizes);
        
        if (res != OPTIX_SUCCESS) {
            std::cerr << "Failed to compute GAS memory usage: " << res << "\n";
            return false;
        }
        
        // Allocate buffers
        CUdeviceptr d_tempBuffer;
        CUDA_CHECK(cuMemAlloc(&d_tempBuffer, gasBufferSizes.tempSizeInBytes));
        CUDA_CHECK(cuMemAlloc(&d_gasBuffer, gasBufferSizes.outputSizeInBytes));
        
        // Build acceleration structure
        res = optixAccelBuild(
            context,
            0, // CUDA stream
            &accelOptions,
            &triangleInput,
            1, // num build inputs
            d_tempBuffer,
            gasBufferSizes.tempSizeInBytes,
            d_gasBuffer,
            gasBufferSizes.outputSizeInBytes,
            &gasHandle,
            nullptr, // emitted property list
            0        // num emitted properties
        );
        
        // Free temp buffer
        cuMemFree(d_tempBuffer);
        
        if (res != OPTIX_SUCCESS) {
            std::cerr << "Failed to build GAS: " << res << "\n";
            return false;
        }
        
        std::cout << "✓ Acceleration structure built (GAS handle: " << gasHandle << ")\n";
        return true;
    }
    
    void cleanup() {
        if (d_sbtBuffer) {
            cuMemFree(d_sbtBuffer);
            d_sbtBuffer = 0;
        }
        if (d_paramsBuffer) {
            cuMemFree(d_paramsBuffer);
            d_paramsBuffer = 0;
        }
        if (d_resultsBuffer) {
            cuMemFree(d_resultsBuffer);
            d_resultsBuffer = 0;
        }
        paramsBufferSize = 0;
        resultsBufferSize = 0;
        if (pipeline) {
            optixPipelineDestroy(pipeline);
            pipeline = nullptr;
        }
        if (module) {
            optixModuleDestroy(module);
            module = nullptr;
        }
        if (d_vertexBuffer) {
            cuMemFree(d_vertexBuffer);
            d_vertexBuffer = 0;
        }
        if (d_indexBuffer) {
            cuMemFree(d_indexBuffer);
            d_indexBuffer = 0;
        }
        if (d_gasBuffer) {
            cuMemFree(d_gasBuffer);
            d_gasBuffer = 0;
        }
        if (context) {
            optixDeviceContextDestroy(context);
            context = nullptr;
        }
    }
    
    static void logCallback(unsigned int level, const char* tag, 
                           const char* message, void* /*cbdata*/) {
        std::cerr << "[OptiX][" << level << "][" << tag << "]: " 
                  << message << "\n";
    }
};

} // namespace visibility
} // namespace radiosity

#endif // USE_OPTIX
