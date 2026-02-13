#pragma once
// OptiX ray-traced renderer — renders the radiosity solution to a PNG image.
//
// Creates a separate OptiX pipeline from the form-factor one (3 payload values
// for RGB, closest-hit only, no any-hit).  Accepts a welded mesh with
// per-vertex colors, traces one primary ray per pixel through a pinhole camera,
// interpolates vertex colors via barycentrics, and writes a PNG via stb.

#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "../app/Config.h"

// stb_image_write — implementation lives in this header which is only ever
// compiled from a single translation unit (main.cpp).
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace Renderer {

// ---- CUDA / OptiX error helpers (prefixed to avoid collision with OptiXContext) ----

#define RENDER_CUDA_CHECK(call)                                                 \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error [Renderer]: " << cudaGetErrorString(err)   \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

#define RENDER_OPTIX_CHECK(call)                                                \
    do {                                                                        \
        OptixResult res = call;                                                 \
        if (res != OPTIX_SUCCESS) {                                             \
            std::cerr << "OptiX error [Renderer]: " << optixGetErrorName(res)   \
                      << " (" << __FILE__ << ":" << __LINE__ << ")\n";          \
            std::exit(1);                                                       \
        }                                                                       \
    } while (0)

// ---- RenderParams — must match the struct in RenderKernels.cu exactly ----

struct RenderParams {
    float4* framebuffer;
    uint32_t width;
    uint32_t height;

    float3 eye;
    float3 U;       // camera right  (scaled)
    float3 V;       // camera up     (scaled, negated for top-down scanlines)
    float3 W;       // camera forward

    float3* vertexColors;
    uint3*  indices;
    OptixTraversableHandle gasHandle;
};

// ---- Camera parameters ----

struct Camera {
    float3 eye;
    float3 lookAt;
    float3 up;
    float  fovY;    // vertical FOV in degrees
};

// Default front view (classic Cornell spec).
inline Camera defaultCamera() {
    return { make_float3(kCameraEyeX, kCameraEyeY, kCameraEyeZ),
             make_float3(0.0f, 0.0f, 0.0f),
             make_float3(0.0f, 1.0f, 0.0f),
             kCameraFovY };
}

// ---- Main renderer class ----

class RayTracedRenderer {
public:
    RayTracedRenderer() = default;
    ~RayTracedRenderer() { cleanup(); }

    /// Initialise the renderer from a welded mesh and per-vertex colors.
    /// @param ptxPath  Path to compiled render_kernels.ptx.
    /// @param positions  Welded vertex positions (float3).
    /// @param indices    Triangle index buffer (uint3).
    /// @param colors     Per-vertex colours (float3), same count as positions.
    void initialize(const std::string& ptxPath,
                    const std::vector<float3>& positions,
                    const std::vector<uint3>& indices,
                    const std::vector<float3>& colors)
    {
        // OptiX device context (reuses the CUDA primary context).
        RENDER_CUDA_CHECK(cudaFree(nullptr));  // ensure CUDA is initialised
        optixInit();                            // safe to call again

        OptixDeviceContextOptions ctxOpts = {};
        ctxOpts.logCallbackFunction = [](unsigned int level, const char*,
                                         const char* msg, void*) {
            if (level <= 2) std::cerr << "[OptiX] " << msg << "\n";
        };
        ctxOpts.logCallbackLevel = 2;

        CUcontext cuCtx = 0;
        RENDER_OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &ctxOpts, &context_));
        RENDER_CUDA_CHECK(cudaStreamCreate(&stream_));

        createModule(ptxPath);
        createProgramGroups();
        createPipeline();
        createSBT();
        buildGAS(positions, indices);
        uploadBuffer(colors, d_colors_);

        // Upload the index buffer used inside the closest-hit shader.
        uploadBuffer(indices, d_shaderIndices_);
    }

    /// Render and save a PNG to `outputPath`.
    void renderAndSave(const std::string& outputPath,
                       const Camera& cam = defaultCamera()) const {
        const uint32_t W = kRenderWidth;
        const uint32_t H = kRenderHeight;

        // Device framebuffer ---------------------------------------------------
        CUdeviceptr d_fb = 0;
        RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fb),
                                     W * H * sizeof(float4)));

        // Camera ---------------------------------------------------------------
        float3 fwd = make_float3(cam.lookAt.x - cam.eye.x,
                                 cam.lookAt.y - cam.eye.y,
                                 cam.lookAt.z - cam.eye.z);
        {
            float len = sqrtf(fwd.x * fwd.x + fwd.y * fwd.y + fwd.z * fwd.z);
            fwd.x /= len; fwd.y /= len; fwd.z /= len;
        }

        // right = normalise(cross(fwd, cam.up))
        float3 right = make_float3(fwd.y * cam.up.z - fwd.z * cam.up.y,
                                   fwd.z * cam.up.x - fwd.x * cam.up.z,
                                   fwd.x * cam.up.y - fwd.y * cam.up.x);
        {
            float len = sqrtf(right.x * right.x + right.y * right.y + right.z * right.z);
            right.x /= len; right.y /= len; right.z /= len;
        }

        // camUp = cross(right, fwd)
        float3 camUp = make_float3(right.y * fwd.z - right.z * fwd.y,
                                   right.z * fwd.x - right.x * fwd.z,
                                   right.x * fwd.y - right.y * fwd.x);

        const float fovRad = cam.fovY * 3.14159265358979323846f / 180.0f;
        const float halfH  = tanf(fovRad * 0.5f);
        const float aspect = static_cast<float>(W) / static_cast<float>(H);
        const float halfW  = halfH * aspect;

        // Kernel camera vectors:
        //   U  =  right  * halfW   (horizontal half-extent of image plane)
        //   V  = -camUp  * halfH   (vertical, negated so py=0 → top of image)
        //   W  =  fwd              (look direction, unit-length)
        RenderParams rp = {};
        rp.framebuffer  = reinterpret_cast<float4*>(d_fb);
        rp.width        = W;
        rp.height       = H;
        rp.eye          = cam.eye;
        rp.U            = make_float3(right.x * halfW, right.y * halfW, right.z * halfW);
        rp.V            = make_float3(-camUp.x * halfH, -camUp.y * halfH, -camUp.z * halfH);
        rp.W            = fwd;
        rp.vertexColors = reinterpret_cast<float3*>(d_colors_);
        rp.indices      = reinterpret_cast<uint3*>(d_shaderIndices_);
        rp.gasHandle    = gasHandle_;

        CUdeviceptr d_params = 0;
        RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RenderParams)));
        RENDER_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &rp,
                                     sizeof(RenderParams), cudaMemcpyHostToDevice));

        // Launch ---------------------------------------------------------------
        RENDER_OPTIX_CHECK(optixLaunch(pipeline_, stream_,
                                       d_params, sizeof(RenderParams),
                                       &sbt_, W, H, 1));
        RENDER_CUDA_CHECK(cudaStreamSynchronize(stream_));

        // Read-back & convert to 8-bit RGBA -----------------------------------
        std::vector<float4> fb(W * H);
        RENDER_CUDA_CHECK(cudaMemcpy(fb.data(), reinterpret_cast<void*>(d_fb),
                                     W * H * sizeof(float4), cudaMemcpyDeviceToHost));
        RENDER_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_fb)));
        RENDER_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

        std::vector<uint8_t> pixels(W * H * 4);
        for (uint32_t i = 0; i < W * H; ++i) {
            auto clamp01 = [](float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
            pixels[i * 4 + 0] = static_cast<uint8_t>(clamp01(fb[i].x) * 255.0f + 0.5f);
            pixels[i * 4 + 1] = static_cast<uint8_t>(clamp01(fb[i].y) * 255.0f + 0.5f);
            pixels[i * 4 + 2] = static_cast<uint8_t>(clamp01(fb[i].z) * 255.0f + 0.5f);
            pixels[i * 4 + 3] = 255;
        }

        // Write PNG ------------------------------------------------------------
        stbi_flip_vertically_on_write(0);
        int ok = stbi_write_png(outputPath.c_str(),
                                static_cast<int>(W), static_cast<int>(H),
                                4, pixels.data(), static_cast<int>(W * 4));
        if (ok) {
            std::cout << "Rendered PNG: " << outputPath << " (" << W << "x" << H << ")\n";
        } else {
            std::cerr << "Failed to write PNG: " << outputPath << "\n";
        }
    }

private:
    OptixDeviceContext context_   = nullptr;
    CUstream           stream_    = nullptr;
    OptixModule        module_    = nullptr;
    OptixProgramGroup  raygenPG_  = nullptr;
    OptixProgramGroup  missPG_    = nullptr;
    OptixProgramGroup  hitPG_     = nullptr;
    OptixPipeline      pipeline_  = nullptr;
    OptixPipelineCompileOptions pcopts_ = {};
    OptixShaderBindingTable     sbt_    = {};

    CUdeviceptr d_gasVerts_      = 0;
    CUdeviceptr d_gasIndices_    = 0;
    CUdeviceptr d_gasOutput_     = 0;
    CUdeviceptr d_colors_        = 0;
    CUdeviceptr d_shaderIndices_ = 0;

    OptixTraversableHandle gasHandle_ = 0;

    // ---- helpers ----

    void cleanup() {
        auto safeFree = [](CUdeviceptr& p) { if (p) { cudaFree(reinterpret_cast<void*>(p)); p = 0; } };
        safeFree(d_colors_);
        safeFree(d_shaderIndices_);
        safeFree(d_gasIndices_);
        safeFree(d_gasVerts_);
        safeFree(d_gasOutput_);

        if (sbt_.raygenRecord)      cudaFree(reinterpret_cast<void*>(sbt_.raygenRecord));
        if (sbt_.missRecordBase)    cudaFree(reinterpret_cast<void*>(sbt_.missRecordBase));
        if (sbt_.hitgroupRecordBase) cudaFree(reinterpret_cast<void*>(sbt_.hitgroupRecordBase));
        sbt_ = {};

        if (pipeline_)  { optixPipelineDestroy(pipeline_);       pipeline_  = nullptr; }
        if (hitPG_)     { optixProgramGroupDestroy(hitPG_);      hitPG_     = nullptr; }
        if (missPG_)    { optixProgramGroupDestroy(missPG_);     missPG_    = nullptr; }
        if (raygenPG_)  { optixProgramGroupDestroy(raygenPG_);   raygenPG_  = nullptr; }
        if (module_)    { optixModuleDestroy(module_);           module_    = nullptr; }
        if (stream_)    { cudaStreamDestroy(stream_);            stream_    = nullptr; }
        if (context_)   { optixDeviceContextDestroy(context_);   context_   = nullptr; }
    }

    template <typename T>
    void uploadBuffer(const std::vector<T>& src, CUdeviceptr& dst) {
        const size_t bytes = src.size() * sizeof(T);
        RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), bytes));
        RENDER_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), src.data(),
                                     bytes, cudaMemcpyHostToDevice));
    }

    void createModule(const std::string& ptxPath) {
        std::ifstream f(ptxPath);
        if (!f) {
            std::cerr << "Renderer: cannot open PTX: " << ptxPath << "\n";
            std::exit(1);
        }
        std::string ptx((std::istreambuf_iterator<char>(f)),
                         std::istreambuf_iterator<char>());

        OptixModuleCompileOptions mco = {};
        mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
        mco.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
        mco.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

        pcopts_ = {};
        pcopts_.usesMotionBlur          = false;
        pcopts_.traversableGraphFlags   = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
        pcopts_.numPayloadValues        = 3;   // r, g, b
        pcopts_.numAttributeValues      = 2;   // built-in triangle barycentrics
        pcopts_.exceptionFlags          = OPTIX_EXCEPTION_FLAG_NONE;
        pcopts_.pipelineLaunchParamsVariableName = "params";

        char log[2048];
        size_t logSz = sizeof(log);
        RENDER_OPTIX_CHECK(optixModuleCreate(context_, &mco, &pcopts_,
                                             ptx.c_str(), ptx.size(),
                                             log, &logSz, &module_));
    }

    void createProgramGroups() {
        char log[2048]; size_t logSz;
        OptixProgramGroupOptions pgOpts = {};

        // Raygen
        OptixProgramGroupDesc rgDesc = {};
        rgDesc.kind                    = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        rgDesc.raygen.module           = module_;
        rgDesc.raygen.entryFunctionName = "__raygen__render";
        logSz = sizeof(log);
        RENDER_OPTIX_CHECK(optixProgramGroupCreate(context_, &rgDesc, 1, &pgOpts,
                                                   log, &logSz, &raygenPG_));

        // Miss
        OptixProgramGroupDesc msDesc = {};
        msDesc.kind                 = OPTIX_PROGRAM_GROUP_KIND_MISS;
        msDesc.miss.module          = module_;
        msDesc.miss.entryFunctionName = "__miss__render";
        logSz = sizeof(log);
        RENDER_OPTIX_CHECK(optixProgramGroupCreate(context_, &msDesc, 1, &pgOpts,
                                                   log, &logSz, &missPG_));

        // Hit group — closest-hit only (no any-hit for primary visibility)
        OptixProgramGroupDesc hgDesc = {};
        hgDesc.kind                           = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        hgDesc.hitgroup.moduleCH              = module_;
        hgDesc.hitgroup.entryFunctionNameCH   = "__closesthit__render";
        hgDesc.hitgroup.moduleAH              = nullptr;
        hgDesc.hitgroup.entryFunctionNameAH   = nullptr;
        logSz = sizeof(log);
        RENDER_OPTIX_CHECK(optixProgramGroupCreate(context_, &hgDesc, 1, &pgOpts,
                                                   log, &logSz, &hitPG_));
    }

    void createPipeline() {
        OptixProgramGroup groups[] = { raygenPG_, missPG_, hitPG_ };
        OptixPipelineLinkOptions linkOpts = {};
        linkOpts.maxTraceDepth = 1;

        char log[2048];
        size_t logSz = sizeof(log);
        RENDER_OPTIX_CHECK(optixPipelineCreate(context_, &pcopts_, &linkOpts,
                                               groups, 3, log, &logSz, &pipeline_));
    }

    void createSBT() {
        struct Record { char header[OPTIX_SBT_RECORD_HEADER_SIZE]; };
        Record rg{}, ms{}, hg{};
        RENDER_OPTIX_CHECK(optixSbtRecordPackHeader(raygenPG_, &rg));
        RENDER_OPTIX_CHECK(optixSbtRecordPackHeader(missPG_,   &ms));
        RENDER_OPTIX_CHECK(optixSbtRecordPackHeader(hitPG_,    &hg));

        auto upload = [&](const Record& rec, CUdeviceptr& dst) {
            RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), sizeof(Record)));
            RENDER_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), &rec,
                                         sizeof(Record), cudaMemcpyHostToDevice));
        };

        upload(rg, sbt_.raygenRecord);

        CUdeviceptr missPtr = 0;
        upload(ms, missPtr);
        sbt_.missRecordBase          = missPtr;
        sbt_.missRecordStrideInBytes = sizeof(Record);
        sbt_.missRecordCount         = 1;

        CUdeviceptr hitPtr = 0;
        upload(hg, hitPtr);
        sbt_.hitgroupRecordBase          = hitPtr;
        sbt_.hitgroupRecordStrideInBytes = sizeof(Record);
        sbt_.hitgroupRecordCount         = 1;
    }

    void buildGAS(const std::vector<float3>& positions,
                  const std::vector<uint3>&  indices) {
        uploadBuffer(positions, d_gasVerts_);
        uploadBuffer(indices,   d_gasIndices_);

        OptixBuildInput bi = {};
        bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        bi.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        bi.triangleArray.vertexStrideInBytes = sizeof(float3);
        bi.triangleArray.numVertices         = static_cast<unsigned>(positions.size());
        bi.triangleArray.vertexBuffers       = &d_gasVerts_;
        bi.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        bi.triangleArray.indexStrideInBytes  = sizeof(uint3);
        bi.triangleArray.numIndexTriplets    = static_cast<unsigned>(indices.size());
        bi.triangleArray.indexBuffer         = d_gasIndices_;
        uint32_t flags = OPTIX_GEOMETRY_FLAG_NONE;
        bi.triangleArray.flags               = &flags;
        bi.triangleArray.numSbtRecords       = 1;

        OptixAccelBuildOptions ao = {};
        ao.buildFlags = OPTIX_BUILD_FLAG_NONE;
        ao.operation  = OPTIX_BUILD_OPERATION_BUILD;

        OptixAccelBufferSizes sizes{};
        RENDER_OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &ao, &bi, 1, &sizes));

        RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gasOutput_),
                                     sizes.outputSizeInBytes));
        CUdeviceptr d_temp = 0;
        RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp),
                                     sizes.tempSizeInBytes));

        RENDER_OPTIX_CHECK(optixAccelBuild(context_, stream_, &ao, &bi, 1,
                                           d_temp, sizes.tempSizeInBytes,
                                           d_gasOutput_, sizes.outputSizeInBytes,
                                           &gasHandle_, nullptr, 0));
        RENDER_CUDA_CHECK(cudaStreamSynchronize(stream_));
        RENDER_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp)));
    }
};

} // namespace Renderer
