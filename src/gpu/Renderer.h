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
        auto pixels = renderToPixels(cam);
        writePNG(outputPath, pixels);
    }

    /// Render with wireframe + vertex dots overlaid and save a PNG.
    /// Uses the same ray-traced render as base, then projects the mesh
    /// edges and vertices onto the image as a CPU post-process.
    void renderWireframeAndSave(const std::string& outputPath,
                                const std::vector<float3>& positions,
                                const std::vector<uint3>& indices,
                                const Camera& cam = defaultCamera()) const {
        auto [pixels, depthBuf] = renderToPixelsWithDepth(cam);
        const uint32_t W = kRenderWidth;
        const uint32_t H = kRenderHeight;

        // Build camera projection (same math as renderToPixels).
        float3 fwd = make_float3(cam.lookAt.x - cam.eye.x,
                                 cam.lookAt.y - cam.eye.y,
                                 cam.lookAt.z - cam.eye.z);
        { float l = sqrtf(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
          fwd.x /= l; fwd.y /= l; fwd.z /= l; }

        float3 right = make_float3(fwd.y*cam.up.z - fwd.z*cam.up.y,
                                   fwd.z*cam.up.x - fwd.x*cam.up.z,
                                   fwd.x*cam.up.y - fwd.y*cam.up.x);
        { float l = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
          right.x /= l; right.y /= l; right.z /= l; }

        float3 camUp = make_float3(right.y*fwd.z - right.z*fwd.y,
                                   right.z*fwd.x - right.x*fwd.z,
                                   right.x*fwd.y - right.y*fwd.x);

        const float fovRad = cam.fovY * 3.14159265358979323846f / 180.0f;
        const float halfH  = tanf(fovRad * 0.5f);
        const float aspect = static_cast<float>(W) / static_cast<float>(H);
        const float halfW  = halfH * aspect;

        // Project a 3D world point → (px, py, depth).  Returns false if behind camera.
        auto project = [&](const float3& p, float& px, float& py, float& pz) -> bool {
            float3 d = make_float3(p.x - cam.eye.x, p.y - cam.eye.y, p.z - cam.eye.z);
            pz = d.x*fwd.x + d.y*fwd.y + d.z*fwd.z;
            if (pz <= 1e-4f) return false;
            float x = d.x*right.x + d.y*right.y + d.z*right.z;
            float y = d.x*camUp.x + d.y*camUp.y + d.z*camUp.z;
            float ndcX =  (x / pz) / halfW;
            float ndcY = -(y / pz) / halfH;
            px = (ndcX * 0.5f + 0.5f) * static_cast<float>(W);
            py = (ndcY * 0.5f + 0.5f) * static_cast<float>(H);
            // pz = distance from eye along the un-normalized ray direction.
            // Convert to actual distance from eye (for comparison with OptiX ray-t).
            pz = sqrtf(d.x*d.x + d.y*d.y + d.z*d.z);
            return true;
        };

        // Depth-bias: small tolerance so edges on the surface don't z-fight.
        constexpr float depthBias = 0.002f;

        // Alpha-blend a pixel at (x, y) only if pointDepth is in front of geometry.
        auto blendPixelZ = [&](int x, int y, float pointDepth,
                               uint8_t r, uint8_t g, uint8_t b, float alpha) {
            if (x < 0 || x >= (int)W || y < 0 || y >= (int)H) return;
            uint32_t pi = static_cast<uint32_t>(y) * W + static_cast<uint32_t>(x);
            if (pointDepth > depthBuf[pi] + depthBias) return;  // occluded
            uint32_t idx = pi * 4;
            float ia = 1.0f - alpha;
            pixels[idx + 0] = static_cast<uint8_t>(pixels[idx + 0] * ia + r * alpha + 0.5f);
            pixels[idx + 1] = static_cast<uint8_t>(pixels[idx + 1] * ia + g * alpha + 0.5f);
            pixels[idx + 2] = static_cast<uint8_t>(pixels[idx + 2] * ia + b * alpha + 0.5f);
        };

        // Bresenham line with depth-tested alpha blending.
        constexpr uint8_t lineR = 20, lineG = 20, lineB = 20;
        constexpr float   lineAlpha = 0.45f;
        auto drawLine = [&](float x0f, float y0f, float z0, float x1f, float y1f, float z1) {
            int x0 = static_cast<int>(x0f + 0.5f), y0 = static_cast<int>(y0f + 0.5f);
            int x1 = static_cast<int>(x1f + 0.5f), y1 = static_cast<int>(y1f + 0.5f);
            int dx = std::abs(x1 - x0), dy = -std::abs(y1 - y0);
            int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1;
            int totalSteps = std::max(dx, std::abs(dy));
            int err = dx + dy;
            int step = 0;
            for (;;) {
                float t = (totalSteps > 0) ? static_cast<float>(step) / static_cast<float>(totalSteps) : 0.0f;
                float z = z0 + t * (z1 - z0);
                blendPixelZ(x0, y0, z, lineR, lineG, lineB, lineAlpha);
                if (x0 == x1 && y0 == y1) break;
                int e2 = 2 * err;
                if (e2 >= dy) { err += dy; x0 += sx; }
                if (e2 <= dx) { err += dx; y0 += sy; }
                ++step;
            }
        };

        // Draw all triangle edges with per-pixel depth test.
        for (const auto& tri : indices) {
            float3 v[3] = { positions[tri.x], positions[tri.y], positions[tri.z] };
            float px[3], py[3], pz[3];
            bool ok[3];
            for (int k = 0; k < 3; ++k) ok[k] = project(v[k], px[k], py[k], pz[k]);
            if (ok[0] && ok[1]) drawLine(px[0], py[0], pz[0], px[1], py[1], pz[1]);
            if (ok[1] && ok[2]) drawLine(px[1], py[1], pz[1], px[2], py[2], pz[2]);
            if (ok[2] && ok[0]) drawLine(px[2], py[2], pz[2], px[0], py[0], pz[0]);
        }

        // Draw vertex dots (filled circles, radius 4 px, 50% alpha, depth-tested).
        constexpr int     dotRad = 4;
        constexpr uint8_t dotColorR = lineR, dotColorG = lineG, dotColorB = lineB;
        constexpr float   dotAlpha  = 0.5f;
        for (const auto& v : positions) {
            float px, py, pz;
            if (!project(v, px, py, pz)) continue;
            int cx = static_cast<int>(px + 0.5f), cy = static_cast<int>(py + 0.5f);
            for (int dyy = -dotRad; dyy <= dotRad; ++dyy)
                for (int dxx = -dotRad; dxx <= dotRad; ++dxx)
                    if (dxx*dxx + dyy*dyy <= dotRad*dotRad)
                        blendPixelZ(cx + dxx, cy + dyy, pz, dotColorR, dotColorG, dotColorB, dotAlpha);
        }

        writePNG(outputPath, pixels);
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

    /// Core render: launch OptiX, read back framebuffer → 8-bit RGBA pixels + float depth buffer.
    struct RenderResult {
        std::vector<uint8_t> pixels;  // W*H*4  RGBA
        std::vector<float>   depth;   // W*H    ray-t (hit distance from eye)
    };

    RenderResult renderToPixelsWithDepth(const Camera& cam) const {
        const uint32_t W = kRenderWidth;
        const uint32_t H = kRenderHeight;

        CUdeviceptr d_fb = 0;
        RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_fb),
                                     W * H * sizeof(float4)));

        float3 fwd = make_float3(cam.lookAt.x - cam.eye.x,
                                 cam.lookAt.y - cam.eye.y,
                                 cam.lookAt.z - cam.eye.z);
        { float l = sqrtf(fwd.x*fwd.x + fwd.y*fwd.y + fwd.z*fwd.z);
          fwd.x /= l; fwd.y /= l; fwd.z /= l; }

        float3 right = make_float3(fwd.y*cam.up.z - fwd.z*cam.up.y,
                                   fwd.z*cam.up.x - fwd.x*cam.up.z,
                                   fwd.x*cam.up.y - fwd.y*cam.up.x);
        { float l = sqrtf(right.x*right.x + right.y*right.y + right.z*right.z);
          right.x /= l; right.y /= l; right.z /= l; }

        float3 camUp = make_float3(right.y*fwd.z - right.z*fwd.y,
                                   right.z*fwd.x - right.x*fwd.z,
                                   right.x*fwd.y - right.y*fwd.x);

        const float fovRad = cam.fovY * 3.14159265358979323846f / 180.0f;
        const float halfH  = tanf(fovRad * 0.5f);
        const float aspect = static_cast<float>(W) / static_cast<float>(H);
        const float halfW  = halfH * aspect;

        RenderParams rp = {};
        rp.framebuffer  = reinterpret_cast<float4*>(d_fb);
        rp.width        = W;
        rp.height       = H;
        rp.eye          = cam.eye;
        rp.U            = make_float3(right.x*halfW, right.y*halfW, right.z*halfW);
        rp.V            = make_float3(-camUp.x*halfH, -camUp.y*halfH, -camUp.z*halfH);
        rp.W            = fwd;
        rp.vertexColors = reinterpret_cast<float3*>(d_colors_);
        rp.indices      = reinterpret_cast<uint3*>(d_shaderIndices_);
        rp.gasHandle    = gasHandle_;

        CUdeviceptr d_params = 0;
        RENDER_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(RenderParams)));
        RENDER_CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &rp,
                                     sizeof(RenderParams), cudaMemcpyHostToDevice));

        RENDER_OPTIX_CHECK(optixLaunch(pipeline_, stream_,
                                       d_params, sizeof(RenderParams),
                                       &sbt_, W, H, 1));
        RENDER_CUDA_CHECK(cudaStreamSynchronize(stream_));

        std::vector<float4> fb(W * H);
        RENDER_CUDA_CHECK(cudaMemcpy(fb.data(), reinterpret_cast<void*>(d_fb),
                                     W * H * sizeof(float4), cudaMemcpyDeviceToHost));
        RENDER_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_fb)));
        RENDER_CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));

        RenderResult result;
        result.pixels.resize(W * H * 4);
        result.depth.resize(W * H);
        for (uint32_t i = 0; i < W * H; ++i) {
            auto clamp01 = [](float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
            result.pixels[i*4+0] = static_cast<uint8_t>(clamp01(fb[i].x) * 255.0f + 0.5f);
            result.pixels[i*4+1] = static_cast<uint8_t>(clamp01(fb[i].y) * 255.0f + 0.5f);
            result.pixels[i*4+2] = static_cast<uint8_t>(clamp01(fb[i].z) * 255.0f + 0.5f);
            result.pixels[i*4+3] = 255;
            result.depth[i] = fb[i].w;
        }
        return result;
    }

    /// Convenience: render and return only pixels (ignores depth).
    std::vector<uint8_t> renderToPixels(const Camera& cam) const {
        return renderToPixelsWithDepth(cam).pixels;
    }

    // -----------------------------------------------------------------------
    // Minimal 5×7 bitmap font for text watermarks
    // -----------------------------------------------------------------------

    static const uint8_t* glyphData(char c) {
        // Each glyph is 5 columns × 7 rows, packed as 7 bytes (one per row,
        // MSB = leftmost column).  Only printable ASCII 32..126 is defined.
        static const uint8_t font[][7] = {
            // ' ' (32)
            {0x00,0x00,0x00,0x00,0x00,0x00,0x00},
            // '!' (33)
            {0x20,0x20,0x20,0x20,0x20,0x00,0x20},
            // '"' (34)
            {0x50,0x50,0x00,0x00,0x00,0x00,0x00},
            // '#' (35)
            {0x50,0xF8,0x50,0x50,0x50,0xF8,0x50},
            // '$' (36)
            {0x20,0x78,0xA0,0x70,0x28,0xF0,0x20},
            // '%' (37)
            {0xC8,0xD0,0x10,0x20,0x40,0x58,0x98},
            // '&' (38)
            {0x40,0xA0,0xA0,0x40,0xA8,0x90,0x68},
            // '\'' (39)
            {0x20,0x20,0x00,0x00,0x00,0x00,0x00},
            // '(' (40)
            {0x10,0x20,0x40,0x40,0x40,0x20,0x10},
            // ')' (41)
            {0x40,0x20,0x10,0x10,0x10,0x20,0x40},
            // '*' (42)
            {0x00,0x20,0xA8,0x70,0xA8,0x20,0x00},
            // '+' (43)
            {0x00,0x20,0x20,0xF8,0x20,0x20,0x00},
            // ',' (44)
            {0x00,0x00,0x00,0x00,0x00,0x20,0x40},
            // '-' (45)
            {0x00,0x00,0x00,0xF8,0x00,0x00,0x00},
            // '.' (46)
            {0x00,0x00,0x00,0x00,0x00,0x00,0x20},
            // '/' (47)
            {0x08,0x08,0x10,0x20,0x40,0x80,0x80},
            // '0' (48)
            {0x70,0x88,0x98,0xA8,0xC8,0x88,0x70},
            // '1' (49)
            {0x20,0x60,0x20,0x20,0x20,0x20,0x70},
            // '2' (50)
            {0x70,0x88,0x08,0x10,0x20,0x40,0xF8},
            // '3' (51)
            {0x70,0x88,0x08,0x30,0x08,0x88,0x70},
            // '4' (52)
            {0x10,0x30,0x50,0x90,0xF8,0x10,0x10},
            // '5' (53)
            {0xF8,0x80,0xF0,0x08,0x08,0x88,0x70},
            // '6' (54)
            {0x30,0x40,0x80,0xF0,0x88,0x88,0x70},
            // '7' (55)
            {0xF8,0x08,0x10,0x20,0x40,0x40,0x40},
            // '8' (56)
            {0x70,0x88,0x88,0x70,0x88,0x88,0x70},
            // '9' (57)
            {0x70,0x88,0x88,0x78,0x08,0x10,0x60},
            // ':' (58)
            {0x00,0x00,0x20,0x00,0x00,0x20,0x00},
            // ';' (59)
            {0x00,0x00,0x20,0x00,0x00,0x20,0x40},
            // '<' (60)
            {0x08,0x10,0x20,0x40,0x20,0x10,0x08},
            // '=' (61)
            {0x00,0x00,0xF8,0x00,0xF8,0x00,0x00},
            // '>' (62)
            {0x80,0x40,0x20,0x10,0x20,0x40,0x80},
            // '?' (63)
            {0x70,0x88,0x08,0x10,0x20,0x00,0x20},
            // '@' (64)
            {0x70,0x88,0xB8,0xA8,0xB8,0x80,0x70},
            // 'A' (65)
            {0x70,0x88,0x88,0xF8,0x88,0x88,0x88},
            // 'B' (66)
            {0xF0,0x88,0x88,0xF0,0x88,0x88,0xF0},
            // 'C' (67)
            {0x70,0x88,0x80,0x80,0x80,0x88,0x70},
            // 'D' (68)
            {0xF0,0x88,0x88,0x88,0x88,0x88,0xF0},
            // 'E' (69)
            {0xF8,0x80,0x80,0xF0,0x80,0x80,0xF8},
            // 'F' (70)
            {0xF8,0x80,0x80,0xF0,0x80,0x80,0x80},
            // 'G' (71)
            {0x70,0x88,0x80,0xB8,0x88,0x88,0x70},
            // 'H' (72)
            {0x88,0x88,0x88,0xF8,0x88,0x88,0x88},
            // 'I' (73)
            {0x70,0x20,0x20,0x20,0x20,0x20,0x70},
            // 'J' (74)
            {0x38,0x10,0x10,0x10,0x10,0x90,0x60},
            // 'K' (75)
            {0x88,0x90,0xA0,0xC0,0xA0,0x90,0x88},
            // 'L' (76)
            {0x80,0x80,0x80,0x80,0x80,0x80,0xF8},
            // 'M' (77)
            {0x88,0xD8,0xA8,0x88,0x88,0x88,0x88},
            // 'N' (78)
            {0x88,0xC8,0xA8,0x98,0x88,0x88,0x88},
            // 'O' (79)
            {0x70,0x88,0x88,0x88,0x88,0x88,0x70},
            // 'P' (80)
            {0xF0,0x88,0x88,0xF0,0x80,0x80,0x80},
            // 'Q' (81)
            {0x70,0x88,0x88,0x88,0xA8,0x90,0x68},
            // 'R' (82)
            {0xF0,0x88,0x88,0xF0,0xA0,0x90,0x88},
            // 'S' (83)
            {0x70,0x88,0x80,0x70,0x08,0x88,0x70},
            // 'T' (84)
            {0xF8,0x20,0x20,0x20,0x20,0x20,0x20},
            // 'U' (85)
            {0x88,0x88,0x88,0x88,0x88,0x88,0x70},
            // 'V' (86)
            {0x88,0x88,0x88,0x88,0x50,0x50,0x20},
            // 'W' (87)
            {0x88,0x88,0x88,0x88,0xA8,0xD8,0x88},
            // 'X' (88)
            {0x88,0x88,0x50,0x20,0x50,0x88,0x88},
            // 'Y' (89)
            {0x88,0x88,0x50,0x20,0x20,0x20,0x20},
            // 'Z' (90)
            {0xF8,0x08,0x10,0x20,0x40,0x80,0xF8},
            // '[' (91)
            {0x70,0x40,0x40,0x40,0x40,0x40,0x70},
            // '\\' (92)
            {0x80,0x80,0x40,0x20,0x10,0x08,0x08},
            // ']' (93)
            {0x70,0x10,0x10,0x10,0x10,0x10,0x70},
            // '^' (94)
            {0x20,0x50,0x88,0x00,0x00,0x00,0x00},
            // '_' (95)
            {0x00,0x00,0x00,0x00,0x00,0x00,0xF8},
            // '`' (96)
            {0x40,0x20,0x00,0x00,0x00,0x00,0x00},
            // 'a' (97)
            {0x00,0x00,0x70,0x08,0x78,0x88,0x78},
            // 'b' (98)
            {0x80,0x80,0xF0,0x88,0x88,0x88,0xF0},
            // 'c' (99)
            {0x00,0x00,0x70,0x80,0x80,0x80,0x70},
            // 'd' (100)
            {0x08,0x08,0x78,0x88,0x88,0x88,0x78},
            // 'e' (101)
            {0x00,0x00,0x70,0x88,0xF8,0x80,0x70},
            // 'f' (102)
            {0x30,0x48,0x40,0xE0,0x40,0x40,0x40},
            // 'g' (103)
            {0x00,0x00,0x78,0x88,0x78,0x08,0x70},
            // 'h' (104)
            {0x80,0x80,0xB0,0xC8,0x88,0x88,0x88},
            // 'i' (105)
            {0x20,0x00,0x60,0x20,0x20,0x20,0x70},
            // 'j' (106)
            {0x10,0x00,0x30,0x10,0x10,0x90,0x60},
            // 'k' (107)
            {0x80,0x80,0x90,0xA0,0xC0,0xA0,0x90},
            // 'l' (108)
            {0x60,0x20,0x20,0x20,0x20,0x20,0x70},
            // 'm' (109)
            {0x00,0x00,0xD0,0xA8,0xA8,0x88,0x88},
            // 'n' (110)
            {0x00,0x00,0xB0,0xC8,0x88,0x88,0x88},
            // 'o' (111)
            {0x00,0x00,0x70,0x88,0x88,0x88,0x70},
            // 'p' (112)
            {0x00,0x00,0xF0,0x88,0xF0,0x80,0x80},
            // 'q' (113)
            {0x00,0x00,0x78,0x88,0x78,0x08,0x08},
            // 'r' (114)
            {0x00,0x00,0xB0,0xC8,0x80,0x80,0x80},
            // 's' (115)
            {0x00,0x00,0x78,0x80,0x70,0x08,0xF0},
            // 't' (116)
            {0x40,0x40,0xE0,0x40,0x40,0x48,0x30},
            // 'u' (117)
            {0x00,0x00,0x88,0x88,0x88,0x98,0x68},
            // 'v' (118)
            {0x00,0x00,0x88,0x88,0x88,0x50,0x20},
            // 'w' (119)
            {0x00,0x00,0x88,0x88,0xA8,0xA8,0x50},
            // 'x' (120)
            {0x00,0x00,0x88,0x50,0x20,0x50,0x88},
            // 'y' (121)
            {0x00,0x00,0x88,0x88,0x78,0x08,0x70},
            // 'z' (122)
            {0x00,0x00,0xF8,0x10,0x20,0x40,0xF8},
            // '{' (123)
            {0x10,0x20,0x20,0x40,0x20,0x20,0x10},
            // '|' (124)
            {0x20,0x20,0x20,0x20,0x20,0x20,0x20},
            // '}' (125)
            {0x40,0x20,0x20,0x10,0x20,0x20,0x40},
            // '~' (126)
            {0x00,0x00,0x40,0xA8,0x10,0x00,0x00},
        };
        int idx = static_cast<int>(c) - 32;
        if (idx < 0 || idx > 94) idx = 0;  // fallback to space
        return font[idx];
    }

    /// Stamp a text string into an RGBA pixel buffer at (startX, startY).
    /// Uses a 5×7 bitmap font scaled by `scale`.  Text colour with alpha.
    static void stampText(std::vector<uint8_t>& pixels, uint32_t imgW, uint32_t imgH,
                          const std::string& text, int startX, int startY,
                          uint8_t r, uint8_t g, uint8_t b, float alpha,
                          int scale = 1) {
        constexpr int GW = 5, GH = 7, SPACING = 1;
        int cx = startX;
        for (char c : text) {
            const uint8_t* glyph = glyphData(c);
            for (int gy = 0; gy < GH; ++gy) {
                uint8_t row = glyph[gy];
                for (int gx = 0; gx < GW; ++gx) {
                    if (row & (0x80 >> gx)) {
                        // Fill a scale×scale block.
                        for (int sy = 0; sy < scale; ++sy) {
                            for (int sx = 0; sx < scale; ++sx) {
                                int px = cx + gx * scale + sx;
                                int py = startY + gy * scale + sy;
                                if (px < 0 || px >= (int)imgW || py < 0 || py >= (int)imgH) continue;
                                uint32_t pi = (static_cast<uint32_t>(py) * imgW + static_cast<uint32_t>(px)) * 4;
                                float ia = 1.0f - alpha;
                                pixels[pi + 0] = static_cast<uint8_t>(pixels[pi + 0] * ia + r * alpha + 0.5f);
                                pixels[pi + 1] = static_cast<uint8_t>(pixels[pi + 1] * ia + g * alpha + 0.5f);
                                pixels[pi + 2] = static_cast<uint8_t>(pixels[pi + 2] * ia + b * alpha + 0.5f);
                            }
                        }
                    }
                }
            }
            cx += (GW + SPACING) * scale;
        }
    }

    /// Overlay standard watermark texts onto pixels.
    static void stampWatermarks(std::vector<uint8_t>& pixels, uint32_t W, uint32_t H) {
        // Scale: 2× the base size (base = W/960, 2× = W/480).
        const int scale = std::max(1, static_cast<int>(W) / 480);
        constexpr int GW = 5, GH = 7, SPACING = 1;

        const std::string leftText  = "https://github.com/nicolasgfx/cornellbox.git";
        const std::string rightText = "drnicolasmenzel@gmail.com";

        int charW = (GW + SPACING) * scale;
        int textH = GH * scale;
        int margin = 20 * scale;

        int leftW  = static_cast<int>(leftText.size()) * charW;
        int rightW = static_cast<int>(rightText.size()) * charW;
        int y = static_cast<int>(H) - textH - margin;

        // Subtle white text with moderate alpha — good contrast on dark
        // renders, unobtrusive on bright areas.
        constexpr uint8_t tr = 200, tg = 200, tb = 200;
        constexpr float   ta = 0.55f;

        stampText(pixels, W, H, leftText,  margin, y, tr, tg, tb, ta, scale);
        stampText(pixels, W, H, rightText, static_cast<int>(W) - rightW - margin, y, tr, tg, tb, ta, scale);
    }

    /// Write an RGBA pixel buffer to a PNG file (with watermarks).
    void writePNG(const std::string& path, std::vector<uint8_t>& pixels) const {
        const uint32_t W = kRenderWidth, H = kRenderHeight;
        stampWatermarks(pixels, W, H);
        stbi_flip_vertically_on_write(0);
        int ok = stbi_write_png(path.c_str(),
                                static_cast<int>(W), static_cast<int>(H),
                                4, pixels.data(), static_cast<int>(W * 4));
        if (ok) {
            std::cout << "Rendered PNG: " << path << " (" << W << "x" << H << ")\n";
        } else {
            std::cerr << "Failed to write PNG: " << path << "\n";
        }
    }

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
        pcopts_.numPayloadValues        = 4;   // r, g, b, depth
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
