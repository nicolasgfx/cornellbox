#pragma once
// Interactive radiosity scene viewer.
// GLFW + legacy OpenGL for navigation; full GPU radiosity pipeline on demand.
//
// Controls:
//   WASD + mouse    FPS camera navigation
//   Shift           Move faster
//   Space/Ctrl      Up / Down
//   R / LMB         Run full radiosity solve → OptiX render → PNG output
//   P               GL screenshot
//   ESC             Release mouse / quit

#define NOMINMAX
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <GLFW/glfw3.h>

#ifndef GL_MULTISAMPLE
#define GL_MULTISAMPLE 0x809D
#endif

#include "../math/Vec3.h"
#include "../math/MathUtils.h"
#include "../mesh/MeshData.h"
#include "../scene/SceneLoader.h"
#include "../scene/CornellBox.h"
#include "../mesh/Subdivision.h"
#include "../mesh/PatchBuilder.h"
#include "../app/Config.h"
#include "../solver/FormFactor.h"
#include "../export/OBJExporter.h"

#ifdef USE_OPTIX
#include "../gpu/OptiXContext.h"
#include "../gpu/Renderer.h"   // also provides stb_image_write with IMPLEMENTATION
#else
#include "stb_image_write.h"   // declarations only; IMPLEMENTATION in viewer_main.cpp
#endif

#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <filesystem>
#include <unordered_map>
#include <tuple>
#include <thread>
#include <mutex>
#include <atomic>
#include <functional>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Viewer {

// ---- Camera ----
struct FPSCamera {
    Vec3  position{0.0f, 0.0f, 0.0f};
    float yaw   = -90.0f;   // degrees
    float pitch = 0.0f;

    float speed       = 2.0f;
    float sensitivity = 0.15f;

    Vec3 front() const {
        float ry = yaw   * (float)M_PI / 180.0f;
        float rp = pitch * (float)M_PI / 180.0f;
        return Vec3(std::cos(rp) * std::cos(ry),
                    std::sin(rp),
                    std::cos(rp) * std::sin(ry)).normalized();
    }
    Vec3 right() const { return front().cross(Vec3(0, 1, 0)).normalized(); }
    Vec3 up()    const { return right().cross(front()).normalized(); }
};

// ---- Global state (GLFW callbacks need file-scope access) ----
static FPSCamera     g_camera;
static bool          g_firstMouse = true;
static double        g_lastMouseX = 0, g_lastMouseY = 0;
static bool          g_keys[512] = {};
static bool          g_triggerSolve = false;
static bool          g_triggerScreenshot = false;
static bool          g_showWireframe = false;
static bool          g_showNewTriangles = false;
static bool          g_toggleWireframe = false;  // consumed per-frame
static bool          g_mouseCaptured = true;
static int           g_winW = kViewerWidth, g_winH = kViewerHeight;
static int           g_modelSwitch = 0;  // 0 = none, 1 = refined, 2 = original

// ---- Background solve state ----
static std::mutex             g_solveMutex;
static std::vector<Vec3>      g_pendingColors;   // written by solver thread
static std::atomic<bool>      g_solveRunning{false};
static std::atomic<bool>      g_solveFinished{false};
static std::atomic<uint32_t>  g_solveIters{0};
static std::thread            g_solveThread;

// ---- GLFW callbacks ----
static void keyCallback(GLFWwindow* w, int key, int, int action, int) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
        if (g_mouseCaptured) {
            g_mouseCaptured = false;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
        } else {
            glfwSetWindowShouldClose(w, GLFW_TRUE);
        }
    }
    if (key >= 0 && key < 512) {
        if (action == GLFW_PRESS)   g_keys[key] = true;
        if (action == GLFW_RELEASE) g_keys[key] = false;
    }
    if (key == GLFW_KEY_R && action == GLFW_PRESS)
        g_triggerSolve = true;
    if (key == GLFW_KEY_P && action == GLFW_PRESS)
        g_triggerScreenshot = true;
    if (key == GLFW_KEY_N && action == GLFW_PRESS)
        g_showNewTriangles = !g_showNewTriangles;
    if (key == GLFW_KEY_B && action == GLFW_PRESS)
        g_toggleWireframe = true;
    if (key == GLFW_KEY_1 && action == GLFW_PRESS)
        g_modelSwitch = 1;  // switch to refined
    if (key == GLFW_KEY_2 && action == GLFW_PRESS)
        g_modelSwitch = 2;  // switch to original
}

static void mouseButtonCallback(GLFWwindow* w, int button, int action, int) {
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        if (!g_mouseCaptured) {
            g_mouseCaptured = true;
            g_firstMouse = true;
            glfwSetInputMode(w, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
        } else {
            g_triggerSolve = true;
        }
    }
}

static void cursorCallback(GLFWwindow*, double xpos, double ypos) {
    if (!g_mouseCaptured) return;
    if (g_firstMouse) {
        g_lastMouseX = xpos; g_lastMouseY = ypos;
        g_firstMouse = false;
        return;
    }
    float dx = static_cast<float>(xpos - g_lastMouseX);
    float dy = static_cast<float>(g_lastMouseY - ypos);
    g_lastMouseX = xpos; g_lastMouseY = ypos;
    g_camera.yaw   += dx * g_camera.sensitivity;
    g_camera.pitch += dy * g_camera.sensitivity;
    g_camera.pitch  = std::max(-89.0f, std::min(89.0f, g_camera.pitch));
}

static void framebufferSizeCallback(GLFWwindow*, int w, int h) {
    g_winW = w; g_winH = h;
    glViewport(0, 0, w, h);
}

// ---- Build per-triangle display colors from material Kd ----
static std::vector<Vec3> buildMaterialColors(const Mesh& mesh) {
    size_t N = mesh.numTriangles();
    std::vector<Vec3> colors(N);
    for (size_t i = 0; i < N; ++i) {
        if (mesh.triangle_emission[i].lengthSq() > 1e-4f)
            colors[i] = Vec3(1.0f, 1.0f, 0.9f);
        else
            colors[i] = mesh.triangle_reflectance[i];
    }
    return colors;
}

// ---- Tone-map radiosity to display colors ----
static std::vector<Vec3> toneMapRadiosity(const std::vector<Vec3>& radiosity) {
    size_t N = radiosity.size();
    std::vector<Vec3> display(N);
    for (size_t i = 0; i < N; ++i) {
        float r = std::pow(std::max(radiosity[i].x * kToneMapExposure, 0.0f), kToneMapGamma);
        float g = std::pow(std::max(radiosity[i].y * kToneMapExposure, 0.0f), kToneMapGamma);
        float b = std::pow(std::max(radiosity[i].z * kToneMapExposure, 0.0f), kToneMapGamma);
        display[i] = Vec3(std::min(r, 1.0f), std::min(g, 1.0f), std::min(b, 1.0f));
    }
    return display;
}

// ---- Render mesh flat-shaded in legacy GL (unlit) ----
static void renderMesh(const Mesh& mesh, const std::vector<Vec3>& colors,
                       bool showNewTriangles = false) {
    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        // If showing new triangles, override subdivided ones with magenta.
        if (showNewTriangles &&
            i < mesh.triangle_is_subdivided.size() &&
            mesh.triangle_is_subdivided[i]) {
            glColor3f(1.0f, 0.0f, 1.0f); // 0xFF00FF
        } else {
            const Vec3& c = colors[i];
            glColor3f(c.x, c.y, c.z);
        }
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();
        glVertex3f(v0.x, v0.y, v0.z);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
    }
    glEnd();
}

// ---- Render wireframe overlay ----
static void renderWireframe(const Mesh& mesh) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glEnable(GL_POLYGON_OFFSET_LINE);
    glPolygonOffset(-1.0f, -1.0f); // pull lines in front of solid fill
    glLineWidth(1.0f);
    glColor3f(0.0f, 0.0f, 0.0f); // black wireframe

    glBegin(GL_TRIANGLES);
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 v0 = mesh.vertices[tri.i0].toVec3();
        Vec3 v1 = mesh.vertices[tri.i1].toVec3();
        Vec3 v2 = mesh.vertices[tri.i2].toVec3();
        glVertex3f(v0.x, v0.y, v0.z);
        glVertex3f(v1.x, v1.y, v1.z);
        glVertex3f(v2.x, v2.y, v2.z);
    }
    glEnd();

    glDisable(GL_POLYGON_OFFSET_LINE);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

// ---- Save GL framebuffer to PNG ----
static void saveScreenshot(const std::string& path, int w, int h) {
    std::vector<uint8_t> pixels(w * h * 3);
    glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
    std::vector<uint8_t> flipped(w * h * 3);
    for (int y = 0; y < h; ++y)
        std::memcpy(&flipped[y * w * 3], &pixels[(h - 1 - y) * w * 3], w * 3);
    std::filesystem::create_directories(
        std::filesystem::path(path).parent_path());
    stbi_write_png(path.c_str(), w, h, 3, flipped.data(), w * 3);
    std::cout << "  GL screenshot saved: " << path << "\n";
}

// ---------------------------------------------------------------------------
// GPU / CPU form-factor computation
// ---------------------------------------------------------------------------
#ifdef USE_OPTIX
static std::filesystem::path findPtx(const char* name) {
    for (const std::string& prefix : {"build/", "", "../build/", "../"})
        if (auto p = std::filesystem::path(prefix + name); std::filesystem::exists(p))
            return p;
    return {};
}
#endif

class FormFactorComputer {
public:
    void initialize(const Mesh& mesh) {
        enabled_ = false;
        initialized_ = true;
        if (!kEnableGPUFormFactors) {
            std::cout << "  Form factors: GPU disabled in config\n";
            return;
        }
#ifdef USE_OPTIX
        auto ptx = findPtx("hemisphere_kernels.ptx");
        if (ptx.empty()) { std::cout << "  Form factors: PTX not found, CPU fallback\n"; return; }
        ctx_.createModule(ptx.string());
        ctx_.createProgramGroups("__raygen__formfactor_row",
                                 "__miss__hemisphere",
                                 "__closesthit__hemisphere",
                                 "__anyhit__hemisphere");
        ctx_.createPipeline();
        ctx_.createSBT();
        ctx_.buildGAS(mesh);
        enabled_ = true;
        std::cout << "  Form factors: GPU (" << kVisibilitySamples << " samples/target)\n";
#else
        std::cout << "  Form factors: CPU only (no OptiX)\n";
#endif
    }

    void computeRow(const Mesh& mesh, uint32_t sourceId, std::vector<float>& row) {
        if (!initialized_) initialize(mesh);
#ifdef USE_OPTIX
        if (enabled_) {
            uint32_t N = static_cast<uint32_t>(mesh.numTriangles());
            row.assign(N, 0.0f);
            ctx_.computeFormFactorRow(mesh, sourceId, row, 1, kVisibilitySamples);
            if (sourceId < row.size()) row[sourceId] = 0.0f;
            for (float& v : row) if (v < 0.0f) v = 0.0f;
            float sum = 0.0f;
            for (float v : row) sum += v;
            if (sum > 1.0f) { float inv = 1.0f / sum; for (float& v : row) v *= inv; }
            return;
        }
#endif
        FormFactor::computeRowCPU(mesh, sourceId, row);
    }
private:
    bool initialized_ = false;
    bool enabled_     = false;
#ifdef USE_OPTIX
    OptiXContext::Context ctx_;
#endif
};

// ---- Radiosity solver (progressive refinement / shooting method) ----
// The optional `onSnapshot` callback is invoked every `snapshotInterval`
// iterations with the current radiosity state.  The solver itself is
// completely decoupled from the viewer — the callback is the only bridge.
using SnapshotCallback = std::function<void(const std::vector<Vec3>& radiosity,
                                            uint32_t iteration)>;

static std::vector<Vec3> solveRadiosity(const Mesh& mesh,
                                        SnapshotCallback onSnapshot = nullptr,
                                        uint32_t snapshotInterval = 1000) {
    const uint32_t N = static_cast<uint32_t>(mesh.numTriangles());
    std::vector<Vec3> radiosity(N), unshot(N);
    for (uint32_t i = 0; i < N; ++i) {
        radiosity[i] = mesh.triangle_emission[i];
        unshot[i]    = mesh.triangle_emission[i];
    }

    constexpr float convergence = 1e-3f;
    constexpr uint32_t maxIters = 500000;
    std::vector<float> ffRow;

    FormFactorComputer ffc;
    ffc.initialize(mesh);

    auto tStart = std::chrono::high_resolution_clock::now();
    auto tLastPrint = tStart;
    uint32_t iters = 0;

    float initialMaxRad = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        float r = unshot[i].length();
        if (r > initialMaxRad) initialMaxRad = r;
    }
    const float logStart = std::log(std::max(initialMaxRad, convergence));
    const float logEnd   = std::log(convergence);
    const float logRange = logStart - logEnd;

    for (uint32_t iter = 0; iter < maxIters; ++iter) {
        float maxFlux = 0.0f, maxRad = 0.0f;
        uint32_t shooter = 0;
        for (uint32_t i = 0; i < N; ++i) {
            float rad  = unshot[i].length();
            float flux = rad * mesh.triangle_area[i];
            if (flux > maxFlux) { maxFlux = flux; shooter = i; }
            if (rad > maxRad) maxRad = rad;
        }
        if (maxFlux < 1e-12f || maxRad < convergence) break;

        auto tNow = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(tNow - tStart).count();
        if (std::chrono::duration<double>(tNow - tLastPrint).count() >= 2.0 || iter == 0) {
            tLastPrint = tNow;
            float pct = 0.0f;
            if (logRange > 0.0f)
                pct = (logStart - std::log(std::max(maxRad, convergence))) / logRange * 100.0f;
            pct = std::min(pct, 99.9f);
            const int barW = 30;
            int filled = static_cast<int>(pct / 100.0f * barW);
            std::ostringstream bar;
            bar << "\r  [";
            for (int b = 0; b < barW; ++b)
                bar << (b < filled ? '=' : (b == filled ? '>' : ' '));
            bar << "]  " << std::fixed << std::setprecision(0) << pct << "%"
                << "  " << std::setprecision(1) << elapsed << "s";
            if (pct > 1.0f) {
                double eta = elapsed / (pct / 100.0f) * (1.0f - pct / 100.0f);
                bar << "  ETA ~" << std::setprecision(0) << eta << "s";
            }
            bar << "   ";
            std::cout << bar.str() << std::flush;
        }

        Vec3 energy = unshot[shooter];
        unshot[shooter] = Vec3(0.0f);
        ffc.computeRow(mesh, shooter, ffRow);

#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int64_t j = 0; j < static_cast<int64_t>(N); ++j) {
            if (static_cast<uint32_t>(j) == shooter) continue;
            float ff = ffRow[j];
            if (ff < 1e-8f) continue;
            Vec3 incoming = energy * ff;
            Vec3 reflected(
                incoming.x * mesh.triangle_reflectance[j].x * kIndirectBoostFactor,
                incoming.y * mesh.triangle_reflectance[j].y * kIndirectBoostFactor,
                incoming.z * mesh.triangle_reflectance[j].z * kIndirectBoostFactor);
            radiosity[j] += reflected;
            unshot[j]    += reflected;
        }
        iters = iter + 1;

        // Publish snapshot for live preview.
        if (onSnapshot && (iters % snapshotInterval == 0)) {
            onSnapshot(radiosity, iters);
        }
    }

    // Final snapshot so the viewer always sees the converged state.
    if (onSnapshot) {
        onSnapshot(radiosity, iters);
    }

    double totalSec = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - tStart).count();
    std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
    std::cout << "  Solve complete: " << iters << " iters, "
              << std::fixed << std::setprecision(1) << totalSec << "s\n";
    return radiosity;
}

// ---------------------------------------------------------------------------
// Build welded render mesh with area-weighted vertex colors.
// Same algorithm as main.cpp — welds vertices sharing position, normal
// direction, and material ID, averaging per-triangle colors by area weight.
// ---------------------------------------------------------------------------
#ifdef USE_OPTIX
static std::tuple<std::vector<float3>, std::vector<uint3>, std::vector<float3>>
buildWeldedRenderMesh(const Mesh& mesh, const std::vector<Vec3>& triColors) {
    const uint32_t N = static_cast<uint32_t>(mesh.numTriangles());

    auto qp = [](float v) -> int64_t  { return static_cast<int64_t>(std::llround(double(v) / 1e-5)); };
    auto qn = [](float v) -> int32_t  { return static_cast<int32_t>(std::llround(double(v) * 10.0)); };

    struct WKey {
        int64_t px, py, pz; int32_t nx, ny, nz; uint32_t mat;
        bool operator==(const WKey& o) const {
            return px==o.px && py==o.py && pz==o.pz &&
                   nx==o.nx && ny==o.ny && nz==o.nz && mat==o.mat;
        }
    };
    struct WHash {
        size_t operator()(const WKey& k) const noexcept {
            size_t h = std::hash<int64_t>{}(k.px);
            h ^= std::hash<int64_t>{}(k.py) + 0x9e3779b9 + (h<<6) + (h>>2);
            h ^= std::hash<int64_t>{}(k.pz) + 0x9e3779b9 + (h<<6) + (h>>2);
            h ^= std::hash<int32_t>{}(k.nx) + 0x9e3779b9 + (h<<6) + (h>>2);
            h ^= std::hash<int32_t>{}(k.ny) + 0x9e3779b9 + (h<<6) + (h>>2);
            h ^= std::hash<int32_t>{}(k.nz) + 0x9e3779b9 + (h<<6) + (h>>2);
            h ^= std::hash<uint32_t>{}(k.mat) + 0x9e3779b9 + (h<<6) + (h>>2);
            return h;
        }
    };

    std::unordered_map<WKey, uint32_t, WHash> weld;
    std::vector<float3> pos;
    std::vector<Vec3>   cSum;
    std::vector<float>  wSum;
    std::vector<uint3>  idx;

    for (size_t i = 0; i < N; ++i) {
        const auto& tri = mesh.indices[i];
        Vec3 vp[3] = { mesh.vertices[tri.i0].toVec3(),
                       mesh.vertices[tri.i1].toVec3(),
                       mesh.vertices[tri.i2].toVec3() };
        float area = MathUtils::triangleArea(vp[0], vp[1], vp[2]);
        const Vec3& col = triColors[i];
        const Vec3& n   = mesh.triangle_normal[i];
        uint32_t mat = (i < mesh.triangle_material_id.size())
                       ? mesh.triangle_material_id[i] : 0u;

        uint32_t ni[3];
        for (int k = 0; k < 3; ++k) {
            WKey key{ qp(vp[k].x), qp(vp[k].y), qp(vp[k].z),
                      qn(n.x), qn(n.y), qn(n.z), mat };
            auto it = weld.find(key);
            if (it == weld.end()) {
                uint32_t id = static_cast<uint32_t>(pos.size());
                weld[key] = id;
                pos.push_back(make_float3(vp[k].x, vp[k].y, vp[k].z));
                cSum.push_back(col * area);
                wSum.push_back(area);
                ni[k] = id;
            } else {
                cSum[it->second] += col * area;
                wSum[it->second] += area;
                ni[k] = it->second;
            }
        }
        idx.push_back(make_uint3(ni[0], ni[1], ni[2]));
    }

    std::vector<float3> colors(pos.size());
    for (size_t i = 0; i < pos.size(); ++i) {
        Vec3 c = (wSum[i] > 1e-8f) ? cSum[i] / wSum[i] : Vec3(0.8f);
        auto cl = [](float v) { return v < 0.f ? 0.f : (v > 1.f ? 1.f : v); };
        colors[i] = make_float3(cl(c.x), cl(c.y), cl(c.z));
    }
    return { std::move(pos), std::move(idx), std::move(colors) };
}
#endif

// ---------------------------------------------------------------------------
// Full radiosity pipeline: solve + OptiX render + OBJ export.
// Called when the user presses R in the viewer.
// Now runs in a background thread with live preview via snapshot callback.
// ---------------------------------------------------------------------------

// Tone-map a raw radiosity vector into display colors.
static std::vector<Vec3> toneMapToDisplay(const std::vector<Vec3>& radiosity) {
    const size_t N = radiosity.size();
    std::vector<Vec3> display(N);
    for (size_t i = 0; i < N; ++i) {
        float r = std::pow(std::max(radiosity[i].x * kToneMapExposure, 0.0f), kToneMapGamma);
        float g = std::pow(std::max(radiosity[i].y * kToneMapExposure, 0.0f), kToneMapGamma);
        float b = std::pow(std::max(radiosity[i].z * kToneMapExposure, 0.0f), kToneMapGamma);
        display[i] = Vec3(r, g, b);
    }
    return display;
}

static void launchBackgroundSolve(
        const Mesh& mesh,
        const FPSCamera& camera,
        const std::string& outputDir,
        const std::string& sceneName) {

    // Snapshot callback: tone-map and publish to shared buffer.
    // Called from the solver thread every N iterations.
    auto onSnapshot = [](const std::vector<Vec3>& radiosity, uint32_t iter) {
        auto display = toneMapToDisplay(radiosity);
        {
            std::lock_guard<std::mutex> lock(g_solveMutex);
            g_pendingColors = std::move(display);
        }
        g_solveIters.store(iter, std::memory_order_relaxed);
    };

    // Capture camera state at launch time for the final render.
    FPSCamera camCopy = camera;
    std::string outDir = outputDir;
    std::string sName = sceneName;

    g_solveRunning.store(true);
    g_solveFinished.store(false);
    g_solveIters.store(0);

    // Take a const reference — the mesh must outlive the thread.
    // (The viewer holds it in `mesh` / `originalMesh` which lives until exit.)
    g_solveThread = std::thread([&mesh, onSnapshot, camCopy, outDir, sName]() {
        std::cout << "\n=== Radiosity solve (background) ===\n";
        auto radiosity = solveRadiosity(mesh, onSnapshot, 1000);

        // Final display colors (converged).
        auto display = toneMapToDisplay(radiosity);
        {
            std::lock_guard<std::mutex> lock(g_solveMutex);
            g_pendingColors = display;  // copy, not move — need display below
        }

        std::filesystem::create_directories(outDir);

        // Export smoothed OBJ.
        std::string objFile = outDir + "/" + sName + "_radiosity.obj";
        if (OBJExporter::exportSmoothedOBJ(objFile, mesh, display))
            std::cout << "  Exported: " << objFile << "\n";

#ifdef USE_OPTIX
        {
            auto ptx = findPtx("render_kernels.ptx");
            if (ptx.empty()) {
                std::cerr << "  render_kernels.ptx not found — skipping PNG render.\n";
            } else {
                Vec3 eye    = camCopy.position;
                Vec3 lookAt = eye + camCopy.front();
                Renderer::Camera cam;
                cam.eye    = make_float3(eye.x, eye.y, eye.z);
                cam.lookAt = make_float3(lookAt.x, lookAt.y, lookAt.z);
                cam.up     = make_float3(0.0f, 1.0f, 0.0f);
                cam.fovY   = 60.0f;

                auto [pos, idx, colors] = buildWeldedRenderMesh(mesh, display);
                std::cout << "  Render mesh: " << pos.size() << " vertices, "
                          << idx.size() << " triangles\n";

                std::string renderFile = outDir + "/" + sName + "_render.png";
                std::string wireFile   = outDir + "/" + sName + "_render_wireframe.png";

                Renderer::RayTracedRenderer renderer;
                renderer.initialize(ptx.string(), pos, idx, colors);
                renderer.renderAndSave(renderFile, cam);
                renderer.renderWireframeAndSave(wireFile, pos, idx, cam);

                std::cout << "  Rendered: " << renderFile << "\n";
                std::cout << "  Rendered: " << wireFile << "\n";
            }
        }
#endif

        g_solveRunning.store(false);
        g_solveFinished.store(true);
        std::cout << "\n=== Solve complete ===\n\n";
    });
    g_solveThread.detach();
}

// ---- Set up projection + view and render one GL frame ----
static void setupAndRenderFrame(const Mesh& mesh,
                                const std::vector<Vec3>& colors,
                                float nearP, float farP) {
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    float aspect = (g_winH > 0) ? (float)g_winW / (float)g_winH : 1.0f;
    float fovRad = 60.0f * (float)M_PI / 180.0f;
    float top    = nearP * std::tan(fovRad * 0.5f);
    float rProj  = top * aspect;

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glFrustum(-rProj, rProj, -top, top, nearP, farP);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    Vec3 eye    = g_camera.position;
    Vec3 center = eye + g_camera.front();
    Vec3 camUp  = g_camera.up();
    Vec3 f = (center - eye).normalized();
    Vec3 s = f.cross(camUp).normalized();
    Vec3 u = s.cross(f);
    float m[16] = {
         s.x,  u.x, -f.x, 0,
         s.y,  u.y, -f.y, 0,
         s.z,  u.z, -f.z, 0,
        -s.dot(eye), -u.dot(eye), f.dot(eye), 1
    };
    glLoadMatrixf(m);

    renderMesh(mesh, colors, g_showNewTriangles);
    if (g_showWireframe)
        renderWireframe(mesh);
}

// ---- Main viewer entry point ----
inline int run(const std::string& scenePath) {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 2);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint(GLFW_SAMPLES, 4);

    GLFWwindow* window = glfwCreateWindow(g_winW, g_winH,
                                           "Radiosity Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glfwSetKeyCallback(window, keyCallback);
    glfwSetMouseButtonCallback(window, mouseButtonCallback);
    glfwSetCursorPosCallback(window, cursorCallback);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // ---- Load scene ----
    std::cout << "\n=== Viewer: Loading scene ===\n";

    Mesh mesh;
    Mesh originalMesh;
    float sceneExtent = 1.0f;
    std::string sceneName = "cornell";

    if (scenePath.empty()) {
        std::cout << "  Using Cornell Box\n";
        mesh = CornellBox::createCornellBox();
        CornellBox::fixNormalsOrientation(mesh);
        mesh = Subdivision::subdivideToUniformArea(mesh, kSubdivisionTargetArea);

        uint32_t lightCount = 0;
        const float epsY = 0.01f, halfLight = 0.15f, ceilingY = 0.5f;
        for (size_t i = 0; i < mesh.numTriangles(); ++i) {
            const TriIdx& tri = mesh.indices[i];
            Vec3 v[3] = { mesh.vertices[tri.i0].toVec3(),
                           mesh.vertices[tri.i1].toVec3(),
                           mesh.vertices[tri.i2].toVec3() };
            bool isLight = true;
            for (int k = 0; k < 3; ++k) {
                if (v[k].y <= ceilingY - epsY ||
                    std::abs(v[k].x) > halfLight ||
                    std::abs(v[k].z) > halfLight) { isLight = false; break; }
            }
            if (isLight) mesh.triangle_material_id[i] = CornellBox::MAT_LIGHT;
            if (mesh.triangle_material_id[i] == CornellBox::MAT_LIGHT) lightCount++;
        }
        PatchBuilder::buildTriangleData(mesh);
        sceneExtent = 1.0f;

        std::cout << "  " << mesh.numTriangles() << " triangles, "
                  << lightCount << " emissive\n";
        g_camera.position = Vec3(0.0f, 0.0f, 1.5f);
        g_camera.yaw   = -90.0f;
        g_camera.pitch = 0.0f;
        g_camera.speed = 0.5f;
    } else {
        SceneLoader::Scene scene = SceneLoader::loadScene(scenePath);
        if (scene.mesh.numTriangles() == 0) {
            glfwTerminate(); return 1;
        }
        mesh = std::move(scene.mesh);
        originalMesh = std::move(scene.preRefinementMesh);

        Vec3 extent = scene.bboxMax - scene.bboxMin;
        sceneExtent = std::max({extent.x, extent.y, extent.z});

        auto p = std::filesystem::path(scenePath).stem().string();
        if (!p.empty()) sceneName = p;

        g_camera.position = scene.cameraEye;
        Vec3 lookDir = (scene.cameraLookAt - scene.cameraEye).normalized();
        g_camera.yaw   = std::atan2(lookDir.z, lookDir.x) * 180.0f / (float)M_PI;
        g_camera.pitch = std::asin(MathUtils::clamp(lookDir.y, -1.0f, 1.0f))
                       * 180.0f / (float)M_PI;
        g_camera.speed = sceneExtent * 0.3f;

        std::cout << "  " << mesh.numTriangles() << " triangles, "
                  << scene.emissiveTriCount << " emissive\n";
        std::cout << "  Scene extent: " << sceneExtent << "\n";
    }

    // ---- Store original (pre-FF-refinement) mesh for toggling ----
    // For OBJ scenes, originalMesh was already set from scene.preRefinementMesh
    // (with geometry + material data pre-built).
    // For Cornell Box, it's the same as mesh (no FF refinement applied).
    if (scenePath.empty()) {
        originalMesh = mesh;
    }
    bool showingOriginal = false;

    // ---- Build initial display: flat unlit material colors ----
    std::vector<Vec3> displayColors = buildMaterialColors(mesh);
    std::vector<Vec3> originalDisplayColors = buildMaterialColors(originalMesh);
    bool showingRadiosity = false;
    std::string outputDir = "output";

    // ---- Hardware info ----
    std::cout << "\n=== Hardware ===\n";
#ifdef _OPENMP
    std::cout << "  CPU threads: " << omp_get_max_threads() << "\n";
#endif
#ifdef USE_OPTIX
    {
        int count = 0;
        cudaGetDeviceCount(&count);
        for (int i = 0; i < count; ++i) {
            cudaDeviceProp p{};
            cudaGetDeviceProperties(&p, i);
            std::cout << "  GPU " << i << ": " << p.name
                      << " (" << (p.totalGlobalMem >> 20) << " MB)\n";
        }
    }
#endif

    std::cout << "\n=== Controls ===\n"
              << "  WASD         Move\n"
              << "  Mouse        Look around\n"
              << "  Shift        Move faster\n"
              << "  Space/Ctrl   Up / Down\n"
              << "  R / LMB      Run radiosity → OptiX render → PNG\n"
              << "  P            GL screenshot\n"              << "  B            Toggle wireframe overlay\n"
              << "  N            Toggle subdivided triangles (magenta)\n"
              << "  1            Show refined mesh\n"
              << "  2            Show original mesh (pre-FF-refinement)\n"              << "  ESC          Release mouse / quit\n\n";

    // ---- OpenGL setup ----
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_MULTISAMPLE);
    glClearColor(0.12f, 0.12f, 0.15f, 1.0f);
    glDisable(GL_LIGHTING);

    float nearP = sceneExtent * 0.001f;
    float farP  = sceneExtent * 20.0f;

    double lastTime = glfwGetTime();

    while (!glfwWindowShouldClose(window)) {
        double now = glfwGetTime();
        float dt = static_cast<float>(now - lastTime);
        lastTime = now;

        glfwPollEvents();

        // ---- Camera movement ----
        if (g_mouseCaptured) {
            float moveSpeed = g_camera.speed * dt;
            if (g_keys[GLFW_KEY_LEFT_SHIFT]) moveSpeed *= 3.0f;
            Vec3 front = g_camera.front();
            Vec3 right = g_camera.right();
            if (g_keys[GLFW_KEY_W]) g_camera.position += front * moveSpeed;
            if (g_keys[GLFW_KEY_S]) g_camera.position -= front * moveSpeed;
            if (g_keys[GLFW_KEY_D]) g_camera.position += right * moveSpeed;
            if (g_keys[GLFW_KEY_A]) g_camera.position -= right * moveSpeed;
            if (g_keys[GLFW_KEY_SPACE])
                g_camera.position += Vec3(0, 1, 0) * moveSpeed;
            if (g_keys[GLFW_KEY_LEFT_CONTROL])
                g_camera.position -= Vec3(0, 1, 0) * moveSpeed;
        }

        // ---- Toggle wireframe ----
        if (g_toggleWireframe) {
            g_toggleWireframe = false;
            g_showWireframe = !g_showWireframe;
        }

        // ---- Model switching (1 = refined, 2 = original) ----
        if (g_modelSwitch != 0) {
            if (g_modelSwitch == 2 && !showingOriginal) {
                showingOriginal = true;
                displayColors = buildMaterialColors(originalMesh);
                showingRadiosity = false;
                std::cout << "Switched to ORIGINAL mesh (" << originalMesh.numTriangles() << " tris)\n";
            } else if (g_modelSwitch == 1 && showingOriginal) {
                showingOriginal = false;
                displayColors = buildMaterialColors(mesh);
                showingRadiosity = false;
                std::cout << "Switched to REFINED mesh (" << mesh.numTriangles() << " tris)\n";
            }
            g_modelSwitch = 0;
        }

        // ---- Trigger full radiosity pipeline (background) ----
        if (g_triggerSolve && !g_solveRunning.load()) {
            g_triggerSolve = false;
            std::cout << "\n=== Camera ===\n";
            std::cout << "  Position: (" << g_camera.position.x << ", "
                      << g_camera.position.y << ", " << g_camera.position.z << ")\n";
            std::cout << "  Yaw: " << g_camera.yaw << "  Pitch: " << g_camera.pitch << "\n";
            Vec3 lookAt = g_camera.position + g_camera.front();
            std::cout << "  LookAt: (" << lookAt.x << ", "
                      << lookAt.y << ", " << lookAt.z << ")\n";

            const Mesh& activeMesh = showingOriginal ? originalMesh : mesh;
            launchBackgroundSolve(activeMesh, g_camera, outputDir, sceneName);
            showingRadiosity = true;
        } else if (g_triggerSolve) {
            g_triggerSolve = false;  // ignore if already running
        }

        // ---- Pick up live radiosity updates from background solver ----
        if (g_solveMutex.try_lock()) {
            if (!g_pendingColors.empty()) {
                displayColors = std::move(g_pendingColors);
                g_pendingColors.clear();
            }
            g_solveMutex.unlock();
        }

        // ---- GL screenshot on P ----
        if (g_triggerScreenshot) {
            g_triggerScreenshot = false;
            const Mesh& screenshotMesh = showingOriginal ? originalMesh : mesh;
            setupAndRenderFrame(screenshotMesh, displayColors, nearP, farP);
            glFinish();
            saveScreenshot(outputDir + "/viewer_screenshot.png", g_winW, g_winH);
        }

        // ---- Render ----
        const Mesh& activeMesh = showingOriginal ? originalMesh : mesh;
        setupAndRenderFrame(activeMesh, displayColors, nearP, farP);

        {
            std::ostringstream title;
            title << "Radiosity Viewer  |  "
                  << activeMesh.numTriangles() << " tris  |  "
                  << (showingOriginal ? "ORIGINAL" : "REFINED") << "  |  "
                  << (g_solveRunning.load() ? "SOLVING" :
                      (showingRadiosity ? "RADIOSITY" : "MATERIALS"));
            if (g_solveRunning.load())
                title << " (iter " << g_solveIters.load(std::memory_order_relaxed) << ")";
            title << "  |  WASD+mouse  R=solve  1/2=model";
            glfwSetWindowTitle(window, title.str().c_str());
        }

        glfwSwapBuffers(window);
    }

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

} // namespace Viewer
