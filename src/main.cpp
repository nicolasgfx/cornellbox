#define NOMINMAX
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "app/Config.h"
#include "scene/CornellBox.h"
#include "mesh/Subdivision.h"
#include "mesh/PatchBuilder.h"
#include "export/OBJExporter.h"
#ifdef USE_OPTIX
#include "gpu/OptiXContext.h"
#include "gpu/Renderer.h"
#endif

#include <iostream>
#include <filesystem>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <unordered_map>
#include <chrono>
#include <iomanip>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr float kSenderHalfspaceEps = 1e-6f;
// Convergence threshold on maximum unshot radiosity (W/m²).
// Resolution-independent, unlike flux (B*A) which scales with patch area.
constexpr float kConvergence    = 1e-3f;
constexpr uint32_t kMaxIterations = 5000000;

#ifdef USE_OPTIX
std::filesystem::path findPtx(const char* name) {
    for (const std::string& prefix : { "build/", "", "../build/", "../" }) {
        std::filesystem::path p = prefix + name;
        if (std::filesystem::exists(p)) return p;
    }
    return {};
}
#endif

// Forward declaration (defined below).
void computeFormFactorRowCPU(const Mesh& mesh, uint32_t sourceId, std::vector<float>& row);

// ---------------------------------------------------------------------------
// GPU / CPU form-factor computation
// ---------------------------------------------------------------------------

class FormFactorComputer {
public:
    void initialize(const Mesh& mesh) {
        enabled_ = false;
        initialized_ = true;

        if (!kEnableGPUFormFactors) {
            std::cout << "Form factors: GPU disabled\n";
            return;
        }
#ifdef USE_OPTIX
        auto ptx = findPtx("hemisphere_kernels.ptx");
        if (ptx.empty()) { std::cout << "Form factors: PTX not found, CPU fallback\n"; return; }

        ctx_.createModule(ptx.string());
        ctx_.createProgramGroups("__raygen__formfactor_row",
                                 "__miss__hemisphere",
                                 "__closesthit__hemisphere",
                                 "__anyhit__hemisphere");
        ctx_.createPipeline();
        ctx_.createSBT();
        ctx_.buildGAS(mesh);
        enabled_ = true;
        std::cout << "Form factors: GPU (" << kVisibilitySamples << " samples/target)\n";
#else
        std::cout << "Form factors: CPU fallback (USE_OPTIX off)\n";
#endif
    }

    // Compute combined form-factor x visibility row for sourceId.
    // GPU: Monte Carlo area-to-area with ray-traced visibility.
    // CPU fallback: centroid-to-centroid geometric form factors (no visibility).
    void computeRow(const Mesh& mesh, uint32_t sourceId, std::vector<float>& row) {
        if (!initialized_) initialize(mesh);
#ifdef USE_OPTIX
        if (enabled_) {
            uint32_t N = static_cast<uint32_t>(mesh.numTriangles());
            row.assign(N, 0.0f);
            ctx_.computeFormFactorRow(mesh, sourceId, row, 1, kVisibilitySamples);
            if (sourceId < row.size()) row[sourceId] = 0.0f;
            for (float& v : row) if (v < 0.0f) v = 0.0f;
            // Energy conservation: row sum must be ≤ 1 for closed geometry.
            float sum = 0.0f;
            for (float v : row) sum += v;
            if (sum > 1.0f) { float inv = 1.0f / sum; for (float& v : row) v *= inv; }
            return;
        }
#endif
        computeFormFactorRowCPU(mesh, sourceId, row);
    }

private:
    bool initialized_ = false;
    bool enabled_ = false;
#ifdef USE_OPTIX
    OptiXContext::Context ctx_;
#endif
};

// ---------------------------------------------------------------------------
// Light classification after subdivision
// ---------------------------------------------------------------------------

void classifyLightTriangles(Mesh& mesh, uint32_t& lightCount) {
    lightCount = 0;
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
}

// Duplicate vertices of light triangles so they're not shared with
// adjacent ceiling patches (prevents color bleeding at light boundary).
void duplicateLightVertices(Mesh& mesh) {
    for (size_t i = 0; i < mesh.numTriangles(); ++i) {
        if (mesh.triangle_material_id[i] != CornellBox::MAT_LIGHT) continue;
        TriIdx& tri = mesh.indices[i];
        uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
        mesh.vertices.push_back(mesh.vertices[tri.i0]);
        mesh.vertices.push_back(mesh.vertices[tri.i1]);
        mesh.vertices.push_back(mesh.vertices[tri.i2]);
        tri.i0 = base;
        tri.i1 = base + 1;
        tri.i2 = base + 2;
    }
}

// ---------------------------------------------------------------------------
// CPU form-factor computation
// ---------------------------------------------------------------------------

// Centroid-to-centroid geometric form factor from source to all receivers.
//   F_ij ≈ cos θ_i · cos θ_j · A_j / (π r²)
// Row is clamped so Σ F_ij ≤ 1 (energy conservation for closed geometry).
// Visibility is handled separately by the OptiX GPU kernel.
void computeFormFactorRowCPU(const Mesh& mesh,
                             uint32_t sourceId,
                             std::vector<float>& row) {
    const uint32_t N = static_cast<uint32_t>(mesh.numTriangles());
    row.assign(N, 0.0f);
    const Vec3& ni = mesh.triangle_normal[sourceId];

    for (uint32_t j = 0; j < N; ++j) {
        if (j == sourceId) continue;
        Vec3 r = mesh.triangle_centroid[j] - mesh.triangle_centroid[sourceId];
        float dist2 = r.dot(r);
        if (dist2 < 1e-10f) continue;
        if (ni.dot(r) <= kSenderHalfspaceEps) continue;   // behind sender

        float dist = std::sqrt(dist2);
        Vec3 rn = r / dist;
        float cosI = ni.dot(rn);
        float cosJ = mesh.triangle_normal[j].dot(rn * (-1.0f));
        if (cosI <= 0.0f || cosJ <= 0.0f) continue;

        float Aj = std::max(mesh.triangle_area[j], 1e-12f);
        // Clamp dist² to a fraction of receiver area for nearly-coincident centroids.
        float clampedDist2 = std::max(dist2, Aj * 0.1f);
        // F_ij = cos_i * cos_j * A_j / (π (r² + k))
        float ff = (cosI * cosJ * Aj) / (float(M_PI) * (clampedDist2 + kDistanceSoftening));
        if (ff > 1.0f) ff = 1.0f;
        if (ff < 1e-8f) continue;
        row[j] = ff;
    }

    // Energy conservation clamp.
    float sum = 0.0f;
    for (float v : row) sum += v;
    if (sum > 1.0f) { float inv = 1.0f / sum; for (float& v : row) v *= inv; }
}

// ---------------------------------------------------------------------------
// Progressive radiosity solver (shooting method)
// ---------------------------------------------------------------------------

// Each iteration selects the patch with the largest unshot flux (ΔB · A),
// distributes its energy to all visible receivers:
//   reflected_j = ΔB_shooter · F_ij · V_ij · ρ_j
// and accumulates into both total radiosity and unshot radiosity.
// Converges when max unshot radiosity drops below kConvergence.
uint32_t runProgressiveRefinement(const Mesh& mesh,
                                  FormFactorComputer& ffc,
                                  std::vector<Vec3>& radiosity,
                                  std::vector<Vec3>& unshot) {
    const uint32_t N = static_cast<uint32_t>(mesh.numTriangles());
    std::vector<float> ffRow;
    uint32_t iters = 0;

    // Measure initial energy for progress estimation.
    float initialMaxRad = 0.0f;
    for (uint32_t i = 0; i < N; ++i) {
        float r = unshot[i].length();
        if (r > initialMaxRad) initialMaxRad = r;
    }
    // Progress is estimated in log-space: maxRad decays roughly exponentially
    // from initialMaxRad down to kConvergence.
    const float logStart = std::log(std::max(initialMaxRad, kConvergence));
    const float logEnd   = std::log(kConvergence);
    const float logRange = logStart - logEnd;

    auto tStart = std::chrono::high_resolution_clock::now();
    auto tLastPrint = tStart;

    std::cout << "Solving (convergence target: " << kConvergence << ")...\n" << std::flush;

    for (uint32_t iter = 0; iter < kMaxIterations; ++iter) {
        // Select shooter: patch with max unshot flux (ΔB · A).
        float maxFlux = 0.0f, maxRad = 0.0f;
        uint32_t shooter = 0;
        for (uint32_t i = 0; i < N; ++i) {
            float rad  = unshot[i].length();
            float flux = rad * mesh.triangle_area[i];
            if (flux > maxFlux) { maxFlux = flux; shooter = i; }
            if (rad > maxRad) maxRad = rad;
        }
        if (maxFlux < 1e-12f || maxRad < kConvergence) break;

        // User-friendly progress: print at most every 2 seconds.
        auto tNow = std::chrono::high_resolution_clock::now();
        double sinceLastPrint = std::chrono::duration<double>(tNow - tLastPrint).count();
        if (sinceLastPrint >= 2.0 || iter == 0) {
            tLastPrint = tNow;
            double elapsed = std::chrono::duration<double>(tNow - tStart).count();
            float pct = 0.0f;
            if (logRange > 0.0f)
                pct = (logStart - std::log(std::max(maxRad, kConvergence))) / logRange * 100.0f;
            pct = std::min(pct, 99.9f);

            // Build progress bar:  [========>         ]  42%  12.3s  ETA ~17s
            const int barWidth = 30;
            int filled = static_cast<int>(pct / 100.0f * barWidth);
            std::ostringstream bar;
            bar << "\r  [";
            for (int b = 0; b < barWidth; ++b)
                bar << (b < filled ? '=' : (b == filled ? '>' : ' '));
            bar << "]  " << std::fixed << std::setprecision(0) << pct << "%";
            bar << "  " << std::fixed << std::setprecision(1) << elapsed << "s";
            if (pct > 1.0f) {
                double eta = elapsed / (pct / 100.0f) * (1.0f - pct / 100.0f);
                bar << "  ETA ~" << std::fixed << std::setprecision(0) << eta << "s";
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
    }
    // Clear progress line.
    std::cout << "\r" << std::string(80, ' ') << "\r" << std::flush;
    return iters;
}

} // anonymous namespace

// ---------------------------------------------------------------------------

static std::string formatDuration(double seconds) {
    std::ostringstream oss;
    if (seconds < 1.0)       oss << std::fixed << std::setprecision(1) << seconds * 1000.0 << " ms";
    else if (seconds < 60.0) oss << std::fixed << std::setprecision(2) << seconds << " s";
    else                     oss << std::fixed << std::setprecision(1) << seconds / 60.0 << " min";
    return oss.str();
}

int main(int argc, char** argv) {
    auto tTotal = std::chrono::high_resolution_clock::now();

    Config config;
    if (!config.parseArgs(argc, argv)) return 0;

    // --- Hardware info ---
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

    // --- Config summary ---
    std::cout << "\n=== Config ===\n";
    std::cout << "  subdivTarget  = " << kSubdivisionTargetArea << "\n";
    std::cout << "  visSamples    = " << kVisibilitySamples << "\n";
    std::cout << "  indirectBoost = " << kIndirectBoostFactor << "\n";
    std::cout << "  exposure      = " << kToneMapExposure << "\n";
    std::cout << "  gamma         = " << kToneMapGamma << "\n";
    std::cout << "  render        = " << kRenderWidth << "x" << kRenderHeight << "\n\n";

    // 1) Build Cornell Box and fix winding order.
    Mesh mesh = CornellBox::createCornellBox();
    CornellBox::fixNormalsOrientation(mesh);

    // 2) Uniform-area subdivision.
    auto tPhase = std::chrono::high_resolution_clock::now();
    mesh = Subdivision::subdivideToUniformArea(mesh, kSubdivisionTargetArea);
    double dtSub = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - tPhase).count();
    std::cout << "Subdivision: " << mesh.numTriangles() << " tris  ["
              << formatDuration(dtSub) << "]\n";

    // 3) Classify ceiling-light patches and isolate their vertices.
    uint32_t lightCount = 0;
    classifyLightTriangles(mesh, lightCount);
    duplicateLightVertices(mesh);

    // 4) Compute per-triangle geometry and material data.
    PatchBuilder::buildTriangleData(mesh);
    if (config.validate && !PatchBuilder::validateMesh(mesh)) return 1;

    std::cout << "\n=== Scene ===\n";
    std::cout << "  Triangles: " << mesh.numTriangles() << "\n";
    std::cout << "  Vertices : " << mesh.numVertices() << "\n";
    std::cout << "  Emissive : " << lightCount << "\n\n";

    // 5) Radiosity solve: initialise B = E, ΔB = E (all energy starts unshot).
    const uint32_t N = static_cast<uint32_t>(mesh.numTriangles());
    std::vector<Vec3> radiosity(N), unshot(N);
    for (uint32_t i = 0; i < N; ++i) {
        radiosity[i] = mesh.triangle_emission[i];
        unshot[i]    = mesh.triangle_emission[i];
    }

    FormFactorComputer ffc;
    ffc.initialize(mesh);

    tPhase = std::chrono::high_resolution_clock::now();
    uint32_t iters = runProgressiveRefinement(mesh, ffc, radiosity, unshot);
    double dtSolve = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - tPhase).count();

    std::cout << "\n=== Radiosity solve ===\n";
    std::cout << "  Iterations: " << iters << "\n";
    std::cout << "  Time      : " << formatDuration(dtSolve) << "\n";

    // 6) Tone-map: exposure scaling + gamma correction.
    std::vector<Vec3> display(N);
    for (uint32_t i = 0; i < N; ++i) {
        float r = std::pow(std::max(radiosity[i].x * kToneMapExposure, 0.0f), kToneMapGamma);
        float g = std::pow(std::max(radiosity[i].y * kToneMapExposure, 0.0f), kToneMapGamma);
        float b = std::pow(std::max(radiosity[i].z * kToneMapExposure, 0.0f), kToneMapGamma);
        display[i] = Vec3(r, g, b);
    }

    // 7) Export smoothed OBJ with area-weighted vertex colors.
    std::filesystem::create_directories(config.outputPath);
    const std::string objFile = config.outputPath + "/cornell_radiosity.obj";
    if (!OBJExporter::exportSmoothedOBJ(objFile, mesh, display)) return 1;

    // 8) Ray-traced PNG renders (OptiX).
#ifdef USE_OPTIX
    {
        auto ptx = findPtx("render_kernels.ptx");
        if (ptx.empty()) {
            std::cerr << "Render PTX not found — skipping PNG.\n";
        } else {
            // Helper: build a welded render mesh with area-weighted vertex colours.
            // `keep` is called for each triangle index; return false to exclude it.
            auto buildRenderMesh = [&](auto keep)
                -> std::tuple<std::vector<float3>, std::vector<uint3>, std::vector<float3>>
            {
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
                    if (!keep(i)) continue;
                    const auto& tri = mesh.indices[i];
                    Vec3 vp[3] = { mesh.vertices[tri.i0].toVec3(),
                                   mesh.vertices[tri.i1].toVec3(),
                                   mesh.vertices[tri.i2].toVec3() };
                    float area = MathUtils::triangleArea(vp[0], vp[1], vp[2]);
                    const Vec3& col = display[i];
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
            };

            // --- Front view (default camera, all triangles) ---
            {
                auto [pos, idx, colors] = buildRenderMesh([](size_t) { return true; });
                Renderer::RayTracedRenderer renderer;
                renderer.initialize(ptx.string(), pos, idx, colors);
                renderer.renderAndSave(config.outputPath + "/cornell_render.png");
            }

            // --- Top view (ceiling removed, camera looking straight down) ---
            {
                auto [pos, idx, colors] = buildRenderMesh([&](size_t i) {
                    // Remove ceiling and light-panel triangles (centroid y ≈ 0.5).
                    return mesh.triangle_centroid[i].y < 0.48f;
                });
                Renderer::Camera topCam;
                topCam.eye    = make_float3(0.0f, 1.8f, 0.0f);
                topCam.lookAt = make_float3(0.0f, 0.0f, 0.0f);
                topCam.up     = make_float3(0.0f, 0.0f, -1.0f);
                topCam.fovY   = 39.3f;
                Renderer::RayTracedRenderer renderer;
                renderer.initialize(ptx.string(), pos, idx, colors);
                renderer.renderAndSave(config.outputPath + "/cornell_render_top.png", topCam);
            }
        }
    }
#endif

    double dtTotal = std::chrono::duration<double>(
        std::chrono::high_resolution_clock::now() - tTotal).count();

    std::cout << "\n=== Done (" << formatDuration(dtTotal) << ") ===\n";
    std::cout << "  " << objFile << "\n";
#ifdef USE_OPTIX
    std::cout << "  " << config.outputPath << "/cornell_render.png\n";
    std::cout << "  " << config.outputPath << "/cornell_render_top.png\n";
#endif
    std::cout << std::endl;
    return 0;
}
