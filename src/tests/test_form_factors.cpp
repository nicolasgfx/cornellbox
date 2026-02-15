// ---------------------------------------------------------------------------
// Form-factor robustness test suite
//
// Tests that the CPU form-factor computation is invariant under mesh
// subdivision — the core property required for correct radiosity.
//
// Uses productive code paths:
//   FormFactor::computeRowCPU   (src/solver/FormFactor.h)
//   Subdivision::subdivideToUniformArea  (src/mesh/Subdivision.h)
//   PatchBuilder::buildTriangleData      (src/mesh/PatchBuilder.h)
//
// Build:   cmake --build --preset release --target test_form_factors
// Run:     build/bin/Release/test_form_factors.exe
// ---------------------------------------------------------------------------

#define NOMINMAX
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "../mesh/MeshData.h"
#include "../mesh/Subdivision.h"
#include "../mesh/PatchBuilder.h"
#include "../math/MathUtils.h"
#include "../solver/FormFactor.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <sstream>
#include <functional>

// ---------------------------------------------------------------------------
// Surface tags (stored in triangle_material_id for identification)
// ---------------------------------------------------------------------------
static constexpr uint32_t TAG_A       = 10;   // source surface
static constexpr uint32_t TAG_B       = 20;   // target surface
static constexpr uint32_t TAG_BLOCKER = 30;   // occluder

// ---------------------------------------------------------------------------
// Mesh construction helpers
// ---------------------------------------------------------------------------

// Add a quad (2 triangles) whose face normal = right × up.
//   center   — centre of the quad
//   halfW/H  — half-extents along right / up axes
static void addQuad(Mesh& mesh, uint32_t matId,
                    const Vec3& center, float halfW, float halfH,
                    const Vec3& right, const Vec3& up)
{
    uint32_t base = static_cast<uint32_t>(mesh.vertices.size());
    mesh.vertices.push_back(Vertex(center - right * halfW - up * halfH));   // BL
    mesh.vertices.push_back(Vertex(center + right * halfW - up * halfH));   // BR
    mesh.vertices.push_back(Vertex(center + right * halfW + up * halfH));   // TR
    mesh.vertices.push_back(Vertex(center - right * halfW + up * halfH));   // TL
    mesh.indices.push_back(TriIdx(base, base + 1, base + 2));
    mesh.indices.push_back(TriIdx(base, base + 2, base + 3));
    mesh.triangle_material_id.push_back(matId);
    mesh.triangle_material_id.push_back(matId);
}

// Extract only the triangles matching `matId` into a new standalone Mesh.
static Mesh extractSurface(const Mesh& full, uint32_t matId) {
    Mesh out;
    for (size_t i = 0; i < full.numTriangles(); ++i) {
        if (full.triangle_material_id[i] != matId) continue;
        uint32_t base = static_cast<uint32_t>(out.vertices.size());
        const TriIdx& tri = full.indices[i];
        out.vertices.push_back(full.vertices[tri.i0]);
        out.vertices.push_back(full.vertices[tri.i1]);
        out.vertices.push_back(full.vertices[tri.i2]);
        out.indices.push_back(TriIdx(base, base + 1, base + 2));
        out.triangle_material_id.push_back(matId);
    }
    return out;
}

// Merge two meshes (vertex indices of `b` are offset).
static Mesh mergeMeshes(const Mesh& a, const Mesh& b) {
    Mesh r;
    r.vertices = a.vertices;
    r.indices  = a.indices;
    r.triangle_material_id = a.triangle_material_id;

    uint32_t off = static_cast<uint32_t>(a.vertices.size());
    for (const auto& v : b.vertices) r.vertices.push_back(v);
    for (const auto& t : b.indices)
        r.indices.push_back(TriIdx(t.i0 + off, t.i1 + off, t.i2 + off));
    for (auto m : b.triangle_material_id) r.triangle_material_id.push_back(m);
    return r;
}

// ---------------------------------------------------------------------------
// Scene factories
// ---------------------------------------------------------------------------

// Two parallel facing squares separated by `distance` along the z-axis.
//   Source at z=0 facing +z,  Target at z=distance facing −z.
static Mesh makeParallelQuads(float distance, float halfSize) {
    Mesh mesh;
    addQuad(mesh, TAG_A, Vec3(0, 0, 0),        halfSize, halfSize,
            Vec3(1, 0, 0), Vec3(0, 1, 0));          // normal = +z
    addQuad(mesh, TAG_B, Vec3(0, 0, distance),  halfSize, halfSize,
            Vec3(-1, 0, 0), Vec3(0, 1, 0));         // normal = −z
    return mesh;
}

// Two perpendicular squares sharing an edge (L-shape):
//   Floor at y = 0 facing +y,  Wall at x = −halfSize facing +x.
static Mesh makePerpendicularQuads(float halfSize) {
    Mesh mesh;
    // Floor: at y=0, spanning x ∈ [−hs, +hs], z ∈ [−hs, +hs]
    addQuad(mesh, TAG_A, Vec3(0, 0, 0), halfSize, halfSize,
            Vec3(1, 0, 0), Vec3(0, 0, -1));         // normal = +y
    // Wall: at x=−halfSize, spanning y ∈ [0, 2*hs], z ∈ [−hs, +hs]
    addQuad(mesh, TAG_B, Vec3(-halfSize, halfSize, 0), halfSize, halfSize,
            Vec3(0, 0, -1), Vec3(0, 1, 0));         // normal = +x
    return mesh;
}

// One large source quad and a much smaller target quad, both parallel.
static Mesh makeAsymmetricQuads(float distance,
                                float srcHalf, float tgtHalf) {
    Mesh mesh;
    addQuad(mesh, TAG_A, Vec3(0, 0, 0),        srcHalf, srcHalf,
            Vec3(1, 0, 0), Vec3(0, 1, 0));
    addQuad(mesh, TAG_B, Vec3(0, 0, distance),  tgtHalf, tgtHalf,
            Vec3(-1, 0, 0), Vec3(0, 1, 0));
    return mesh;
}

// Mini Cornell-box-like enclosure (5 walls, open front):
//   Floor, ceiling, back wall, left wall, right wall.
//   A small TAG_A patch sits inside, representing a box face.
static Mesh makeMiniCornellBox(float halfRoom, float patchHalf) {
    Mesh mesh;

    // Room walls (TAG_B)
    // Floor (y = −halfRoom, facing +y)
    addQuad(mesh, TAG_B, Vec3(0, -halfRoom, 0), halfRoom, halfRoom,
            Vec3(1, 0, 0), Vec3(0, 0, -1));
    // Ceiling (y = +halfRoom, facing −y)
    addQuad(mesh, TAG_B, Vec3(0, halfRoom, 0), halfRoom, halfRoom,
            Vec3(1, 0, 0), Vec3(0, 0, 1));
    // Back wall (z = −halfRoom, facing +z)
    addQuad(mesh, TAG_B, Vec3(0, 0, -halfRoom), halfRoom, halfRoom,
            Vec3(1, 0, 0), Vec3(0, 1, 0));
    // Left wall (x = −halfRoom, facing +x)
    addQuad(mesh, TAG_B, Vec3(-halfRoom, 0, 0), halfRoom, halfRoom,
            Vec3(0, 0, -1), Vec3(0, 1, 0));
    // Right wall (x = +halfRoom, facing −x)
    addQuad(mesh, TAG_B, Vec3(halfRoom, 0, 0), halfRoom, halfRoom,
            Vec3(0, 0, 1), Vec3(0, 1, 0));

    // Small patch inside (TAG_A) — like a box face, facing +z
    addQuad(mesh, TAG_A, Vec3(0.15f, -halfRoom + 0.15f, 0.1f),
            patchHalf, patchHalf, Vec3(1, 0, 0), Vec3(0, 1, 0));

    return mesh;
}

// Two facing quads with an occluder in between.
// (CPU form factors have no visibility, so this documents the limitation.)
static Mesh makeOccludedQuads(float distance, float halfSize) {
    Mesh mesh;
    addQuad(mesh, TAG_A, Vec3(0, 0, 0),              halfSize, halfSize,
            Vec3(1, 0, 0), Vec3(0, 1, 0));
    addQuad(mesh, TAG_B, Vec3(0, 0, distance),       halfSize, halfSize,
            Vec3(-1, 0, 0), Vec3(0, 1, 0));
    // Blocker in between
    addQuad(mesh, TAG_BLOCKER, Vec3(0, 0, distance * 0.5f), halfSize * 1.2f, halfSize * 1.2f,
            Vec3(1, 0, 0), Vec3(0, 1, 0));
    return mesh;
}

// ---------------------------------------------------------------------------
// Refinement study
// ---------------------------------------------------------------------------

struct SubdivResult {
    float targetArea;         // subdivision target area (0 = coarse)
    uint32_t srcTris;
    uint32_t tgtTris;
    float tgtMinArea, tgtMaxArea, tgtAvgArea;
    float ffRaw;              // F_{A→B} without energy-conservation clamp
    float ffClamped;          // F_{A→B} with energy-conservation clamp
    float maxRowSumRaw;       // largest row sum across source tris (raw)
    bool  conservationFired;  // did any source row exceed 1.0?
    // Per-FF diagnostics
    uint32_t numFFClamped;    // how many individual F_ij hit the 1.0 cap
    float maxSingleFF;        // largest single F_ij in the row
};

// Run a subdivision refinement study.
//   baseSrc/baseTgt — coarse meshes for each surface (tagged with matId).
//   subdivLevels    — list of target areas to try (0 = coarse, no subdivision).
//   subdivSource    — if true, also subdivide the source surface.
static std::vector<SubdivResult> runRefinementStudy(
    const Mesh& baseSrc, const Mesh& baseTgt,
    uint32_t srcTag, uint32_t tgtTag,
    const std::vector<float>& subdivLevels,
    bool subdivSource = false)
{
    std::vector<SubdivResult> results;

    for (float level : subdivLevels) {
        Mesh src = baseSrc;
        Mesh tgt = baseTgt;

        if (level > 0.0f) {
            tgt = Subdivision::subdivideToUniformArea(tgt, level);
            if (subdivSource)
                src = Subdivision::subdivideToUniformArea(src, level);
        }

        Mesh combined = mergeMeshes(src, tgt);
        PatchBuilder::buildTriangleData(combined);

        uint32_t N = static_cast<uint32_t>(combined.numTriangles());
        uint32_t srcCount = 0, tgtCount = 0;
        float tgtMinA = 1e30f, tgtMaxA = 0.0f, tgtSumA = 0.0f;
        for (uint32_t i = 0; i < N; ++i) {
            if (combined.triangle_material_id[i] == srcTag) srcCount++;
            if (combined.triangle_material_id[i] == tgtTag) {
                tgtCount++;
                float a = combined.triangle_area[i];
                tgtMinA = std::min(tgtMinA, a);
                tgtMaxA = std::max(tgtMaxA, a);
                tgtSumA += a;
            }
        }

        FormFactor::Options rawOpts;
        rawOpts.clampEnergyConservation = false;  // raw
        FormFactor::Options clampOpts;             // production defaults

        double rawWeighted = 0.0, clampWeighted = 0.0, srcTotalArea = 0.0;
        float maxRowSum = 0.0f;
        bool conservFired = false;
        uint32_t ffClampCount = 0;
        float maxSingleFF = 0.0f;

        std::vector<float> row;
        for (uint32_t i = 0; i < N; ++i) {
            if (combined.triangle_material_id[i] != srcTag) continue;
            float Ai = combined.triangle_area[i];
            srcTotalArea += Ai;

            // --- Raw (unclamped row sum) ---
            FormFactor::computeRowCPU(combined, i, row, rawOpts);
            double ffToTgt = 0.0, rowSum = 0.0;
            for (uint32_t j = 0; j < N; ++j) {
                rowSum += row[j];
                if (combined.triangle_material_id[j] == tgtTag) ffToTgt += row[j];
                if (row[j] >= 0.999f) ffClampCount++;
                if (row[j] > maxSingleFF) maxSingleFF = row[j];
            }
            rawWeighted += Ai * ffToTgt;
            if (static_cast<float>(rowSum) > maxRowSum) maxRowSum = static_cast<float>(rowSum);
            if (rowSum > 1.0) conservFired = true;

            // --- Clamped (production behaviour) ---
            FormFactor::computeRowCPU(combined, i, row, clampOpts);
            ffToTgt = 0.0;
            for (uint32_t j = 0; j < N; ++j)
                if (combined.triangle_material_id[j] == tgtTag) ffToTgt += row[j];
            clampWeighted += Ai * ffToTgt;
        }

        SubdivResult sr{};
        sr.targetArea    = level;
        sr.srcTris       = srcCount;
        sr.tgtTris       = tgtCount;
        sr.tgtMinArea    = tgtMinA;
        sr.tgtMaxArea    = tgtMaxA;
        sr.tgtAvgArea    = tgtCount ? tgtSumA / tgtCount : 0.0f;
        sr.ffRaw         = srcTotalArea > 1e-12 ? float(rawWeighted / srcTotalArea) : 0.0f;
        sr.ffClamped     = srcTotalArea > 1e-12 ? float(clampWeighted / srcTotalArea) : 0.0f;
        sr.maxRowSumRaw  = maxRowSum;
        sr.conservationFired = conservFired;
        sr.numFFClamped  = ffClampCount;
        sr.maxSingleFF   = maxSingleFF;
        results.push_back(sr);
    }
    return results;
}

// ---------------------------------------------------------------------------
// Pretty-printing helpers
// ---------------------------------------------------------------------------

static void printHeader() {
    std::cout << "  " << std::left
              << std::setw(11) << "Tgt Subdiv"
              << std::setw(7)  << "SrcT"
              << std::setw(7)  << "TgtT"
              << std::setw(13) << "Tgt Avg A"
              << std::setw(13) << "F_A>B raw"
              << std::setw(13) << "F_A>B clmp"
              << std::setw(10) << "RowSum"
              << std::setw(8)  << "Consrv"
              << std::setw(8)  << "FF>=1"
              << std::setw(10) << "MaxFF"
              << "\n";
    std::cout << "  " << std::string(100, '-') << "\n";
}

static void printRow(const SubdivResult& r) {
    std::cout << "  " << std::left;
    if (r.targetArea <= 0.0f)
        std::cout << std::setw(11) << "coarse";
    else
        std::cout << std::setw(11) << std::fixed << std::setprecision(6) << r.targetArea;
    std::cout << std::setw(7)  << r.srcTris
              << std::setw(7)  << r.tgtTris
              << std::setw(13) << std::scientific << std::setprecision(3) << r.tgtAvgArea
              << std::setw(13) << std::fixed << std::setprecision(6) << r.ffRaw
              << std::setw(13) << std::fixed << std::setprecision(6) << r.ffClamped
              << std::setw(10) << std::fixed << std::setprecision(4) << r.maxRowSumRaw
              << std::setw(8)  << (r.conservationFired ? "YES" : "no")
              << std::setw(8)  << r.numFFClamped
              << std::setw(10) << std::fixed << std::setprecision(4) << r.maxSingleFF
              << "\n";
}

// ---------------------------------------------------------------------------
// Test runner
// ---------------------------------------------------------------------------

static const std::vector<float> kSubdivLevels = {
    0.0f,           // coarse (no subdivision)
    0.01f,
    0.005f,
    0.002f,
    0.001f,         // ← production default that "works"
    0.0005f,        // ← production value that causes dark box
    0.0002f,
    0.0001f,
};

struct TestResult {
    std::string name;
    bool passed;
    float maxDeviation;       // worst relative deviation from coarse F
};

// Run a subdivision invariance test on a given scene.
//   Returns true if max deviation ≤ tolerance.
static TestResult runSubdivInvarianceTest(
    const std::string& name,
    const std::string& description,
    const Mesh& scene,         // complete scene (src + tgt tagged)
    uint32_t srcTag, uint32_t tgtTag,
    const std::vector<float>& levels,
    float tolerance = 0.15f,   // 15% max allowed deviation
    bool subdivSource = false)
{
    Mesh src = extractSurface(scene, srcTag);
    Mesh tgt = extractSurface(scene, tgtTag);

    auto results = runRefinementStudy(src, tgt, srcTag, tgtTag, levels, subdivSource);

    std::cout << "\n--- " << name << " ---\n";
    std::cout << "  " << description << "\n";

    // Print area of source surface
    {
        Mesh srcReady = src;
        PatchBuilder::buildTriangleData(srcReady);
        float srcArea = 0.0f;
        for (auto a : srcReady.triangle_area) srcArea += a;
        std::cout << "  Source area: " << std::fixed << std::setprecision(6) << srcArea << "\n";
    }
    {
        Mesh tgtReady = tgt;
        PatchBuilder::buildTriangleData(tgtReady);
        float tgtArea = 0.0f;
        for (auto a : tgtReady.triangle_area) tgtArea += a;
        std::cout << "  Target area: " << std::fixed << std::setprecision(6) << tgtArea << "\n";
    }
    std::cout << "\n";

    printHeader();
    for (const auto& r : results) printRow(r);

    // Evaluate: compare all levels to coarse (first entry)
    float refFF = results[0].ffRaw;
    float maxDev = 0.0f;
    std::string worstLevel;
    for (size_t i = 1; i < results.size(); ++i) {
        float dev = (refFF > 1e-10f)
                    ? std::abs(results[i].ffRaw - refFF) / refFF
                    : std::abs(results[i].ffRaw - refFF);
        if (dev > maxDev) {
            maxDev = dev;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(6) << results[i].targetArea;
            worstLevel = oss.str();
        }
    }

    bool passed = maxDev <= tolerance;
    std::cout << "\n  Reference (coarse) F_raw = "
              << std::fixed << std::setprecision(6) << refFF << "\n";
    std::cout << "  Max deviation from coarse: "
              << std::fixed << std::setprecision(1) << (maxDev * 100.0f)
              << "% at subdiv=" << worstLevel << "\n";

    // ---- Convergence analysis: compare adjacent subdivision levels ----
    // This is the REAL robustness test: if F is stable between fine levels
    // (especially 0.001 -> 0.0005), subdivision won't cause darkening.
    // We exclude the coarse->first-refined step because centroid-to-centroid
    // is inherently inaccurate for large coarse triangles.
    float maxAdjacentDev = 0.0f;
    std::string adjacentPair;
    float maxFineAdjacentDev = 0.0f;  // excluding coarse->first step
    std::string fineAdjacentPair;
    for (size_t i = 1; i < results.size(); ++i) {
        float prev = results[i - 1].ffRaw;
        float curr = results[i].ffRaw;
        float ref  = std::max(prev, curr);
        float adj  = (ref > 1e-10f) ? std::abs(curr - prev) / ref : 0.0f;
        std::ostringstream oss;
        if (results[i - 1].targetArea <= 0.0f)
            oss << "coarse";
        else
            oss << std::fixed << std::setprecision(6) << results[i - 1].targetArea;
        oss << " -> " << std::fixed << std::setprecision(6) << results[i].targetArea;
        if (adj > maxAdjacentDev) {
            maxAdjacentDev = adj;
            adjacentPair = oss.str();
        }
        // Track fine-only (both levels with targetArea > 0)
        if (results[i - 1].targetArea > 0.0f && adj > maxFineAdjacentDev) {
            maxFineAdjacentDev = adj;
            fineAdjacentPair = oss.str();
        }
    }
    // Specific 0.001 -> 0.0005 check (the production transition).
    float prodDev = 0.0f;
    for (size_t i = 0; i + 1 < results.size(); ++i) {
        if (std::abs(results[i].targetArea - 0.001f) < 1e-7f &&
            std::abs(results[i + 1].targetArea - 0.0005f) < 1e-7f) {
            float ref = std::max(results[i].ffRaw, results[i + 1].ffRaw);
            if (ref > 1e-10f) prodDev = std::abs(results[i].ffRaw - results[i + 1].ffRaw) / ref;
        }
    }

    std::cout << "  Coarse->first step deviation: "
              << std::fixed << std::setprecision(2) << (maxAdjacentDev * 100.0f)
              << "% (" << adjacentPair << ") -- centroid approx error, expected\n";
    bool converged = maxFineAdjacentDev <= 0.05f;  // 5% max between fine levels
    std::cout << "  Max fine-level deviation: "
              << std::fixed << std::setprecision(2) << (maxFineAdjacentDev * 100.0f)
              << "% (" << fineAdjacentPair << ")"
              << (converged ? "  CONVERGED" : "  NOT CONVERGED") << "\n";
    if (prodDev > 0.0f)
        std::cout << "  Production transition (0.001 -> 0.0005): "
                  << std::fixed << std::setprecision(2) << (prodDev * 100.0f)
                  << "% deviation\n";

    // PASS if fine levels are converged (stable).
    passed = converged;

    if (passed)
        std::cout << "  -->  PASS (fine levels converged, max fine dev <= 5%)\n";
    else
        std::cout << "  -->  FAIL (fine levels not converged)\n";

    // Check if conservation clamp is the culprit
    bool conservIssue = false;
    for (size_t i = 0; i < results.size(); ++i) {
        if (results[i].conservationFired && !results[0].conservationFired) {
            conservIssue = true;
            float rawDev = (refFF > 1e-10f)
                           ? std::abs(results[i].ffRaw - refFF) / refFF : 0.0f;
            float clmpDev = (refFF > 1e-10f)
                            ? std::abs(results[i].ffClamped - refFF) / refFF : 0.0f;
            if (clmpDev > rawDev * 1.5f) {
                std::cout << "  *** Energy-conservation clamp fires at subdiv="
                          << std::fixed << std::setprecision(6)
                          << results[i].targetArea
                          << " (row sum = " << results[i].maxRowSumRaw
                          << ") — this SUPPRESSES clamped F relative to raw F.\n";
            }
        }
    }
    if (conservIssue)
        std::cout << "  *** Conservation clamp first fires at a subdivision level "
                     "not triggered at coarse resolution.\n";

    return TestResult{name, passed, maxDev};
}

// ---------------------------------------------------------------------------
// Reciprocity test:  A_A × F_{A→B}  ≈  A_B × F_{B→A}
// ---------------------------------------------------------------------------

static TestResult runReciprocityTest(
    const std::string& name,
    const Mesh& scene,
    uint32_t tagA, uint32_t tagB,
    float tolerance = 0.15f)
{
    Mesh combined = scene;
    PatchBuilder::buildTriangleData(combined);

    uint32_t N = static_cast<uint32_t>(combined.numTriangles());
    std::vector<float> row;

    // Compute F_{A→B} and F_{B→A}
    auto computeAggregateFF = [&](uint32_t srcTag, uint32_t tgtTag,
                                  double& outFF, double& outArea)
    {
        outFF = 0.0; outArea = 0.0;
        FormFactor::Options opts;
        opts.clampEnergyConservation = false;
        for (uint32_t i = 0; i < N; ++i) {
            if (combined.triangle_material_id[i] != srcTag) continue;
            float Ai = combined.triangle_area[i];
            outArea += Ai;
            FormFactor::computeRowCPU(combined, i, row, opts);
            double ff = 0.0;
            for (uint32_t j = 0; j < N; ++j)
                if (combined.triangle_material_id[j] == tgtTag) ff += row[j];
            outFF += Ai * ff;
        }
        if (outArea > 1e-12) outFF /= outArea;
    };

    double ffAB, areaA, ffBA, areaB;
    computeAggregateFF(tagA, tagB, ffAB, areaA);
    computeAggregateFF(tagB, tagA, ffBA, areaB);

    double fluxAB = areaA * ffAB;
    double fluxBA = areaB * ffBA;
    double dev = (fluxAB > 1e-12)
                 ? std::abs(fluxAB - fluxBA) / std::max(fluxAB, fluxBA)
                 : 0.0;

    bool passed = dev <= tolerance;

    std::cout << "\n--- " << name << " ---\n";
    std::cout << "  A_A = " << std::fixed << std::setprecision(6) << areaA
              << "   F_{A->B} = " << ffAB
              << "   A_A*F = " << fluxAB << "\n";
    std::cout << "  A_B = " << std::fixed << std::setprecision(6) << areaB
              << "   F_{B->A} = " << ffBA
              << "   A_B*F = " << fluxBA << "\n";
    std::cout << "  Reciprocity deviation: "
              << std::fixed << std::setprecision(1) << (dev * 100.0)
              << "%" << (passed ? "  -->  PASS" : "  -->  FAIL") << "\n";

    return TestResult{name, passed, float(dev)};
}

// ---------------------------------------------------------------------------
// Diagnostic: triangle area histogram for the production Cornell Box
// ---------------------------------------------------------------------------

static void runCornellBoxDiagnostics() {
    std::cout << "\n=========================================="
                 "==========================================\n";
    std::cout << " DIAGNOSTIC: Cornell Box triangle area distribution\n";
    std::cout << "==========================================="
                 "==========================================\n";

    for (float subdiv : {0.001f, 0.0005f, 0.0002f}) {
        Mesh mesh = CornellBox::createCornellBox();
        CornellBox::fixNormalsOrientation(mesh);
        mesh = Subdivision::subdivideToUniformArea(mesh, subdiv);
        PatchBuilder::buildTriangleData(mesh);

        uint32_t N = static_cast<uint32_t>(mesh.numTriangles());

        // Classify triangles by surface
        struct SurfaceStats {
            std::string name;
            uint32_t count = 0;
            float minArea = 1e30f, maxArea = 0.0f, sumArea = 0.0f;
        };

        // Short box materials: centroid near kShortBoxCenter
        const Vec3 shortC(CornellBox::kShortBoxCenterX,
                          -0.5f + CornellBox::kShortBoxHeight * 0.5f,
                          CornellBox::kShortBoxCenterZ);
        const Vec3 tallC(CornellBox::kTallBoxCenterX,
                         -0.5f + CornellBox::kTallBoxHeight * 0.5f,
                         CornellBox::kTallBoxCenterZ);

        SurfaceStats shortBox{"short box"}, tallBox{"tall box"},
                     walls{"walls/floor/ceiling"}, light{"light"};

        for (uint32_t i = 0; i < N; ++i) {
            float a = mesh.triangle_area[i];
            const Vec3& c = mesh.triangle_centroid[i];
            SurfaceStats* s = &walls;

            if (mesh.triangle_material_id[i] == CornellBox::MAT_LIGHT)
                s = &light;
            else if ((c - shortC).length() < 0.35f && std::abs(c.x - shortC.x) < 0.25f)
                s = &shortBox;
            else if ((c - tallC).length() < 0.5f && std::abs(c.x - tallC.x) < 0.25f)
                s = &tallBox;

            s->count++;
            s->minArea = std::min(s->minArea, a);
            s->maxArea = std::max(s->maxArea, a);
            s->sumArea += a;
        }

        std::cout << "\n  SubdivTarget = " << std::fixed << std::setprecision(4) << subdiv
                  << "  (total " << N << " triangles)\n";
        std::cout << "  " << std::left
                  << std::setw(22) << "Surface"
                  << std::setw(8)  << "Count"
                  << std::setw(14) << "Min Area"
                  << std::setw(14) << "Max Area"
                  << std::setw(14) << "Avg Area"
                  << "\n";
        std::cout << "  " << std::string(72, '-') << "\n";
        for (const auto* s : {&shortBox, &tallBox, &walls, &light}) {
            if (s->count == 0) continue;
            std::cout << "  " << std::left << std::setw(22) << s->name
                      << std::setw(8)  << s->count
                      << std::setw(14) << std::scientific << std::setprecision(4) << s->minArea
                      << std::setw(14) << std::scientific << std::setprecision(4) << s->maxArea
                      << std::setw(14) << std::scientific << std::setprecision(4) << s->sumArea / s->count
                      << "\n";
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main() {
    std::cout << "======================================================="
                 "=========================\n";
    std::cout << " FORM FACTOR ROBUSTNESS TEST SUITE\n";
    std::cout << " Tests subdivision invariance of the CPU form-factor computation.\n";
    std::cout << " Production constants: distanceSoftening="
              << kDistanceSoftening
              << ", no area-dependent dist clamp\n";
    std::cout << "======================================================="
                 "=========================\n";

    std::vector<TestResult> allResults;

    // -----------------------------------------------------------------------
    // Test 1:  Parallel squares, d = 0.50 (moderate distance)
    // Expected: F should be well-behaved, no clamping issues.
    // -----------------------------------------------------------------------
    {
        auto scene = makeParallelQuads(0.50f, 0.15f);
        allResults.push_back(runSubdivInvarianceTest(
            "T1: Parallel d=0.50 (target subdiv)",
            "Two 0.30x0.30 squares facing each other at d=0.50. Subdivide TARGET only.",
            scene, TAG_A, TAG_B, kSubdivLevels));
    }

    // -----------------------------------------------------------------------
    // Test 2:  Parallel squares, d = 0.20 (closer)
    // -----------------------------------------------------------------------
    {
        auto scene = makeParallelQuads(0.20f, 0.15f);
        allResults.push_back(runSubdivInvarianceTest(
            "T2: Parallel d=0.20 (target subdiv)",
            "Two 0.30x0.30 squares at d=0.20. Closer range stresses distance clamping.",
            scene, TAG_A, TAG_B, kSubdivLevels));
    }

    // -----------------------------------------------------------------------
    // Test 3:  Parallel squares, d = 0.05 (very close — stress test)
    // Expected: this is where centroid-to-centroid breaks down.
    // -----------------------------------------------------------------------
    {
        auto scene = makeParallelQuads(0.05f, 0.15f);
        allResults.push_back(runSubdivInvarianceTest(
            "T3: Parallel d=0.05 (target subdiv)",
            "Two 0.30x0.30 squares at d=0.05. Very close — centroid approx may diverge.",
            scene, TAG_A, TAG_B, kSubdivLevels));
    }

    // -----------------------------------------------------------------------
    // Test 4:  Parallel squares d=0.50 — SOURCE subdivision
    // Expected: area-weighted F_{A→B} should be invariant.
    // -----------------------------------------------------------------------
    {
        auto scene = makeParallelQuads(0.50f, 0.15f);
        allResults.push_back(runSubdivInvarianceTest(
            "T4: Parallel d=0.50 (source subdiv)",
            "Same geometry, subdivide SOURCE instead of target.",
            scene, TAG_A, TAG_B, kSubdivLevels, 0.15f, true));
    }

    // -----------------------------------------------------------------------
    // Test 5:  Parallel squares d=0.50 — BOTH subdivided
    // Expected: should converge to the true form factor integral.
    // -----------------------------------------------------------------------
    {
        auto scene = makeParallelQuads(0.50f, 0.15f);

        // Custom test: subdivide both surfaces to each level
        Mesh srcBase = extractSurface(scene, TAG_A);
        Mesh tgtBase = extractSurface(scene, TAG_B);

        std::cout << "\n--- T5: Parallel d=0.50 (both subdivided) ---\n";
        std::cout << "  Subdivide both source and target to same level.\n\n";
        printHeader();

        std::vector<SubdivResult> bothResults;
        for (float level : kSubdivLevels) {
            Mesh s = (level > 0.0f)
                     ? Subdivision::subdivideToUniformArea(srcBase, level)
                     : srcBase;
            Mesh t = (level > 0.0f)
                     ? Subdivision::subdivideToUniformArea(tgtBase, level)
                     : tgtBase;
            Mesh combined = mergeMeshes(s, t);
            PatchBuilder::buildTriangleData(combined);

            uint32_t N = static_cast<uint32_t>(combined.numTriangles());
            uint32_t sc = 0, tc = 0;
            float tMin = 1e30f, tMax = 0, tSum = 0;
            for (uint32_t i = 0; i < N; ++i) {
                if (combined.triangle_material_id[i] == TAG_A) sc++;
                if (combined.triangle_material_id[i] == TAG_B) {
                    tc++;
                    float a = combined.triangle_area[i];
                    tMin = std::min(tMin, a); tMax = std::max(tMax, a); tSum += a;
                }
            }

            FormFactor::Options rawOpts; rawOpts.clampEnergyConservation = false;
            FormFactor::Options clampOpts;
            double rawW = 0, clampW = 0, srcA = 0;
            float maxRow = 0; bool cf = false;
            uint32_t ffcc = 0; float maxff = 0;
            std::vector<float> row;
            for (uint32_t i = 0; i < N; ++i) {
                if (combined.triangle_material_id[i] != TAG_A) continue;
                float Ai = combined.triangle_area[i]; srcA += Ai;
                FormFactor::computeRowCPU(combined, i, row, rawOpts);
                double ff = 0, rs = 0;
                for (uint32_t j = 0; j < N; ++j) {
                    rs += row[j];
                    if (combined.triangle_material_id[j] == TAG_B) ff += row[j];
                    if (row[j] >= 0.999f) ffcc++;
                    if (row[j] > maxff) maxff = row[j];
                }
                rawW += Ai * ff;
                if ((float)rs > maxRow) maxRow = (float)rs;
                if (rs > 1.0) cf = true;

                FormFactor::computeRowCPU(combined, i, row, clampOpts);
                ff = 0;
                for (uint32_t j = 0; j < N; ++j)
                    if (combined.triangle_material_id[j] == TAG_B) ff += row[j];
                clampW += Ai * ff;
            }

            SubdivResult sr{};
            sr.targetArea = level; sr.srcTris = sc; sr.tgtTris = tc;
            sr.tgtMinArea = tMin; sr.tgtMaxArea = tMax;
            sr.tgtAvgArea = tc ? tSum / tc : 0;
            sr.ffRaw = srcA > 0 ? float(rawW / srcA) : 0;
            sr.ffClamped = srcA > 0 ? float(clampW / srcA) : 0;
            sr.maxRowSumRaw = maxRow; sr.conservationFired = cf;
            sr.numFFClamped = ffcc; sr.maxSingleFF = maxff;
            printRow(sr);
            bothResults.push_back(sr);
        }

        float ref = bothResults[0].ffRaw;
        float maxDev = 0;
        for (size_t i = 1; i < bothResults.size(); ++i) {
            float d = ref > 1e-10f ? std::abs(bothResults[i].ffRaw - ref) / ref : 0;
            maxDev = std::max(maxDev, d);
        }
        bool pass = maxDev <= 0.15f;
        std::cout << "\n  Max deviation: " << std::fixed << std::setprecision(1)
                  << (maxDev * 100) << "%" << (pass ? "  -->  PASS" : "  -->  FAIL") << "\n";
        allResults.push_back({"T5: Parallel d=0.50 (both subdiv)", pass, maxDev});
    }

    // -----------------------------------------------------------------------
    // Test 6:  Perpendicular squares (L-shape)
    // Expected: form factor at grazing angle, should be invariant.
    // -----------------------------------------------------------------------
    {
        auto scene = makePerpendicularQuads(0.15f);
        allResults.push_back(runSubdivInvarianceTest(
            "T6: Perpendicular (target subdiv)",
            "Floor + wall at 90 degrees (L-shape). Subdivide wall (target).",
            scene, TAG_A, TAG_B, kSubdivLevels));
    }

    // -----------------------------------------------------------------------
    // Test 7:  Asymmetric sizes — big source, small target
    // -----------------------------------------------------------------------
    {
        auto scene = makeAsymmetricQuads(0.30f, 0.25f, 0.05f);
        allResults.push_back(runSubdivInvarianceTest(
            "T7: Asymmetric (big src, small tgt)",
            "Large source (0.50x0.50) and tiny target (0.10x0.10) at d=0.30.",
            scene, TAG_A, TAG_B, kSubdivLevels));
    }

    // -----------------------------------------------------------------------
    // Test 8:  Mini Cornell box (enclosed geometry)
    // Expected: conservation clamp should fire here.
    // -----------------------------------------------------------------------
    {
        auto scene = makeMiniCornellBox(0.5f, 0.07f);
        allResults.push_back(runSubdivInvarianceTest(
            "T8: Mini Cornell box (enclosed)",
            "Small patch inside 5-wall enclosure. Tests conservation clamp behaviour.",
            scene, TAG_A, TAG_B, kSubdivLevels));
    }

    // -----------------------------------------------------------------------
    // Test 9:  With occluder (CPU — no visibility, documents limitation)
    // Expected: CPU form factors ignore the blocker — same F as unoccluded.
    // -----------------------------------------------------------------------
    {
        auto sceneOcc = makeOccludedQuads(0.50f, 0.15f);
        auto sceneOpen = makeParallelQuads(0.50f, 0.15f);
        PatchBuilder::buildTriangleData(sceneOcc);
        PatchBuilder::buildTriangleData(sceneOpen);

        FormFactor::Options opts;
        opts.clampEnergyConservation = false;
        std::vector<float> rowOcc, rowOpen;

        Mesh meshOcc = sceneOcc;
        Mesh meshOpen = sceneOpen;
        PatchBuilder::buildTriangleData(meshOcc);
        PatchBuilder::buildTriangleData(meshOpen);

        FormFactor::computeRowCPU(meshOcc, 0, rowOcc, opts);
        FormFactor::computeRowCPU(meshOpen, 0, rowOpen, opts);

        // Sum F to TAG_B
        double ffOcc = 0, ffOpen = 0;
        for (uint32_t j = 0; j < (uint32_t)meshOcc.numTriangles(); ++j)
            if (meshOcc.triangle_material_id[j] == TAG_B) ffOcc += rowOcc[j];
        for (uint32_t j = 0; j < (uint32_t)meshOpen.numTriangles(); ++j)
            if (meshOpen.triangle_material_id[j] == TAG_B) ffOpen += rowOpen[j];

        double dev = (ffOpen > 1e-12)
                     ? std::abs(ffOcc - ffOpen) / ffOpen : 0.0;
        bool pass = dev < 0.01;  // should be identical (no visibility in CPU)

        std::cout << "\n--- T9: Occluder test (CPU — no visibility) ---\n";
        std::cout << "  F_A>B without occluder: " << std::fixed << std::setprecision(6) << ffOpen << "\n";
        std::cout << "  F_A>B with occluder:    " << std::fixed << std::setprecision(6) << ffOcc << "\n";
        std::cout << "  Deviation: " << std::fixed << std::setprecision(1) << (dev * 100.0)
                  << "% (expected: ~0% since CPU ignores visibility)\n";
        std::cout << (pass ? "  -->  PASS" : "  -->  FAIL") << "\n";
        allResults.push_back({"T9: Occluder (CPU, no vis)", pass, float(dev)});
    }

    // -----------------------------------------------------------------------
    // Test 10:  Reciprocity
    // -----------------------------------------------------------------------
    {
        auto scene = makeParallelQuads(0.30f, 0.15f);
        allResults.push_back(runReciprocityTest(
            "T10: Reciprocity (parallel d=0.30)",
            scene, TAG_A, TAG_B));
    }
    {
        auto scene = makePerpendicularQuads(0.15f);
        allResults.push_back(runReciprocityTest(
            "T11: Reciprocity (perpendicular)",
            scene, TAG_A, TAG_B));
    }

    // -----------------------------------------------------------------------
    // Cornell Box diagnostics
    // -----------------------------------------------------------------------
    runCornellBoxDiagnostics();

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    std::cout << "\n======================================================="
                 "=========================\n";
    std::cout << " SUMMARY\n";
    std::cout << "======================================================="
                 "=========================\n\n";
    int passed = 0, failed = 0;
    for (const auto& r : allResults) {
        std::cout << "  " << (r.passed ? "PASS" : "FAIL")
                  << "  " << std::left << std::setw(45) << r.name
                  << " (max dev "
                  << std::fixed << std::setprecision(1) << (r.maxDeviation * 100.0f) << "%)\n";
        if (r.passed) passed++; else failed++;
    }
    std::cout << "\n  " << passed << " passed, " << failed << " failed out of "
              << allResults.size() << " tests.\n\n";

    return failed > 0 ? 1 : 0;
}
