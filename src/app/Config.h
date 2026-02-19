#pragma once
#include <string>
#include <iostream>
#include <cstdint>

// ---------------------------------------------------------------------------
// Tuning constants — edit these to control quality vs. speed.
// ---------------------------------------------------------------------------

// Target triangle area for adaptive subdivision (lower = more triangles).
inline constexpr float kSubdivisionTargetArea = 0.001f; // 0.001f

// Maximum absolute triangle area for OBJ scenes (scene-scale independent).
// Caps the effective target so large-scene patches aren't oversized.
// Matches Cornell Box density when set equal to kSubdivisionTargetArea.
// 0 = disabled (use scene-scaled target only).
inline constexpr float kMaxAbsoluteTriangleArea = 0.005f;

// Maximum edge-length ratio (longest / shortest) before shape refinement.
// Elongated slivers hurt radiosity accuracy; the longest edge is split.
// 0 = disabled.
inline constexpr float kMaxTriangleEdgeRatio = 4.0f;

// Maximum triangle count budget for subdivision (uniform + FF combined).
inline constexpr uint32_t kMaxSubdivisionTriangles = 5000000;

// Light source brightness multiplier (1.0 = Cornell spec).
inline constexpr float kLightBrightnessScale = 1.0f; // 1.0f

// GPU form-factor computation via OptiX ray-traced visibility.
inline constexpr bool  kEnableGPUFormFactors  = true; // true
inline constexpr uint32_t kVisibilitySamples  = 16; // 16 to 64

// Indirect-light boost applied to reflected energy each bounce.
// 1.0 = strict physics.  Slightly >1 brightens indirect light.
inline constexpr float kIndirectBoostFactor = 1.2f; // 1.2f

// Distance softening added to r² in the form-factor denominator.
// 0.0 = standard physics.  Small positive values soften near-field contrast.
inline constexpr float kDistanceSoftening = 0.0001f;

// Form-factor-driven adaptive refinement.
// Enables iterative subdivision where interacting patches are too coarse
// relative to the distance between them (near-contact, corners, folds).
inline constexpr bool  kEnableFFRefinement        = true;
inline constexpr float kFFRefinementAccuracyRatio  = 0.5f;   // subdivide when h/d > this
inline constexpr float kFFRefinementMinFormFactor  = 1e-5f;  // ignore weak interactions
inline constexpr uint32_t kFFRefinementMaxPasses   = 10;     // max refinement iterations
inline constexpr bool  kFFRefinementSplitAll3Edges = false;  // false=longest-edge, true=1-to-4

// Tone mapping: exposure scales linear values, gamma compresses dynamic range.
inline constexpr float kToneMapExposure = 1.2f; // 1.4f
inline constexpr float kToneMapGamma   = 0.6f; // 0.8f

// Render output resolution (OptiX ray-traced PNG).
inline constexpr uint32_t kRenderWidth  = 3840;
inline constexpr uint32_t kRenderHeight = 3840;

// Viewer window resolution.
inline constexpr int kViewerWidth  = 2560;
inline constexpr int kViewerHeight = 1440;

// Camera: classic Cornell Box front view.
inline constexpr float kCameraEyeX = 0.0f;
inline constexpr float kCameraEyeY = 0.0f;
inline constexpr float kCameraEyeZ = 1.94f;
inline constexpr float kCameraFovY = 39.3f;   // degrees (Cornell spec)

// Progressive refinement animation: capture frames during solve.
// Two-pass content-aware: analysis pass measures visual change per iteration,
// then render pass allocates screen time proportional to change magnitude.
// Slow where the scene changes dramatically, fast where it barely changes.
inline constexpr bool     kEnableProgressiveAnimation = false;
inline constexpr uint32_t kAnimationFPS               = 30;
inline constexpr float    kAnimationDurationSeconds   = 10.0f;

// ---------------------------------------------------------------------------
// Command-line config
// ---------------------------------------------------------------------------

struct Config {
    std::string outputPath = "output";
    std::string scenePath;            // empty = Cornell Box, else path to .obj
    bool validate = true;

    bool useOBJScene() const { return !scenePath.empty(); }

    bool parseArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                outputPath = argv[++i];
            } else if (arg == "--scene" && i + 1 < argc) {
                scenePath = argv[++i];
            } else if (arg == "--no-validate") {
                validate = false;
            } else if (arg == "--help" || arg == "-h") {
                printHelp();
                return false;
            } else {
                std::cerr << "Unknown argument: " << arg << "\n";
                printHelp();
                return false;
            }
        }
        return true;
    }

    void printHelp() const {
        std::cout << "Radiosity Renderer\n\n"
                  << "Usage: radiosity [options]\n\n"
                  << "Options:\n"
                  << "  --scene PATH    Load OBJ scene (default: Cornell Box)\n"
                  << "  --output PATH   Output directory (default: output)\n"
                  << "  --no-validate   Skip mesh validation\n"
                  << "  --help, -h      Show this help\n\n"
                  << "Tuning constants are in src/app/Config.h.\n";
    }
};
