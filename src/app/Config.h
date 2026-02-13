#pragma once
#include <string>
#include <iostream>
#include <cstdint>

// ---------------------------------------------------------------------------
// Tuning constants — edit these to control quality vs. speed.
// ---------------------------------------------------------------------------

// Target triangle area for adaptive subdivision (lower = more triangles).
inline constexpr float kSubdivisionTargetArea = 0.001f;

// Light source brightness multiplier (1.0 = Cornell spec).
inline constexpr float kLightBrightnessScale = 1.0f;

// GPU form-factor computation via OptiX ray-traced visibility.
inline constexpr bool  kEnableGPUFormFactors  = true;
inline constexpr uint32_t kVisibilitySamples  = 32;

// Indirect-light boost applied to reflected energy each bounce.
// 1.0 = strict physics.  Slightly >1 brightens indirect light.
inline constexpr float kIndirectBoostFactor = 1.3f;

// Distance softening added to r² in the form-factor denominator.
// 0.0 = standard physics.  Small positive values soften near-field contrast.
inline constexpr float kDistanceSoftening = 0.0001f;

// Tone mapping: exposure scales linear values, gamma compresses dynamic range.
inline constexpr float kToneMapExposure = 1.4f;
inline constexpr float kToneMapGamma   = 0.8f;

// Render output resolution (OptiX ray-traced PNG).
inline constexpr uint32_t kRenderWidth  = 3840;
inline constexpr uint32_t kRenderHeight = 3840;

// Camera: classic Cornell Box front view.
inline constexpr float kCameraEyeX = 0.0f;
inline constexpr float kCameraEyeY = 0.0f;
inline constexpr float kCameraEyeZ = 1.94f;
inline constexpr float kCameraFovY = 39.3f;   // degrees (Cornell spec)

// ---------------------------------------------------------------------------
// Command-line config
// ---------------------------------------------------------------------------

struct Config {
    std::string outputPath = "output";
    bool validate = true;

    bool parseArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--output" && i + 1 < argc) {
                outputPath = argv[++i];
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
        std::cout << "Radiosity Cornell Box Renderer\n\n"
                  << "Usage: radiosity [options]\n\n"
                  << "Options:\n"
                  << "  --output PATH   Output directory (default: output)\n"
                  << "  --no-validate   Skip mesh validation\n"
                  << "  --help, -h      Show this help\n\n"
                  << "Tuning constants are in src/app/Config.h.\n";
    }
};
