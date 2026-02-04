#pragma once
#include <string>
#include <iostream>

enum class Profile {
    LOW,
    MEDIUM,
    HIGH
};

struct Config {
    Profile profile = Profile::LOW;
    std::string outputPath = "output/scenes";
    bool validate = true;
    int phase = 2;
    
    // Parse command line arguments
    bool parseArgs(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "--low") {
                profile = Profile::LOW;
            }
            else if (arg == "--medium") {
                profile = Profile::MEDIUM;
            }
            else if (arg == "--high") {
                profile = Profile::HIGH;
            }
            else if (arg == "--output" && i + 1 < argc) {
                outputPath = argv[++i];
            }
            else if (arg == "--no-validate") {
                validate = false;
            }
            else if (arg == "--phase" && i + 1 < argc) {
                phase = std::atoi(argv[++i]);
            }
            else if (arg == "--help" || arg == "-h") {
                printHelp();
                return false;
            }
            else {
                std::cerr << "Unknown argument: " << arg << std::endl;
                printHelp();
                return false;
            }
        }
        return true;
    }
    
    std::string getProfileName() const {
        switch (profile) {
            case Profile::LOW: return "low";
            case Profile::MEDIUM: return "medium";
            case Profile::HIGH: return "high";
            default: return "unknown";
        }
    }
    
    // Target triangle area for adaptive subdivision
    // Triangles larger than this will be subdivided
    float getTargetArea() const {
        switch (profile) {
            case Profile::LOW: return 0.002f;      // ~4000 triangles
            case Profile::MEDIUM: return 0.0005f;  // ~16000 triangles (4x refined)
            case Profile::HIGH: return 0.000125f;  // ~64000 triangles (4x refined)
            default: return 0.002f;
        }
    }
    
    void printHelp() const {
        std::cout << "Radiosity Cornell Box\n\n";
        std::cout << "Usage: radiosity [options]\n\n";
        std::cout << "Options:\n";
        std::cout << "  --low           Low profile (~4K triangles)\n";
        std::cout << "  --medium        Medium profile (~16K triangles)\n";
        std::cout << "  --high          High profile (~64K triangles)\n";
        std::cout << "  --phase N       Run phases up to N (1=geometry, 2=radiosity, default: 2)\n";
        std::cout << "  --output PATH   Output directory (default: output/scenes)\n";
        std::cout << "  --no-validate   Skip validation\n";
        std::cout << "  --help, -h      Show help\n";
    }
};
