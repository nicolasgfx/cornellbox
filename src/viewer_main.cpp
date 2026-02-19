// Interactive radiosity scene viewer — entry point.
// Usage: viewer [--scene PATH]
//   Default: Cornell Box.
//   --scene PATH: load OBJ scene from given path.

#define NOMINMAX

// Provide stb_image_write implementation for non-OptiX builds.
// When USE_OPTIX is defined, Renderer.h (included via Viewer.h) provides it.
#ifndef USE_OPTIX
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#undef STB_IMAGE_WRITE_IMPLEMENTATION
#endif

#include "app/Config.h"
#include "viewer/Viewer.h"
#include <iostream>

int main(int argc, char** argv) {
    std::string scenePath;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--scene") && i + 1 < argc) {
            scenePath = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Radiosity Viewer\n\n"
                      << "Usage: viewer [--scene PATH]\n\n"
                      << "  --scene PATH   Load OBJ scene (default: Cornell Box)\n"
                      << "  --help, -h     Show this help\n\n"
                      << "Controls:\n"
                      << "  WASD           Move camera\n"
                      << "  Mouse          Look around\n"
                      << "  Shift          Move faster\n"
                      << "  Space/Ctrl     Move up/down\n"
                      << "  R / LMB        Run radiosity solve\n"
                      << "  ESC            Release mouse / quit\n";
            return 0;
        }
    }

    return Viewer::run(scenePath);
}
