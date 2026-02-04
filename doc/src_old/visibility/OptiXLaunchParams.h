#pragma once

#include <optix.h>

namespace radiosity {
namespace visibility {

struct TriangleData {
    float v0[3];
    float v1[3];
    float v2[3];
    int patch_id;
};

struct RayGenParams {
    TriangleData source;
    TriangleData target;
    float source_uv[2];
    float target_uv[2];
    int sample_count;
    int result_offset;
    OptixTraversableHandle traversable;
    unsigned int* results;
};

} // namespace visibility
} // namespace radiosity

static_assert(sizeof(radiosity::visibility::TriangleData) == 40, "Unexpected TriangleData size");
static_assert(sizeof(radiosity::visibility::RayGenParams) == 120, "Unexpected RayGenParams size");

#ifdef __CUDACC__
using radiosity::visibility::TriangleData;
using radiosity::visibility::RayGenParams;
#endif
