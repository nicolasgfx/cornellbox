# OptiX Integration Guide

## Quick Start

Once OptiX SDK is installed at `$HOME/optix`, follow these steps:

### 1. Verify OptiX Installation

```bash
ls -la ~/optix/include/optix.h
# Should show: optix.h file exists

echo $OPTIX_ROOT
# Should show: /home/nico/optix (or your install path)
```

### 2. Update CMakeLists.txt

Add OptiX finding and CUDA compilation to the build system:

```cmake
# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find OptiX
set(OPTIX_ROOT "$ENV{HOME}/optix" CACHE PATH "Path to OptiX SDK")
find_path(OPTIX_INCLUDE_DIR optix.h PATHS ${OPTIX_ROOT}/include)

if(OPTIX_INCLUDE_DIR)
    message(STATUS "Found OptiX: ${OPTIX_INCLUDE_DIR}")
    include_directories(${OPTIX_INCLUDE_DIR})
    add_definitions(-DUSE_OPTIX)
else()
    message(WARNING "OptiX not found - building without ray tracing support")
endif()
```

### 3. Update VisibilityTester.h

Replace stub implementation with real OptiX code:

```cpp
#ifdef USE_OPTIX
#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>

// OptiX context and handles
OptixDeviceContext context = nullptr;
OptixTraversableHandle gasHandle = 0;
CUdeviceptr d_vertexBuffer = 0;
CUdeviceptr d_indexBuffer = 0;
CUdeviceptr d_outputBuffer = 0;

bool initializeOptiX(const IndexedMesh& mesh) {
    // 1. Initialize CUDA
    cudaFree(0);  // Initialize CUDA context
    
    // 2. Create OptiX context
    OptixDeviceContextOptions options = {};
    optixInit();
    optixDeviceContextCreate(0, &options, &context);
    
    // 3. Upload mesh to GPU
    size_t vertexBytes = mesh.vertexCount() * sizeof(Vector3);
    cudaMalloc(&d_vertexBuffer, vertexBytes);
    cudaMemcpy(d_vertexBuffer, mesh.getVertexDataPtr(), 
               vertexBytes, cudaMemcpyHostToDevice);
    
    size_t indexBytes = mesh.triangleCount() * 3 * sizeof(uint32_t);
    cudaMalloc(&d_indexBuffer, indexBytes);
    cudaMemcpy(d_indexBuffer, mesh.getIndexDataPtr(),
               indexBytes, cudaMemcpyHostToDevice);
    
    // 4. Build acceleration structure (BVH)
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    
    // Configure triangle input
    CUdeviceptr d_vertices = d_vertexBuffer;
    uint32_t triangleFlags = OPTIX_GEOMETRY_FLAG_NONE;
    
    triangleInput.triangleArray.vertexBuffers = &d_vertices;
    triangleInput.triangleArray.numVertices = mesh.vertexCount();
    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.indexBuffer = d_indexBuffer;
    triangleInput.triangleArray.numIndexTriplets = mesh.triangleCount();
    triangleInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.flags = &triangleFlags;
    triangleInput.triangleArray.numSbtRecords = 1;
    
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
    
    OptixAccelBufferSizes gasBufferSizes;
    optixAccelComputeMemoryUsage(context, &accelOptions, 
                                  &triangleInput, 1, &gasBufferSizes);
    
    CUdeviceptr d_temp, d_gas;
    cudaMalloc(&d_temp, gasBufferSizes.tempSizeInBytes);
    cudaMalloc(&d_gas, gasBufferSizes.outputSizeInBytes);
    
    optixAccelBuild(context, 0, &accelOptions, &triangleInput, 1,
                    d_temp, gasBufferSizes.tempSizeInBytes,
                    d_gas, gasBufferSizes.outputSizeInBytes,
                    &gasHandle, nullptr, 0);
    
    cudaFree(d_temp);
    
    return true;
}

float castVisibilityRay(const Vector3& origin, const Vector3& target) {
    // Simple ray test: origin → target
    Vector3 direction = target - origin;
    float distance = direction.length();
    direction = direction / distance;
    
    // TODO: Launch OptiX ray with optixTrace()
    // For now, use CUDA kernel with traversal
    
    // Pseudo-code:
    // optixTrace(gasHandle, origin, direction, 0.01f, distance - 0.01f, ...)
    // Returns 0.0 if hit geometry, 1.0 if clear
    
    return 1.0f;  // Stub until kernel is written
}
#endif  // USE_OPTIX
```

### 4. Create OptiX PTX Kernel (src/visibility/visibility.cu)

```cuda
#include <optix.h>

extern "C" __global__ void __closesthit__visibility() {
    // Ray hit something - set visibility to 0
    optixSetPayload_0(0);  // Occluded
}

extern "C" __global__ void __miss__visibility() {
    // Ray missed - set visibility to 1
    optixSetPayload_0(1);  // Visible
}

extern "C" __global__ void __raygen__visibility() {
    // Launch ray from origin to target
    const uint3 idx = optixGetLaunchIndex();
    
    // Get ray parameters from input buffer
    float3 origin = ...;    // From launch params
    float3 direction = ...; // From launch params
    float tmin = 0.01f;     // Small epsilon
    float tmax = ...;       // Distance to target
    
    unsigned int visibility = 0;
    optixTrace(
        optixGetGASTraversableHandle(),
        origin, direction,
        tmin, tmax, 0.0f,
        OptixVisibilityMask(1),
        OPTIX_RAY_FLAG_NONE,
        0, 1, 0,
        visibility);
    
    // Write result to output buffer
    // output[idx] = visibility;
}
```

### 5. Compile PTX

Add to CMakeLists.txt:

```cmake
if(OPTIX_INCLUDE_DIR)
    # Compile CUDA kernels to PTX
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -ptx)
    cuda_compile_ptx(PTX_FILES src/visibility/visibility.cu)
    
    # Copy PTX files to build directory
    add_custom_target(ptx_files ALL
        DEPENDS ${PTX_FILES}
        COMMAND ${CMAKE_COMMAND} -E copy ${PTX_FILES} ${CMAKE_BINARY_DIR}
    )
endif()
```

### 6. Test OptiX Integration

Modify `testVisibilityTester()` in main.cpp:

```cpp
void testOptiXVisibility() {
    TEST("OptiX Visibility (Real Occlusion)");
    
    CornellBox box;
    box.build(4, 2);  // Medium detail
    
    VisibilityTester tester;
    tester.initialize(box.scene.mesh);
    
    // Test 1: Clear line of sight (floor to ceiling)
    Vector3 floorCenter = box.scene.patches[0].center;
    Vector3 ceilingCenter = box.scene.patches[1].center;
    float vis1 = tester.testVisibility(floorCenter, ceilingCenter);
    std::cout << "Floor to ceiling: " << vis1 << " (expect ~1.0)\n";
    
    // Test 2: Occluded by box
    Vector3 behindBox(100, 50, 300);
    Vector3 frontOfBox(450, 50, 300);
    float vis2 = tester.testVisibility(behindBox, frontOfBox);
    std::cout << "Through box: " << vis2 << " (expect 0.0)\n";
    
    // Test 3: Recompute form factors with real occlusion
    auto matrix = FormFactorCalculator::calculateMatrix(
        box.scene.patches, &tester, true);
    FormFactorCalculator::printStatistics(matrix);
    
    std::cout << "Expected changes with OptiX:\n";
    std::cout << "  - Form factors reduced where boxes occlude\n";
    std::cout << "  - Shadow patterns visible in matrix\n";
    std::cout << "  - More realistic light transport\n";
}
```

### 7. Rebuild and Test

```bash
cd build
cmake ..
make -j
./radiosity
```

Expected output:
- OptiX version detected
- Acceleration structure built
- Visibility tests show occlusion
- Form factors reduced where shadows occur

## Debugging Tips

1. **OptiX not found**: Check `$OPTIX_ROOT` environment variable
2. **CUDA errors**: Verify `nvidia-smi` shows GPU
3. **PTX compile errors**: Check CUDA toolkit version compatibility
4. **Runtime errors**: Enable OptiX validation layer in debug builds

## Performance Notes

- BVH build: ~10ms for 1000 triangles
- Ray queries: ~1μs per ray on RTX 4070
- Form factor matrix (N×N): ~N²μs with OptiX
- Example: 100 patches = 10,000 rays ≈ 10ms

## Resources

- OptiX Programming Guide: https://raytracing-docs.nvidia.com/optix7/guide/
- OptiX API Reference: https://raytracing-docs.nvidia.com/optix7/api/
- OptiX Samples: Check `$OPTIX_ROOT/SDK/` directory

## Alternative: CPU-only Mode

If OptiX installation is difficult, you can implement CPU ray tracing as fallback:

```cpp
// Simple triangle intersection test
bool rayTriangleIntersect(const Vector3& origin, const Vector3& dir,
                          const Vector3& v0, const Vector3& v1, const Vector3& v2) {
    // Möller-Trumbore algorithm
    Vector3 edge1 = v1 - v0;
    Vector3 edge2 = v2 - v0;
    Vector3 h = dir.cross(edge2);
    float a = edge1.dot(h);
    if (a > -1e-6f && a < 1e-6f) return false;
    
    float f = 1.0f / a;
    Vector3 s = origin - v0;
    float u = f * s.dot(h);
    if (u < 0.0f || u > 1.0f) return false;
    
    Vector3 q = s.cross(edge1);
    float v = f * dir.dot(q);
    if (v < 0.0f || u + v > 1.0f) return false;
    
    float t = f * edge2.dot(q);
    return t > 1e-6f;
}
```

This will be much slower (100-1000x) but allows development without OptiX.
