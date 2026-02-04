cmake_minimum_required(VERSION 3.21)
project(RadiosityRenderer LANGUAGES CXX CUDA)

# C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Fast build optimizations
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES OFF)

# Output directories (Ninja-friendly)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "=== Radiosity Renderer ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Generator: ${CMAKE_GENERATOR}")

# OptiX ray tracing (enabled by default)
option(USE_OPTIX "Enable OptiX ray tracing" ON)

if(USE_OPTIX)
    message(STATUS "OptiX: ENABLED")
    
    # Find OptiX SDK
    set(OptiX_INSTALL_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0" 
        CACHE PATH "OptiX SDK install directory")
    
    if(NOT EXISTS "${OptiX_INSTALL_DIR}")
        message(FATAL_ERROR "OptiX SDK not found at: ${OptiX_INSTALL_DIR}")
    endif()
    
    # Find CUDA Toolkit
    find_package(CUDAToolkit REQUIRED)
    
    message(STATUS "  OptiX SDK: ${OptiX_INSTALL_DIR}")
    message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
    
    add_definitions(-DUSE_OPTIX)
else()
    message(STATUS "OptiX: DISABLED (CPU-only mode)")
endif()

# Source files
set(SOURCES
    src/main.cpp
)

# Header-only files (for IDE organization)
set(HEADERS
    src/math/Vec3.h
    src/math/MathUtils.h
    src/math/HemisphereSampling.h
    src/mesh/MeshData.h
    src/mesh/Subdivision.h
    src/mesh/PatchBuilder.h
    src/scene/CornellBox.h
    src/scene/OBJLoader.h
    src/export/VertexColor.h
    src/export/OBJExporter.h
    src/export/AOExporter.h
    src/visibility/SamplePatterns.h
    src/visibility/VisibilityCache.h
    src/visibility/VisibilityCompute.h
    src/radiosity/FormFactor.h
    src/radiosity/FormFactorCache.h
    src/radiosity/FormFactorPrecompute.h
    src/radiosity/Solver.h
    src/gpu/OptiXContext.h
    src/app/Config.h
)

# OptiX PTX kernel compilation
if(USE_OPTIX)
    set(CUDA_SOURCES 
        src/gpu/optix_kernels.cu
        src/gpu/HemisphereFormFactorKernels.cu
    )
    
    # Optimized PTX compilation flags
    set(CUDA_NVCC_FLAGS
        -ptx
        -use_fast_math
        -O3
        -I${OptiX_INSTALL_DIR}/include
        -I${CUDAToolkit_INCLUDE_DIRS}
        -I${CMAKE_SOURCE_DIR}/src
    )
    
    # Compile main OptiX kernel to PTX
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/optix_kernels.ptx
        COMMAND ${CMAKE_CUDA_COMPILER}
            ${CUDA_NVCC_FLAGS}
            -o ${CMAKE_BINARY_DIR}/optix_kernels.ptx
            ${CMAKE_SOURCE_DIR}/src/gpu/optix_kernels.cu
        DEPENDS ${CMAKE_SOURCE_DIR}/src/gpu/optix_kernels.cu
        COMMENT "Compiling OptiX kernel to PTX"
        VERBATIM
    )
    
    # Compile hemisphere form factor kernel to PTX
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/hemisphere_kernels.ptx
        COMMAND ${CMAKE_CUDA_COMPILER}
            ${CUDA_NVCC_FLAGS}
            -o ${CMAKE_BINARY_DIR}/hemisphere_kernels.ptx
            ${CMAKE_SOURCE_DIR}/src/gpu/HemisphereFormFactorKernels.cu
        DEPENDS 
            ${CMAKE_SOURCE_DIR}/src/gpu/HemisphereFormFactorKernels.cu
            ${CMAKE_SOURCE_DIR}/src/math/HemisphereSampling.h
        COMMENT "Compiling Hemisphere Form Factor kernel to PTX"
        VERBATIM
    )
    
    # PTX target
    add_custom_target(compile_ptx ALL
        DEPENDS 
            ${CMAKE_BINARY_DIR}/optix_kernels.ptx
            ${CMAKE_BINARY_DIR}/hemisphere_kernels.ptx
    )
endif()

# Create executable
add_executable(radiosity ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(radiosity PRIVATE
    ${CMAKE_SOURCE_DIR}/src
)

# Link OptiX and CUDA if enabled
if(USE_OPTIX)
    target_include_directories(radiosity PRIVATE
        ${OptiX_INSTALL_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    
    target_link_libraries(radiosity PRIVATE
        CUDA::cudart
        CUDA::cuda_driver
    )
    
    add_dependencies(radiosity compile_ptx)
endif()

# Platform-specific compiler settings
if(MSVC)
    target_compile_options(radiosity PRIVATE 
        /W3           # Warning level 3
        /MP           # Parallel compilation
    )
    target_compile_definitions(radiosity PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(radiosity PRIVATE -Wall -Wextra)
endif()

# Debug configuration
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(radiosity PRIVATE RAD_DEBUG)
endif()

# Copy output to project output folder for convenience
add_custom_command(TARGET radiosity POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_SOURCE_DIR}/output/bin
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:radiosity> ${CMAKE_SOURCE_DIR}/output/bin/
    COMMENT "Copying to output/bin/"
)

# Summary
message(STATUS "=========================")
message(STATUS "Configuration complete")
message(STATUS "=========================")

