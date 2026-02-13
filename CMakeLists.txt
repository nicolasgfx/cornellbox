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

# Default to Release only for single-config generators (Ninja/Makefiles).
# Multi-config generators (Visual Studio) choose config at build time.
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "=== Radiosity Renderer ===")
if(CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Build type: multi-config (${CMAKE_CONFIGURATION_TYPES})")
else()
    message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "Generator: ${CMAKE_GENERATOR}")

# OptiX ray tracing (enabled by default)
option(USE_OPTIX "Enable OptiX ray tracing" ON)
option(USE_OPENMP "Enable OpenMP parallelism" ON)

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
    src/export/OBJExporter.h
    src/gpu/OptiXContext.h
    src/gpu/Renderer.h
    src/app/Config.h
)

# OptiX PTX kernel compilation
if(USE_OPTIX)
    # Optimized PTX compilation flags
    set(CUDA_NVCC_FLAGS
        -ptx
        -use_fast_math
        -O3
        -I${OptiX_INSTALL_DIR}/include
        -I${CUDAToolkit_INCLUDE_DIRS}
        -I${CMAKE_SOURCE_DIR}/src
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

    # Compile render kernel to PTX
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/render_kernels.ptx
        COMMAND ${CMAKE_CUDA_COMPILER}
            ${CUDA_NVCC_FLAGS}
            -o ${CMAKE_BINARY_DIR}/render_kernels.ptx
            ${CMAKE_SOURCE_DIR}/src/gpu/RenderKernels.cu
        DEPENDS
            ${CMAKE_SOURCE_DIR}/src/gpu/RenderKernels.cu
        COMMENT "Compiling Render kernel to PTX"
        VERBATIM
    )
    
    # PTX target
    add_custom_target(compile_ptx ALL
        DEPENDS
            ${CMAKE_BINARY_DIR}/hemisphere_kernels.ptx
            ${CMAKE_BINARY_DIR}/render_kernels.ptx
    )
endif()

# Create executable
add_executable(radiosity ${SOURCES} ${HEADERS})

# Include directories
target_include_directories(radiosity PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
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
        $<$<CONFIG:Debug>:/Od>   # Disable optimization in Debug
        $<$<CONFIG:Debug>:/Zi>   # Generate debug symbols in Debug
    )
    if(USE_OPENMP)
        target_compile_options(radiosity PRIVATE /openmp)
        message(STATUS "OpenMP: ENABLED (MSVC /openmp, no extra install required)")
    else()
        message(STATUS "OpenMP: DISABLED")
    endif()
    target_link_options(radiosity PRIVATE
        $<$<CONFIG:Debug>:/DEBUG:FULL>  # Full PDBs for reliable breakpoints
    )
    target_compile_definitions(radiosity PRIVATE _CRT_SECURE_NO_WARNINGS)
else()
    target_compile_options(radiosity PRIVATE -Wall -Wextra)
    if(USE_OPENMP)
        find_package(OpenMP REQUIRED COMPONENTS CXX)
        target_link_libraries(radiosity PRIVATE OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP: ENABLED")
    else()
        message(STATUS "OpenMP: DISABLED")
    endif()
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
