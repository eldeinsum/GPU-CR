cmake_minimum_required(VERSION 3.18)
project(vGPU LANGUAGES CXX)

option(GPUCR_BUILD_CPP "Build legacy C++ GPU-CR artifacts" ON)
option(GPUCR_BUILD_RUST "Build Rust GPU-CR artifacts" OFF)
option(GPUCR_RUST_RELEASE "Build Rust artifacts in release mode" OFF)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Detect GPU vendor from -DGPU_VENDOR first, then the environment.
# Usage: cmake -DGPU_VENDOR=NVIDIA .. or GPU_VENDOR=NVIDIA cmake ..
if(NOT DEFINED GPU_VENDOR AND DEFINED ENV{GPU_VENDOR})
    set(GPU_VENDOR $ENV{GPU_VENDOR})
elseif(NOT DEFINED GPU_VENDOR)
    set(GPU_VENDOR "NVIDIA")  # Default to NVIDIA
endif()
set(GPU_VENDOR "${GPU_VENDOR}" CACHE STRING "GPU vendor backend")
set_property(CACHE GPU_VENDOR PROPERTY STRINGS NVIDIA AMD)

message(STATUS "Building for GPU vendor: ${GPU_VENDOR}")
string(TOLOWER "${GPU_VENDOR}" GPU_VENDOR_LOWER)

# Set paths based on vendor
if(GPU_VENDOR STREQUAL "NVIDIA")
    set(CUDA_ROOT "/usr/local/cuda" CACHE PATH "CUDA installation directory")
    set(GPU_INCLUDE_DIR "${CUDA_ROOT}/include")
    set(GPU_LIB_DIR "${CUDA_ROOT}/lib64")
    set(GPU_LIBS cudart cuda)
elseif(GPU_VENDOR STREQUAL "AMD")
    set(ROCM_ROOT "/opt/rocm" CACHE PATH "ROCm installation directory")
    set(GPU_INCLUDE_DIR "${ROCM_ROOT}/include")
    set(GPU_LIB_DIR "${ROCM_ROOT}/lib")
    set(GPU_LIBS amdhip64)
    # Add HIP platform definition
    add_compile_definitions(__HIP_PLATFORM_AMD__=1)
else()
    message(FATAL_ERROR "Unknown GPU_VENDOR: ${GPU_VENDOR}. Must be NVIDIA or AMD")
endif()

if(GPUCR_BUILD_CPP)
# Collect source and header files automatically from src/
file(GLOB_RECURSE VGPU_SOURCE_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

# 排除不需要编译的文件
list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/cr_client\\.cpp$")
list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "-pre\\.cpp$")     # 排除 *-pre.cpp 文件

# 根据 GPU vendor 排除对应的目录
if(GPU_VENDOR STREQUAL "NVIDIA")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/AMD/")       # 排除 AMD 目录
    message(STATUS "Excluding AMD files")
elseif(GPU_VENDOR STREQUAL "AMD")
    list(FILTER VGPU_SOURCE_FILES EXCLUDE REGEX "/NVIDIA/")    # 排除 NVIDIA 目录
    message(STATUS "Excluding NVIDIA files")
endif()

file(GLOB_RECURSE VGPU_HEADER_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.hpp"
)

set(SOURCES
    ${VGPU_SOURCE_FILES}
    ${VGPU_HEADER_FILES}
)

# Create shared library for LD_PRELOAD
add_library(vGPU SHARED ${SOURCES})

# Set library name based on vendor (vGPU-nvidia.so or vGPU-amd.so)
set_target_properties(vGPU PROPERTIES
    OUTPUT_NAME "vGPU-${GPU_VENDOR}"
    PREFIX ""
)

# Include directories
target_include_directories(vGPU PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

# Link directories
target_link_directories(vGPU PRIVATE
    ${GPU_LIB_DIR}
)

# Link libraries
target_link_libraries(vGPU PRIVATE
    ${GPU_LIBS}
    dl
    stdc++
    pthread
)

# Compiler flags (matching gcc command: -fPIC -O3 -g)
target_compile_options(vGPU PRIVATE
    -fPIC
    -O3
    -g
)

# Linker flags (matching gcc command: -shared -fPIC)
target_link_options(vGPU PRIVATE
    -shared
    -fPIC
)

# Export all symbols for LD_PRELOAD (needed for constructor and signal handlers)
set_target_properties(vGPU PROPERTIES
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

add_executable(cr_client
    ${CMAKE_CURRENT_SOURCE_DIR}/src/cr_client.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/comm/share_mem.cpp
)

target_include_directories(cr_client PRIVATE
    ${GPU_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_link_directories(cr_client PRIVATE
    ${GPU_LIB_DIR}
)

target_link_libraries(cr_client PRIVATE
    ${GPU_LIBS}
    dl
    stdc++
    pthread
)

target_compile_options(cr_client PRIVATE
    -O3
    -g
)
endif()

if(GPUCR_BUILD_RUST)
    if(NOT GPU_VENDOR STREQUAL "NVIDIA")
        message(FATAL_ERROR "The Rust runtime currently supports NVIDIA only")
    endif()
    find_program(CARGO_EXECUTABLE cargo REQUIRED)

    set(GPUCR_RUST_ARGS build --manifest-path "${CMAKE_CURRENT_SOURCE_DIR}/rust/Cargo.toml" --workspace)
    set(GPUCR_RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust/target/debug")
    if(GPUCR_RUST_RELEASE)
        list(APPEND GPUCR_RUST_ARGS --release)
        set(GPUCR_RUST_TARGET_DIR "${CMAKE_CURRENT_SOURCE_DIR}/rust/target/release")
    endif()

    add_custom_target(gpucr_nvidia_runtime ALL
        COMMAND "${CARGO_EXECUTABLE}" ${GPUCR_RUST_ARGS}
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${GPUCR_RUST_TARGET_DIR}/libgpucr_preload.so"
                "${CMAKE_CURRENT_BINARY_DIR}/gpucr-${GPU_VENDOR_LOWER}.so"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "${GPUCR_RUST_TARGET_DIR}/gpucr-client"
                "${CMAKE_CURRENT_BINARY_DIR}/gpucr-client"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
        COMMENT "Building GPU-CR NVIDIA runtime"
    )
endif()
