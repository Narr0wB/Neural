cmake_minimum_required(VERSION 3.20)
project(NNeural LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

# Core library (reusable by both the main executable and future tests)
add_library(nneural_core
    src/dataset.cpp
    src/datasets/mnist.cpp
)
target_include_directories(nneural_core PUBLIC src/include)
target_compile_options(nneural_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3 -march=native>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
target_compile_definitions(nneural_core PUBLIC
    $<$<NOT:$<CONFIG:Debug>>:NDEBUG>
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(nneural_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_sources(nneural_core PRIVATE src/cuda/cuda_ops.cu)
    target_compile_definitions(nneural_core PUBLIC __GPU_ACCELERATION_CUDA)

    find_package(CUDAToolkit REQUIRED)
    find_path(CUDNN_INCLUDE_DIR cudnn.h
        HINTS ${CUDAToolkit_INCLUDE_DIRS} $ENV{CUDNN_PATH}/include $ENV{CUDNN_HOME}/include $ENV{CUDA_PATH}/include
    )
    find_library(CUDNN_LIBRARY NAMES cudnn
        HINTS ${CUDAToolkit_LIBRARY_DIR}
              $ENV{CUDNN_PATH}/lib64 $ENV{CUDNN_PATH}/lib $ENV{CUDNN_PATH}/lib/x64
              $ENV{CUDNN_HOME}/lib64 $ENV{CUDNN_HOME}/lib
              $ENV{CUDA_PATH}/lib64  $ENV{CUDA_PATH}/lib/x64
    )
    if(NOT CUDNN_INCLUDE_DIR OR NOT CUDNN_LIBRARY)
        message(FATAL_ERROR "cuDNN not found. Set CUDNN_INCLUDE_DIR and CUDNN_LIBRARY (or CUDNN_PATH).")
    endif()

    target_include_directories(nneural_core PRIVATE ${CUDNN_INCLUDE_DIR})
    target_link_libraries(nneural_core PUBLIC CUDA::cudart CUDA::cublas ${CUDNN_LIBRARY})
endif()

# Main executable
add_executable(nneural
    src/main.cpp
)
target_link_libraries(nneural PRIVATE nneural_core)
