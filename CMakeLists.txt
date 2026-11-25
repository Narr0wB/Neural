cmake_minimum_required(VERSION 3.20)
project(Neural LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(USE_CUDA "Enable CUDA GPU acceleration" OFF)

# Core library (reusable by both the main executable and future tests)
add_library(neural_core
    src/dataset.cpp
    src/datasets/mnist.cpp
)
target_include_directories(neural_core PUBLIC src/include)
target_compile_options(neural_core PUBLIC
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(neural_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    target_sources(neural_core PRIVATE src/cuda/cuda_ops.cu)
    target_compile_definitions(neural_core PUBLIC __GPU_ACCELERATION_CUDA)

    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(neural_core PUBLIC CUDA::cudart CUDA::cublas)
endif()

# Main executable
add_executable(neural
    src/main.cpp
)
target_link_libraries(neural PRIVATE neural_core)
