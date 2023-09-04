cmake_minimum_required(VERSION 3.10)

add_subdirectory(cudalinear)

if (DEFINED __GPU_ACCELERATION_CUDA)

add_compile_definitions(__GPU_ACCELERATION_CUDA)

endif (DEFINED __GPU_ACCELERATION_CUDA)

add_executable(neural main.cpp neural.cpp image.cpp)

target_link_libraries(neural PRIVATE linalg)

