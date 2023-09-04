cmake_minimum_required(VERSION 3.10)

if (DEFINED __GPU_ACCELERATION_CUDA)

add_library(linalg SHARED 
    linalg.cpp 
    cudalinear.cu
)

else (DEFINED __GPU_ACCELERATION_CUDA)

add_library(linalg SHARED 
    linalg.cpp 
)

endif (DEFINED __GPU_ACCELERATION_CUDA)