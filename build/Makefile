ifeq ($(__CUDA), 1)

CC=nvcc
LINEAR= ../src/linalg/cudalinear.cu \
	../src/linalg/linalg.cpp
CFLAGS = -O3 -arch=compute_61 -D__GPU_ACCELERATION_CUDA

else 

CC=g++
LINEAR= ../src/linalg/linalg.cpp
CFLAGS = -O3 -D__CPU

endif

all: build

build:
	$(CC) $(CFLAGS) $(LINEAR) ../src/*.cpp -o neural