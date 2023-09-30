ifeq ($(__CUDA), 1)

CC=nvcc
CUDALINEAR= ../src/linalg/cudalinear.cu \
	../src/linalg/linalg.cpp
CUDADEFINE= __GPU_ACCELERATION_CUDA
CUDAFLAGS=-arch=compute_61

else 

CC=g++
CUDALINEAR= ../src/linalg/linalg.cpp
CUDADEFINE= __CPU
CUDAFLAGS= 

endif


CFLAGS = -O3

all: build

build:
	$(CC) $(CFLAGS) $(CUDALINEAR) ../src/image.cpp ../src/neural.cpp ../src/main.cpp -D$(CUDADEFINE) -o neural $(CUDAFLAGS)

run: neural
	neural.exe