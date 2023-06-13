all: cpu run

gpu:
	nvcc main.cpp image.cpp neural.cpp cudalinear/linalg.cpp cudalinear/cudalinear.cu -D__GPU_ACCELERATION_CUDA -o test
cpu:
	g++ main.cpp image.cpp neural.cpp cudalinear/linalg.cpp -o test
run:
	./test.exe

.PHONY: all run
