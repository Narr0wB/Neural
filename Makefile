all: compile run

compile:
	nvcc main.cpp image.cpp neural.cu cudalinear/linalg.cpp cudalinear/cudalinear.cu -o test

run:
	nvprof ./test.exe

.PHONY: all run
