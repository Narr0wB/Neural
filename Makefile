all: compile run

compile:
	nvcc -O3 main.cpp image.cpp neural.cpp convneural.cpp Conv2D.cpp cudalinear/linalg.cpp cudalinear/cudalinear.cu -g -o test

run:
	./test

.PHONY: all run
