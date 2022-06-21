all: compile run

compile:
	nvcc main.cpp image.cpp neural.cpp cudalinear/linalg.cpp cudalinear/cudalinear.cu -o test

run:
	./test.exe

.PHONY: all run
