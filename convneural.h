
#ifndef CONVNEURAL_H
#define CONVNEURAL_H

#include <fstream>
#include <chrono>

#include "cudalinear/linalg.h"
#include "image.h"
#include "Conv2D.h"
#include "neural.h"

struct ConvNN
{
    private:
        MatrixList kernels;
        MatrixList cB0;

        Matrix W0; 
        Matrix B0;

        Matrix W1;
        Matrix B1;
    
    public:
        ConvNN(ImageList data_set, size_t epochs, double learn_rate, size_t kernel = 1, bool verbose = false, size_t batch_size = 100) : kernels(kernel, 3, 3), cB0(4, 26, 26), W0(300, 26*26*kernel), B0(300, 1), W1(10, 300), B1(10, 1)
        {
            kernels.randn(0, 1);
            cB0.randn(0, 1);
            W0.randn(0, 1);
            B0.randn(0, 1);
            W1.randn(0, 1);
            B1.randn(0, 1);
            train(data_set, epochs, learn_rate, verbose, batch_size);
        };

        ConvNN(const char* path, size_t kernel = 1) : kernels(kernel, 3, 3), cB0(4, 26, 26), W0(300, 26*26*kernel), B0(300, 1), W1(10, 300), B1(10, 1)
        {
            load(path);
        };

        ConvNN() : kernels(1, 3, 3), cB0(4, 26, 26), W0(300, 26*26), B0(300, 1), W1(10, 300), B1(10, 1)
        {
            kernels.randn(0, 1);
            cB0.randn(0, 1);
            W0.randn(0, 1);
            B0.randn(0, 1);
            W1.randn(0, 1);
            B1.randn(0, 1);
        };

        double evaluate(ImageList data_set, bool verbose);
        int predict(Image* i);

        void train(ImageList data_set, size_t epochs, double alpha, bool verbose = false, size_t batch_size = 100);

        void save(const char* path);
        void load(const char* path);

        Matrix run(Image* i);
        void run_visual(Image* i);
};

#endif