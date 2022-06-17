
#ifndef NEURAL_H
#define NEURAL_H

#include <fstream>
#include <chrono>

#include "image.h"

#define __GPU_ACCELERATION_CUDA

typedef double** HPDOUBLE; // Host pointer (double)
typedef double* DPDOUBLE; // Device pointer (double)

struct DMATRIX {
    DPDOUBLE dataPtr;
    size_t rows;
    size_t cols;
};

extern DMATRIX dMatAdd(DMATRIX A, DMATRIX B);
extern DMATRIX dMatSubtract(DMATRIX A, DMATRIX B);
extern DMATRIX dMatDot(DMATRIX A, DMATRIX B);
extern DMATRIX dMatMultiply(DMATRIX A, DMATRIX B);
extern DMATRIX dMatScale(DMATRIX A, double s);
extern DMATRIX dMatTranspose(DMATRIX A);
extern DMATRIX dMatApply(DMATRIX A, int op_idx);
extern DMATRIX dMatCreate(size_t rows, size_t cols);
extern void dMatFree(DMATRIX A);
extern HPDOUBLE copyDeviceToHost(DMATRIX A);
extern DMATRIX copyHostToDevice(HPDOUBLE A, size_t rows, size_t cols);

struct SimpleNeuralNetwork
{
    private:
        // HIDDEN LAYER -------------------------------
        Matrix W0; // A0 -> A1 Weights
        Matrix B0; // A0 -> A1 Biases

        // OUTPUT LAYER -------------------------------
        Matrix W1; // A1 -> A2 Weights
        Matrix B1; // A1 -> A2 Biases


    public:
        SimpleNeuralNetwork(ImageList data_set, size_t epochs, size_t batch_size, double alpha, bool verbose = false) : W0(300, 28*28), B0(300, 1), W1(10, 300), B1(10, 1)
        {   
            W0.randn(0, 1);
            B0.randn(0, 1);
            W1.randn(0, 1);
            B1.randn(0, 1);
            train(data_set, epochs, batch_size, alpha, verbose);
        };

        SimpleNeuralNetwork(const char* path) : W0(300, 28*28), B0(300, 1), W1(10, 300), B1(10, 1)
        {
            load(path);
        };

        SimpleNeuralNetwork() : W0(300, 28*28), B0(300, 1), W1(10, 300), B1(10, 1)
        {
            W0.randn(0, 1);
            B0.randn(0, 1);
            W1.randn(0, 1);
            B1.randn(0, 1);
        };

        double evaluate(ImageList data_set, bool verbose);

        void train(ImageList data_set, size_t epochs, size_t batch_size, double alpha, bool verbose = false);

        void save(const char* path);
        void load(const char* path);

        Matrix run(Image* i);
        void run_visual(Image* i);
};

Matrix one_hot(int label);

// Activation functions 
inline double ReLU(double in);
double ReLU_d(double in);

double sigmoid(double in);
Matrix sigmoid_d(const Matrix& in);
double sigmoid_d(double in);

double tanh_d(double in);

double mish(double in);
double mish_d(double in);

double softplus(double in);
double softplus_d(double in);

double MSE(Matrix A, DMATRIX B);

double mean(Matrix in);

#endif
