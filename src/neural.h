
#ifndef NEURAL_H
#define NEURAL_H

#include <fstream>
#include <chrono>

#include "image.h"

typedef double** HPDOUBLE; // Host Pointer (DOUBLE)
typedef double* DPDOUBLE; // Device Pointer (DOUBLE)

//#define __GPU_ACCELERATION_CUDA

#ifdef __GPU_ACCELERATION_CUDA

#include "cudalinear/cudalinear.h"

inline double MSE(DeviceMatrix subtracted);

#endif

struct SimpleNeuralNetwork
{
    private:
        // HIDDEN LAYER -------------------------------
        Matrix W0; // A0 -> A1 Weights
        Matrix B0; // A0 -> A1 Biases

        // OUTPUT LAYER -------------------------------
        Matrix W1; // A1 -> A2 Weights
        Matrix B1; // A1 -> A2 Biases

#ifdef __GPU_ACCELERATION_CUDA
        std::vector<DeviceImage> allocBatch(std::vector<Image> data_set, size_t batch_number, size_t batch_size);
        void freeBatch(DeviceImage first);
#endif

    public:
        SimpleNeuralNetwork(std::vector<Image> data_set, size_t epochs, double learn_rate, bool verbose = false, size_t batch_size = 100) : W0(300, 784), B0(300, 1), W1(10, 300), B1(10, 1)
        {   
            W0.randn(0, 1);
            B0.randn(0, 1);
            W1.randn(0, 1);
            B1.randn(0, 1);
            train(data_set, epochs, learn_rate, verbose, batch_size);
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

        double evaluate(std::vector<Image> data_set, bool verbose = false, size_t batch_size = 100);
        int predict(Image i);

        void train(std::vector<Image> data_set, size_t epochs, double alpha, bool verbose = true, size_t batch_size = 100);
        
        void save(const char* path);
        void load(const char* path);

        Matrix run(Image i);
        void run_visual(Image i);
};

Matrix one_hot(int label);

// Activation functions 
double ReLU(double in);
double ReLU_d(double in);

double sigmoid(double in);
Matrix sigmoid_d(const Matrix& in);
double sigmoid_d(double in);

inline double tanh_d(double in);

inline double mish(double in);
inline double mish_d(double in);

inline double softplus(double in);
inline double softplus_d(double in);

inline double mean(Matrix in);

#endif
