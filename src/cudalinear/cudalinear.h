
#ifndef CUDALIN_H
#define CUDALIN_H

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "linalg.h"

#define ERR(msg) {std::cerr << msg << std::endl; exit(EXIT_FAILURE);}

#define KERNEL_BLOCK_SIZE 1024
#define KERNEL_GRID_SIZE (dim3(4, 1, 1))

#define NFUNCTIONS 3
#define SIGMOID 0
#define SIGMOID_DERIVATIVE 1
#define SQUARE 2

#define ix(row, col, N) (row * N) + col

typedef double** HPDOUBLE; // Host Pointer (DOUBLE)
typedef double* DPDOUBLE; // Device Pointer (DOUBLE)

struct DMATRIX {
    DPDOUBLE dataPtr;
    size_t rows;
    size_t cols;
};

struct DeviceImage {
    DPDOUBLE image;
    DPDOUBLE label;
};

class DeviceMatrix {
    private:
        DPDOUBLE m_Data;
        size_t m_rows;
        size_t m_cols;

        bool m_alloced = false;
        
    public:
        DeviceMatrix(Matrix m);
        DeviceMatrix(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_alloced(true) {
            if(cudaMalloc(&m_Data, (m_rows * m_cols) * sizeof(double)) != cudaSuccess) ERR("[ERROR] (DeviceMatrix) Could not allocate device memory!")
        };
        DeviceMatrix(DPDOUBLE data, size_t rows, size_t cols) : m_Data(data), m_rows(rows), m_cols(cols) {};

        DeviceMatrix() {};

        DeviceMatrix operator+(DeviceMatrix& other);
        DeviceMatrix operator-(DeviceMatrix& other);
        DeviceMatrix operator*(DeviceMatrix& other);
        DeviceMatrix operator*(double scalar);
        void operator=(DeviceMatrix other);

        DeviceMatrix transpose();
        void apply(DeviceMatrix& result, int op_idx);
        void scale(double scalar);
        void subtract(DeviceMatrix& other);

        ~DeviceMatrix() { 
            if (!m_alloced)
                return;

            if(cudaFree(m_Data) != cudaSuccess) 
                ERR("[ERROR] (~DeviceMatrix) Could not free the device memory!"); 
        };

        HPDOUBLE toHost();
        inline DPDOUBLE data() { return m_Data; }
        void set(Matrix m);
        inline void set(DPDOUBLE data, size_t rows, size_t cols) {
            m_Data = data;
            m_rows = rows;
            m_cols = cols;

            m_alloced = false;
        }

        inline size_t rows() const { return m_rows; }
        inline size_t cols() const { return m_cols; }

        friend DeviceMatrix hadamard(DeviceMatrix mat1, DeviceMatrix mat2);
};

__global__ void mat_add(DPDOUBLE A, DPDOUBLE B, DPDOUBLE sum, size_t matRows, size_t matCols);

__global__ void mat_subtract(DPDOUBLE A, DPDOUBLE B, DPDOUBLE subtracted, size_t matRows, size_t matCols);

__global__ void mat_dot(DPDOUBLE A, DPDOUBLE B, DPDOUBLE dot, size_t dot_rows, size_t dot_cols, size_t commonDim);

__global__ void mat_multiply(DPDOUBLE A, DPDOUBLE B, DPDOUBLE multiplied, size_t matRows, size_t matCols);

__global__ void mat_scale(DPDOUBLE A, DPDOUBLE scaled, double s, size_t matRows, size_t matCols);

__global__ void mat_transpose(DPDOUBLE A, DPDOUBLE transposed, size_t A_rows, size_t A_cols);

__global__ void mat_apply(DPDOUBLE A, DPDOUBLE applied, int activationIdx, size_t matRows, size_t matCols);

__global__ void dense_forwardprop(DPDOUBLE Wn, DPDOUBLE An, DPDOUBLE Bn, DPDOUBLE Zm, size_t Wrows, size_t Wcols);

__global__ void dense_backprop(DPDOUBLE dY, DPDOUBLE Zm, DPDOUBLE An, DPDOUBLE dBn, DPDOUBLE dWn, size_t Zmrows, size_t Anrows, int derivativeIdx);

// IN-DEVICE MATRIX OPERATIONS ---------------------------------------------------------------------

void denseForwardProp(DeviceMatrix& Zm, DeviceMatrix Wn, DeviceMatrix An, DeviceMatrix Bn);

void denseBackProp(DeviceMatrix& dWn, DeviceMatrix& dBn, DeviceMatrix dY, DeviceMatrix Zm, DeviceMatrix An, int op_idx);

#endif