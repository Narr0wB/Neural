
#ifndef CUDALIN_H
#define CUDALIN_H

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include "linalg.h"

#define ERR(msg) {std::cerr << msg << std::endl; exit(EXIT_FAILURE);}

float clamp(float x, float max, float min);

#define NFUNCTIONS 3
#define SIGMOID 0
#define SIGMOID_DERIVATIVE 1
#define SQUARE 2

#define ix(row, col, N) (row * N) + col

typedef double** HPDOUBLE; // Host Pointer (DOUBLE)
typedef double* DPDOUBLE; // Device Pointer (DOUBLE)

struct DMATRIX {
    DPDOUBLE data;
    size_t rows;
    size_t cols;
};

struct DIMAGE {
    DMATRIX image;
    DMATRIX label;
};

struct dim4 {
    size_t x;
    size_t y;
    size_t z;
    size_t w;
};

dim4 _kernel_block_size(size_t rows, size_t cols);

#define KERNEL_BLOCK_SIZE(rows, c) _kernel_block_size(rows, c)  
#define KERNEL_GRID_SIZE 1

// IN-DEVICE MATRIX OPERATIONS ---------------------------------------------------------------------

__global__ void mat_add(DPDOUBLE A, DPDOUBLE B, DPDOUBLE sum, size_t matRows, size_t matCols, size_t mask_rows, size_t mask_cols);

__global__ void mat_subtract(DPDOUBLE A, DPDOUBLE B, DPDOUBLE subtracted, size_t matRows, size_t matCols, size_t mask_rows, size_t mask_cols);

__global__ void mat_dot(DPDOUBLE A, DPDOUBLE B, DPDOUBLE dot, size_t dot_rows, size_t dot_cols, size_t commonDim);

__global__ void mat_multiply(DPDOUBLE A, DPDOUBLE B, DPDOUBLE multiplied, size_t matRows, size_t matCols, size_t mask_rows, size_t mask_cols);

__global__ void mat_scale(DPDOUBLE A, DPDOUBLE scaled, double s, size_t matRows, size_t matCols, size_t mask_rows, size_t mask_cols);

__global__ void mat_transpose(DPDOUBLE A, DPDOUBLE transposed, size_t A_rows, size_t A_cols, size_t mask_rows, size_t mask_cols);

__global__ void mat_apply(DPDOUBLE A, DPDOUBLE applied, int activationIdx, size_t matRows, size_t matCols, size_t mask_rows, size_t mask_cols);

__global__ void dense_forwardprop(DPDOUBLE Wn, DPDOUBLE An, DPDOUBLE Bn, DPDOUBLE Zm, size_t Wrows, size_t Wcols);

__global__ void dense_backprop(DPDOUBLE dY, DPDOUBLE Zm, DPDOUBLE An, DPDOUBLE dBn, DPDOUBLE dWn, size_t Zmrows, size_t Anrows, int derivativeIdx);

// HOST WRAPPERS ------------------------------------------------------------------------------------

void denseForwardProp(DMATRIX Zm, DMATRIX Wn, DMATRIX An, DMATRIX Bn);

void denseBackProp(DMATRIX dWn, DMATRIX dBn, DMATRIX dY, DMATRIX Zm, DMATRIX An, int op_idx);

void DMatrixAdd(DMATRIX A, DMATRIX B, DMATRIX result);

void DMatrixSubtract(DMATRIX A, DMATRIX B, DMATRIX result);

void DMatrixDot(DMATRIX A, DMATRIX B, DMATRIX result);

void DMatrixHadamard(DMATRIX A, DMATRIX B, DMATRIX result);

void DMatrixScale(DMATRIX A, DMATRIX result, double scalar);

void DMatrixTranspose(DMATRIX A, DMATRIX result);

void DMatrixApply(DMATRIX A, DMATRIX result, int op_code);

DMATRIX DMatrixCreate(size_t rows, size_t cols);
void DMatrixFree(DMATRIX mat);

// ATTENTION! THESE FUNCTIONS ALLOCATE MEMORY THAT HAS TO BE FREED MANUALLY!
DMATRIX copyHostToDevice(HPDOUBLE host_data, size_t rows, size_t cols);
HPDOUBLE copyDeviceToHost(DMATRIX device_data);

void __hostToDeviceData(HPDOUBLE host_data, DPDOUBLE device_data, size_t rows, size_t cols);

#endif