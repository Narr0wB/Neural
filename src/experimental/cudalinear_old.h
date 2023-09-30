
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

// WRAPPERS ----------------------------------------------------------------------------------------

void cudaw_mat_add(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE sum_host, size_t matRows, size_t matCols);

void cudaw_mat_subtract(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE sub_host, size_t matRows, size_t matCols);

void cudaw_mat_dot(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE dot_host, size_t dot_rows, size_t dot_cols, size_t commonDim);

void cudaw_mat_multiply(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE mul_host, size_t matRows, size_t matCols);

void cudaw_mat_scale(HPDOUBLE A_host, HPDOUBLE scal_host, double s, size_t matRows, size_t matCols);

void cudaw_mat_transpose(HPDOUBLE A_host, HPDOUBLE transp_host, size_t A_rows, size_t A_cols);

DMATRIX dMatAdd(DMATRIX A, DMATRIX B);

DMATRIX dMatSubtract(DMATRIX A, DMATRIX B);

DMATRIX dMatDot(DMATRIX A, DMATRIX B);

DMATRIX dMatMultiply(DMATRIX A, DMATRIX B);

DMATRIX dMatScale(DMATRIX A, double s);

DMATRIX dMatTranspose(DMATRIX A);

DMATRIX dMatApply(DMATRIX A, int op_idx);

DMATRIX dDenseForwardProp(DMATRIX Wn, DMATRIX An, DMATRIX Bn);

DMATRIX* dDenseBackProp(DMATRIX dY, DMATRIX Zm, DMATRIX An, int derivativeIdx);

// MISC --------------------------------------------------------------------------------------------

DMATRIX copyHostToDevice(HPDOUBLE A, size_t rows, size_t cols);

HPDOUBLE copyDeviceToHost(DMATRIX A);

DMATRIX dMatCreate(size_t rows, size_t cols);

void dMatFree(DMATRIX A);

#endif