
#ifndef CUDALIN_H
#define CUDALIN_H

#include <math.h>
#include <stdio.h>
#include <cuda.h>
#include <iostream>

#define KERNEL_BLOCK_SIZE 1024
#define KERNEL_GRID_SIZE (dim3(4, 1, 1))

#define NFUNCTIONS 3

#define ix(row, col, N) (row * N) + col

typedef double** HPDOUBLE; // Host Pointer (DOUBLE)
typedef double* DPDOUBLE; // Device Pointer (DOUBLE)

struct DMATRIX {
    DPDOUBLE dataPtr;
    size_t rows;
    size_t cols;
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

DMATRIX dMatAdd(DMATRIX A, DMATRIX B);

DMATRIX dMatSubtract(DMATRIX A, DMATRIX B);

DMATRIX dMatDot(DMATRIX A, DMATRIX B);

DMATRIX dMatMultiply(DMATRIX A, DMATRIX B);

DMATRIX dMatScale(DMATRIX A, double s);

DMATRIX dMatTranspose(DMATRIX A);

DMATRIX dMatApply(DMATRIX A, int op_idx);

DMATRIX dDenseForwardProp(DMATRIX Wn, DMATRIX An, DMATRIX Bn);

DMATRIX* dDenseBackProp(DMATRIX dY, DMATRIX Zm, DMATRIX An, int derivativeIdx);

// WRAPPERS ----------------------------------------------------------------------------------------

void cudaw_mat_add(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE sum_host, size_t matRows, size_t matCols);

void cudaw_mat_subtract(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE sub_host, size_t matRows, size_t matCols);

void cudaw_mat_dot(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE dot_host, size_t dot_rows, size_t dot_cols, size_t commonDim);

void cudaw_mat_multiply(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE mul_host, size_t matRows, size_t matCols);

void cudaw_mat_scale(HPDOUBLE A_host, HPDOUBLE scal_host, double s, size_t matRows, size_t matCols);

void cudaw_mat_transpose(HPDOUBLE A_host, HPDOUBLE transp_host, size_t A_rows, size_t A_cols);

// MISC --------------------------------------------------------------------------------------------

DMATRIX copyHostToDevice(HPDOUBLE A, size_t rows, size_t cols);

HPDOUBLE copyDeviceToHost(DMATRIX A);

DMATRIX dMatCreate(size_t rows, size_t cols);

void dMatFree(DMATRIX A);

#endif