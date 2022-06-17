
#include "cudalinear.h"
#include <cuda.h>

__device__ double square (double in)
{
    return pow(in, 2);
}

__device__ double dsigmoid (double in)
{
    return 1.0 / (1 + exp(-1 * in));
}

__device__ double dsigmoid_d (double in)
{
    return dsigmoid(in) * (1 - dsigmoid(in));
}

typedef double (*op_func) (double);

__device__ op_func funclist[NFUNCTIONS] = { dsigmoid, dsigmoid_d, square };


__global__ void mat_add(DPDOUBLE A, DPDOUBLE B, DPDOUBLE sum, size_t matRows, size_t matCols)
{
    size_t idx = threadIdx.x;
    size_t stride = blockDim.x;

    for (size_t i = idx; i < matRows; i += stride)
        for (size_t j = 0; j < matCols; j++)
            sum[ix(i, j, matCols)] = A[ix(i, j, matCols)] + B[ix(i, j, matCols)];
}

__global__ void mat_subtract(DPDOUBLE A, DPDOUBLE B, DPDOUBLE subtracted, size_t matRows, size_t matCols)
{
    size_t idx = threadIdx.x;
    size_t stride = blockDim.x;

    for (size_t i = idx; i < matRows; i += stride)
        for (size_t j = 0; j < matCols; j++)
            subtracted[ix(i, j, matCols)] = A[ix(i, j, matCols)] - B[ix(i, j, matCols)];
}

__global__ void mat_dot(DPDOUBLE A, DPDOUBLE B, DPDOUBLE dot, size_t dotRows, size_t dotCols, size_t commonDim)
{
    size_t idx = threadIdx.x;
    size_t stride = blockDim.x;

    for (size_t i = idx; i < dotRows; i += stride)
        for (size_t j = 0; j < dotCols; j++)
        {
            double sum = 0;
            for (size_t k = 0; k < commonDim; k++)
                sum += A[ix(i, k, commonDim)] * B[ix(k, j, dotCols)];
            dot[ix(i, j, dotCols)] = sum;
        }
}

__global__ void mat_multiply(DPDOUBLE A, DPDOUBLE B, DPDOUBLE multiplied, size_t matRows, size_t matCols)
{
    size_t idx = threadIdx.x;
    size_t stride = blockDim.x;

    for (size_t i = idx; i < matRows; i += stride)
        for (size_t j = 0; j < matCols; j++)
            multiplied[ix(i, j, matCols)] = A[ix(i, j, matCols)] * B[ix(i, j, matCols)];
}

__global__ void mat_scale(DPDOUBLE A, DPDOUBLE scaled, double s, size_t matRows, size_t matCols)
{
    size_t idx = threadIdx.x;
    size_t stride = blockDim.x;

    for (size_t i = idx; i < matRows; i += stride)
        for (size_t j = 0; j < matCols; j++)
            scaled[ix(i, j, matCols)] = A[ix(i, j, matCols)] * s;
}

__global__ void mat_transpose(DPDOUBLE A, DPDOUBLE transposed, size_t A_rows, size_t A_cols)
{
    size_t idx = threadIdx.x;
    size_t stride = blockDim.x;

    for (size_t i = idx; i < A_cols; i += stride)
        for (size_t j = 0; j < A_rows; j++)
            transposed[ix(i, j, A_rows)] = A[ix(j, i, A_cols)];
}

__global__ void mat_apply(DPDOUBLE A, DPDOUBLE applied, int op_idx, size_t matRows, size_t matCols)
{
    size_t idx = threadIdx.x;
    size_t stride = blockDim.x;

    for (size_t i = idx; i < matRows; i += stride)
        for (size_t j = 0; j < matCols; j++)
            applied[ix(i, j, matCols)] = funclist[op_idx](A[ix(i, j, matCols)]);
}

// IN-DEVICE MATRIX OPERATIONS ---------------------------------------------------------------------

DMATRIX dMatAdd(DMATRIX A, DMATRIX B)
{
    if (A.rows != B.rows || 
        A.cols != B.cols)
        throw "INVALID MATRICES";

    DMATRIX sum;
    sum.rows = A.rows;
    sum.cols = A.cols;
    cudaMalloc(&sum.dataPtr, (sum.rows * sum.cols) * sizeof(double));

    mat_add<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A.dataPtr, B.dataPtr, sum.dataPtr, sum.rows, sum.cols);

    //cudaDeviceSynchronize();

    return sum;
}

DMATRIX dMatSubtract(DMATRIX A, DMATRIX B)
{
    if (A.rows != B.rows || 
        A.cols != B.cols)
        throw "INVALID MATRICES";
    
    DMATRIX diff;
    diff.rows = A.rows;
    diff.cols = A.cols;
    cudaMalloc(&diff.dataPtr, (diff.rows * diff.cols) * sizeof(double));

    mat_subtract<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A.dataPtr, B.dataPtr, diff.dataPtr, diff.rows, diff.cols);

    //cudaDeviceSynchronize();

    return diff;
}

DMATRIX dMatDot(DMATRIX A, DMATRIX B)
{
    if (A.cols != B.rows)
        throw "INVALID MATRICES";

    DMATRIX dot;
    dot.rows = A.rows;
    dot.cols = B.cols;
    cudaMalloc(&dot.dataPtr, (dot.rows * dot.cols) * sizeof(double));

    mat_dot<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A.dataPtr, B.dataPtr, dot.dataPtr, dot.rows, dot.cols, A.cols);

    //cudaDeviceSynchronize();

    return dot;
}

DMATRIX dMatMultiply(DMATRIX A, DMATRIX B)
{
    if (A.rows != B.rows || 
        A.cols != B.cols)
        throw "INVALID MATRICES";
    
    DMATRIX mul;
    mul.rows = A.rows;
    mul.cols = A.cols;
    cudaMalloc(&mul.dataPtr, (mul.rows * mul.cols) * sizeof(double));

    mat_multiply<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A.dataPtr, B.dataPtr, mul.dataPtr, mul.rows, mul.cols);

    //cudaDeviceSynchronize();

    return mul;
}

DMATRIX dMatScale(DMATRIX A, double s)
{
    if (A.rows == 0 ||
        A.cols == 0)
        throw "INVALID MATRIX";
    
    DMATRIX scal;
    scal.rows = A.rows;
    scal.cols = A.cols;
    cudaMalloc(&scal.dataPtr, (scal.rows * scal.cols) * sizeof(double));

    mat_scale<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A.dataPtr, scal.dataPtr, s, scal.rows, scal.cols);

    //cudaDeviceSynchronize();

    return scal;
}

DMATRIX dMatTranspose(DMATRIX A)
{
    if (A.rows == 0 ||
        A.cols == 0)
        throw "INVALID MATRIX";
    
    DMATRIX transp;
    transp.rows = A.cols;
    transp.cols = A.rows;
    cudaMalloc(&transp.dataPtr, (transp.rows * transp.cols) * sizeof(double));

    mat_transpose<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A.dataPtr, transp.dataPtr, A.rows, A.cols);

    //cudaDeviceSynchronize();

    return transp;
}

DMATRIX dMatApply(DMATRIX A, int op_idx)
{
    if (A.rows == 0 ||
        A.cols == 0)
        throw "INVALID MATRIX";

    if (op_idx > NFUNCTIONS)
        throw "INVALID FUNCTION INDEX";

    DMATRIX app;
    app.rows = A.rows;
    app.cols = A.cols;
    cudaMalloc(&app.dataPtr, (app.rows * app.cols) * sizeof(double));

    mat_apply<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A.dataPtr, app.dataPtr, op_idx, app.rows, app.cols);

    //cudaDeviceSynchronize();

    return app;
}

// WRAPPERS ----------------------------------------------------------------------------------------

// NOTE: The caller has to make sure that the matrices dimensions match before calling the wrapper functions, doing otherwise will lead to undefined behaviour 
void cudaw_mat_add(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE sum_host, size_t matRows, size_t matCols)
{
    if (A_host == nullptr ||
        B_host == nullptr ||
        sum_host == nullptr)
        return;

    DPDOUBLE A_dev;
    DPDOUBLE B_dev;
    DPDOUBLE sum_dev;

    cudaMalloc(&A_dev, (matRows * matCols)*sizeof(double));
    cudaMalloc(&B_dev, (matRows * matCols)*sizeof(double));
    cudaMalloc(&sum_dev, (matRows * matCols)*sizeof(double));

    for (int i = 0; i < matRows; i++)
    {
        cudaMemcpy(A_dev + (matCols * i), A_host[i], matCols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(B_dev + (matCols * i), B_host[i], matCols * sizeof(double), cudaMemcpyHostToDevice);
    }
    
    mat_add<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A_dev, B_dev, sum_dev, matRows, matCols);

    cudaDeviceSynchronize();

    for (int i = 0; i < matRows; i++)
        cudaMemcpy(sum_host[i], sum_dev + (matCols * i), matCols * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(sum_dev);
}

void cudaw_mat_subtract(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE sub_host, size_t matRows, size_t matCols)
{
    if (A_host == nullptr ||
        B_host == nullptr ||
        sub_host == nullptr)
        return;

    DPDOUBLE A_dev;
    DPDOUBLE B_dev;
    DPDOUBLE sub_dev;

    cudaMalloc(&A_dev, (matRows * matCols)*sizeof(double));
    cudaMalloc(&B_dev, (matRows * matCols)*sizeof(double));
    cudaMalloc(&sub_dev, (matRows * matCols)*sizeof(double));

    for (int i = 0; i < matRows; i++)
    {
        cudaMemcpy(A_dev + (matCols * i), A_host[i], matCols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(B_dev + (matCols * i), B_host[i], matCols * sizeof(double), cudaMemcpyHostToDevice);
    }

    mat_subtract<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A_dev, B_dev, sub_dev, matRows, matCols);

    cudaDeviceSynchronize();

    for (int i = 0; i < matRows; i++)
        cudaMemcpy(sub_host[i], sub_dev + (matCols * i), matCols * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(sub_dev);
}

void cudaw_mat_dot(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE dot_host, size_t dot_rows, size_t dot_cols, size_t commonDim)
{
    if (A_host == nullptr ||
        B_host == nullptr ||
        dot_host == nullptr)
        return;
    
    DPDOUBLE A_dev;
    DPDOUBLE B_dev;
    DPDOUBLE dot_dev;

    cudaMalloc(&A_dev, (dot_rows * commonDim)*sizeof(double));
    cudaMalloc(&B_dev, (commonDim * dot_cols)*sizeof(double));
    cudaMalloc(&dot_dev, (dot_rows * dot_cols)*sizeof(double));

    for (int i = 0; i < dot_rows; i++)
        cudaMemcpy(A_dev + (commonDim * i), A_host[i], commonDim * sizeof(double), cudaMemcpyHostToDevice);
    
    for (int i = 0; i < commonDim; i++)
        cudaMemcpy(B_dev + (dot_cols * i), B_host[i], dot_cols * sizeof(double), cudaMemcpyHostToDevice);

    mat_dot<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A_dev, B_dev, dot_dev, dot_rows, dot_cols, commonDim);

    cudaDeviceSynchronize();

    for (int i = 0; i < dot_rows; i++)
        cudaMemcpy(dot_host[i], dot_dev + (dot_cols * i), dot_cols * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(dot_dev);
}

void cudaw_mat_multiply(HPDOUBLE A_host, HPDOUBLE B_host, HPDOUBLE mul_host, size_t matRows, size_t matCols)
{
    if (A_host == nullptr ||
        B_host == nullptr ||
        mul_host == nullptr)
        return;

    DPDOUBLE A_dev;
    DPDOUBLE B_dev;
    DPDOUBLE mul_dev;

    cudaMalloc(&A_dev, (matRows * matCols)*sizeof(double));
    cudaMalloc(&B_dev, (matRows * matCols)*sizeof(double));
    cudaMalloc(&mul_dev, (matRows * matCols)*sizeof(double));

    for (int i = 0; i < matRows; i++)
    {
        cudaMemcpy(A_dev + (matCols * i), A_host[i], matCols * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(B_dev + (matCols * i), B_host[i], matCols * sizeof(double), cudaMemcpyHostToDevice);
    }
    
    mat_multiply<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A_dev, B_dev, mul_dev, matRows, matCols);

    cudaDeviceSynchronize();

    for (int i = 0; i < matRows; i++)
        cudaMemcpy(mul_host[i], mul_dev + (matCols * i), matCols * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(A_dev);
    cudaFree(B_dev);
    cudaFree(mul_dev);
}

void cudaw_mat_scale(HPDOUBLE A_host, HPDOUBLE scal_host, double s, size_t matRows, size_t matCols)
{
    if (A_host == nullptr ||
        scal_host == nullptr)
        return;

    DPDOUBLE A_dev;
    DPDOUBLE scal_dev;

    cudaMalloc(&A_dev, (matRows * matCols)*sizeof(double));
    cudaMalloc(&scal_dev, (matRows * matCols)*sizeof(double));

    for (int i = 0; i < matRows; i++)
        cudaMemcpy(A_dev + (matCols * i), A_host[i], matCols * sizeof(double), cudaMemcpyHostToDevice);

    mat_scale<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A_dev, scal_dev, s, matRows, matCols);

    cudaDeviceSynchronize();

    for (int i = 0; i < matRows; i++)
        cudaMemcpy(scal_host[i], scal_dev + (matCols * i), matCols * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(A_dev);
    cudaFree(scal_dev);
}

void cudaw_mat_transpose(HPDOUBLE A_host, HPDOUBLE transp_host, size_t A_rows, size_t A_cols)
{
    if (A_host == nullptr ||
        transp_host == nullptr)
        return;

    DPDOUBLE A_dev;
    DPDOUBLE transp_dev;

    cudaMalloc(&A_dev, (A_rows * A_cols)*sizeof(double));
    cudaMalloc(&transp_dev, (A_rows * A_cols)*sizeof(double));

    for (int i = 0; i < A_rows; i++)
        cudaMemcpy(A_dev + (A_cols * i), A_host[i], A_cols * sizeof(double), cudaMemcpyHostToDevice);

    mat_transpose<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(A_dev, transp_dev, A_rows, A_cols);

    cudaDeviceSynchronize();

    for (int i = 0; i < A_cols; i++)
        cudaMemcpy(transp_host[i], transp_dev + (A_rows * i), A_rows * sizeof(double), cudaMemcpyDeviceToHost);
    
    cudaFree(A_dev);
    cudaFree(transp_dev);
}

// MISC --------------------------------------------------------------------------------------------

DMATRIX copyHostToDevice(HPDOUBLE A, size_t rows, size_t cols)
{   
    DMATRIX d_mat;
    DPDOUBLE d_A;

    auto result = cudaMalloc(&d_A, (rows * cols) * sizeof(double));

    if (result != cudaSuccess)
        throw "Memory allocation error";

    for (int i = 0; i < rows; i++)
    {
        result = cudaMemcpy(d_A + (cols * i), A[i], cols * sizeof(double), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
            throw "Memory copy error";
    }
    
    d_mat.dataPtr = d_A;
    d_mat.rows = rows;
    d_mat.cols = cols;

    return d_mat;
}

HPDOUBLE copyDeviceToHost(DMATRIX A)
{
    HPDOUBLE data = new double*[A.rows];
    for (int i = 0; i < A.rows; i++)
    {
        data[i] = new double[A.cols];
        auto result = cudaMemcpy(data[i], A.dataPtr + (A.cols * i), A.cols * sizeof(double), cudaMemcpyDeviceToHost);
        if (result != cudaSuccess)
            throw "Memory copy error";
    }

    return data;
}

DMATRIX dMatCreate(size_t rows, size_t cols)
{
    DMATRIX mat;
    mat.rows = rows;
    mat.cols = cols;

    auto result = cudaMalloc(&mat.dataPtr, (mat.rows * mat.cols) * sizeof(double));

    if (result != cudaSuccess)
        throw "Memory allocation error";

    return mat;
}

void dMatFree(DMATRIX A)
{
    auto result = cudaFree(A.dataPtr);

    if (result != cudaSuccess)
        throw "Memory free error";
}