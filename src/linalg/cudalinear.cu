
#include "cudalinear.h"
#include <cuda.h>

#define MAX_THREADS 16384
#define MAX_THREADS_PER_BLOCK 32

dim4 _kernel_block_size(size_t rows, size_t cols) {
    size_t mat_rows = 0;
    size_t mat_cols = 0;
    size_t n_blocks = 0;
    size_t threads_per_block = MAX_THREADS_PER_BLOCK;

    if (rows * cols < MAX_THREADS) {

        mat_rows = rows > MAX_THREADS_PER_BLOCK ? rows - (rows % MAX_THREADS_PER_BLOCK) : rows;
        mat_cols = cols > MAX_THREADS_PER_BLOCK ? cols - (cols % MAX_THREADS_PER_BLOCK) : cols;

        if (mat_rows * mat_cols < MAX_THREADS_PER_BLOCK) {
            n_blocks = 1;
            threads_per_block = mat_rows*mat_cols;
        }
        else {
            n_blocks = (mat_rows * mat_cols) / MAX_THREADS_PER_BLOCK;
        }
    }
    else {
        float ratio = clamp((float) rows / cols, 128, 1/128);

        size_t temp_rows = (ratio * 128);
        size_t temp_cols = (1.0f / ratio) * 128;

        mat_rows = temp_rows > MAX_THREADS_PER_BLOCK ? temp_rows - ((size_t)temp_rows % MAX_THREADS_PER_BLOCK) : temp_rows;
        mat_cols = temp_cols > MAX_THREADS_PER_BLOCK ? temp_cols - ((size_t)temp_cols % MAX_THREADS_PER_BLOCK) : temp_cols;

        n_blocks = (mat_rows * mat_cols) / MAX_THREADS_PER_BLOCK;
    }

    return dim4{n_blocks, mat_rows, mat_cols, threads_per_block};
}

float clamp(float x, float max, float min) {
    if (x > max) {
        return max;
    }

    if (x < min) {
        return min;
    }

    return x;
}

__device__ double square(double in)
{
    return pow(in, 2);
}

__device__ double dsigmoid(double in)
{
    return 1.0 / (1 + exp(-1 * in));
}

__device__ double dsigmoid_d(double in)
{
    return dsigmoid(in) * (1 - dsigmoid(in));
}

typedef double (*op_func) (double);

__device__ op_func funclist[NFUNCTIONS] = { dsigmoid, dsigmoid_d, square };

/*
                                                                ########## MASK MATRIX ##########
    The mask matrix is a matrix of threads which is overlaid over one or more matrices similar to a 2D Convolution or Cross-Correlation and whose 
    dimentions are equal (if possible) or proportional to the dimetions of the original matrix, allowing for faster computation of same-shaped or 
    single matrix operations. The dimentions of the thread matrix (or mask matrix) are calculated as follows:

    Given a matrix m x n, we first have to see if the number of elements in that matrix is less than the number of available threads.

    If said condition is true, then:

    1.1) Look for the multiple of 32 that is closest and less than m, that will be the mask matrix's x dimention (the mask matrix's "m" dimention)
    1.2) Look for the multiple of 32 that is closest and less than n, same as with m, and that will be the mask matrix's y dimention (the mask matrix's "n" dimention)

    else, if the number of elements exceeds the number of avilable threads, then:

    2.1) Take the ratio of the matrix that we are trying to perform calucations on, i.e. m/n, and we clamp said ratio to be in the range of the square root 
    of the number of available threads and one over that same square root ( clamp(ratio, 1/sqrt(MAX_THREADS), sqrt(MAX_THREADS)) ).

    2.2) Calculate the temporary x (temp_x) dimention using the formula: sqrt(MAX_THREADS) * ratio
    2.3) Calculate the temporary y (temp_y) dimention using the formula: sqrt(MAX_THREADS) * (1.0f / ratio)

    2.4) As in step 1.1, find the multiple of 32 that is closest and less than temp_x and that will be the mask matrix's x dimention
    2.4) As in step 1.2, find the multiple of 32 that is closest and less than temp_y and that will be the mask matrix's y dimention

    (NOTE 1: If m or n is less than 32 then we take the dimention as it is)
    (NOTE 2: We have chosen dimentions that are multiple of 32 because cuda threads run in warps, meaning threads are in blocks of 32 that execute the same instructions)
*/


__global__ void mat_add(DPDOUBLE A, DPDOUBLE B, DPDOUBLE sum, size_t a_rows, size_t a_cols, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;
    
    for (size_t i = row_index; i < a_rows; i += mask_rows) {
        for (size_t j = column_index; j < a_cols; j += mask_cols) {
            sum[ix(i, j, a_cols)] = A[ix(i, j, a_cols)] + B[ix(i, j, a_cols)];
        }
    }
}

__global__ void mat_subtract(DPDOUBLE A, DPDOUBLE B, DPDOUBLE subtracted, size_t a_rows, size_t a_cols, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;
    
    for (size_t i = row_index; i < a_rows; i += mask_rows)
        for (size_t j = column_index; j < a_cols; j += mask_cols)
            subtracted[ix(i, j, a_cols)] = A[ix(i, j, a_cols)] - B[ix(i, j, a_cols)];
}

__global__ void mat_dot(DPDOUBLE A, DPDOUBLE B, DPDOUBLE dot, size_t dotRows, size_t dotCols, size_t commonDim)
{
    size_t row_index = threadIdx.x;
    size_t column_index = 0;
    size_t stride_x = blockDim.x;
    size_t stride_y = 1;

    if (threadIdx.y != 0) return;

    for (size_t i = row_index; i < dotRows; i += stride_x) {
        for (size_t j = 0; j < dotCols; j++)
        {
            double sum = 0;

            for (size_t k = column_index; k < commonDim; k += stride_y) {
                sum += A[ix(i, k, commonDim)] * B[ix(k, j, dotCols)];
            } 

            dot[ix(i, j, dotCols)] = sum; 
        }
    }
}

__global__ void mat_multiply(DPDOUBLE A, DPDOUBLE B, DPDOUBLE multiplied, size_t a_rows, size_t a_cols, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;
    
    for (size_t i = row_index; i < a_rows; i += mask_rows) {
        for (size_t j = column_index; j < a_cols; j += mask_cols) {
            multiplied[ix(i, j, a_cols)] = A[ix(i, j, a_cols)] * B[ix(i, j, a_cols)];
        }
    }
}

__global__ void mat_scale(DPDOUBLE A, DPDOUBLE scaled, double s, size_t a_rows, size_t a_cols, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;
    
    for (size_t i = row_index; i < a_rows; i += mask_rows)
        for (size_t j = column_index; j < a_cols; j += mask_cols)
            scaled[ix(i, j, a_cols)] = A[ix(i, j, a_cols)] * s;
}

__global__ void mat_transpose(DPDOUBLE A, DPDOUBLE transposed, size_t A_rows, size_t A_cols, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;

    for (size_t i = column_index; i < A_cols; i += mask_cols)
        for (size_t j = row_index; j < A_rows; j += mask_rows)
            transposed[ix(i, j, A_rows)] = A[ix(j, i, A_cols)];
}

__global__ void mat_apply(DPDOUBLE A, DPDOUBLE applied, int activationIdx, size_t a_rows, size_t a_cols, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;

    for (size_t i = row_index; i < a_rows; i += mask_rows)
        for (size_t j = column_index; j < a_cols; j += mask_cols)
            applied[ix(i, j, a_cols)] = funclist[activationIdx](A[ix(i, j, a_cols)]);
}

__global__ void dense_forwardprop(DPDOUBLE Wn, DPDOUBLE An, DPDOUBLE Bn, DPDOUBLE Zm, size_t Wrows, size_t Wcols, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;

    for (size_t i = row_index; i < Wrows; i += mask_rows)
    {
        for (size_t k = column_index; k < Wcols; k += mask_cols) {
            atomicAdd(&Zm[ix(i, 0, 1)], (Wn[ix(i, k, Wcols)] * An[ix(k, 0, 1)]));
        }
        
        if (column_index == 0) {
            Zm[ix(i, 0, 1)] += Bn[ix(i, 0, 1)];
        }
    }
}

__global__ void dense_backprop(DPDOUBLE dY, DPDOUBLE Zm, DPDOUBLE An, DPDOUBLE dBn, DPDOUBLE dWn, size_t Zmrows, size_t Anrows, int derivativeIdx, size_t mask_rows, size_t mask_cols)
{
    size_t row_index = (blockIdx.x * blockDim.x + threadIdx.x) / mask_cols;
    size_t column_index = (blockIdx.x * blockDim.x + threadIdx.x) % mask_cols;

    for (size_t i = row_index; i < Zmrows; i += mask_rows)
    {   
        double product = funclist[derivativeIdx](Zm[ix(i, 0, 1)]) * dY[ix(i, 0, 1)];

        if (column_index == 0) {
            dBn[ix(i, 0, 1)] = product;
        }

        for (size_t j = column_index; j < Anrows; j += mask_cols) {
            dWn[ix(i, j, Anrows)] = product * An[ix(j, 0, 1)];
        }
    }
}

// IN-DEVICE MATRIX OPERATIONS ---------------------------------------------------------------------

void denseForwardProp(DMATRIX Zm, DMATRIX Wn, DMATRIX An, DMATRIX Bn) {
    if (An.cols != 1 ||
        Bn.cols != 1 ||
        An.rows != Wn.cols ||
        Bn.rows != Wn.rows)
        ERR("[ERROR] (denseForwardProp) Invlid matrices!")

    if (Zm.rows != Wn.rows || Zm.cols != 1)
        ERR("[ERROR] (denseForwardProp) Invlid matrices!")

    dim4 dimentions = KERNEL_BLOCK_SIZE(Wn.rows, Wn.cols);

    dense_forwardprop<<<dimentions.x, dimentions.w>>>(Wn.data, An.data, Bn.data, Zm.data, Wn.rows, Wn.cols, dimentions.y, dimentions.z);
}

void denseBackProp(DMATRIX dWn, DMATRIX dBn, DMATRIX dY, DMATRIX Zm, DMATRIX An, int op_idx) {
    if (An.cols != 1 ||
        Zm.cols != 1 ||
        dY.cols != 1 ||
        Zm.rows != dY.rows)
        ERR("[ERROR] (denseBackProp) Invalid matrices!")

    if (dWn.rows != Zm.rows ||
        dWn.cols != An.rows ||
        dBn.rows != Zm.rows ||
        dBn.cols != 1)
        ERR("[ERROR] (denseBackProp) Invalid matrices!");

    dim4 dimentions = KERNEL_BLOCK_SIZE(dWn.rows, dWn.cols);

    dense_backprop<<<dimentions.x, dimentions.w>>>(dY.data, Zm.data, An.data, dBn.data, dWn.data, Zm.rows, An.rows, op_idx, dimentions.y, dimentions.z);
}

void DMatrixAdd(DMATRIX A, DMATRIX B, DMATRIX result) {
    if (A.rows != B.rows || 
        A.cols != B.cols ||
        result.rows != A.rows ||
        result.cols != A.cols)
        ERR("[ERROR] (DMatrixAdd) Invalid matrices for the \"+\" operation!");

    dim4 dimentions = KERNEL_BLOCK_SIZE(A.rows, A.cols);

    mat_add<<<dimentions.x, dimentions.w>>>(A.data, B.data, result.data, A.rows, A.cols, dimentions.y, dimentions.z);
}

void DMatrixSubtract(DMATRIX A, DMATRIX B, DMATRIX result) {
    if (A.rows!= B.rows || 
        A.cols != B.cols ||
        result.rows != A.rows ||
        result.cols != A.cols)
        ERR("[ERROR] (DMatrixSubtract) Invalid matrices for the \"-\" operation!");

    dim4 dimentions = KERNEL_BLOCK_SIZE(A.rows, A.cols);

    mat_subtract<<<dimentions.x, dimentions.w>>>(A.data, B.data, result.data, A.rows, A.cols, dimentions.y, dimentions.z);
}

void DMatrixDot(DMATRIX A, DMATRIX B, DMATRIX result) {
    if (A.cols != B.rows ||
        result.rows != A.rows ||
        result.cols != B.cols)
        ERR("[ERROR] (DMatrixDot) Invalid matrices for the \"*\" operation!");

    mat_dot<<<KERNEL_GRID_SIZE, dim3(1024, 1, 1)>>>(A.data, B.data, result.data, A.rows, B.cols, A.cols);
}

void DMatrixHadamard(DMATRIX A, DMATRIX B, DMATRIX result) {
    if (A.rows != B.rows || 
        A.cols != B.cols ||
        result.rows != A.rows ||
        result.cols != A.cols)
        ERR("[ERROR] (DMatrixHadamard) Invalid matrices for the \"hadamard\" operation!");

    dim4 dimentions = KERNEL_BLOCK_SIZE(A.rows, A.cols);

    mat_multiply<<<dimentions.x, dimentions.w>>>(A.data, B.data, result.data, A.rows, A.cols, dimentions.y, dimentions.z);
}

void DMatrixScale(DMATRIX A, DMATRIX result, double scalar) {
    if (result.rows != A.rows ||
        result.cols != A.cols)
        ERR("[ERROR] (DMatrixScale) Invalid matrices for the \"scale\" operation!");

    dim4 dimentions = KERNEL_BLOCK_SIZE(A.rows, A.cols);

    mat_scale<<<dimentions.x, dimentions.w>>>(A.data, result.data, scalar, A.rows, A.cols, dimentions.y, dimentions.z);
}

void DMatrixTranspose(DMATRIX A, DMATRIX result) {
    if (A.rows != result.cols ||
        A.cols != result.rows)
        ERR("[ERROR] (DMatrixTranspose) Invalid matrices for the \"transpose\" operation!");

    dim4 dimentions = KERNEL_BLOCK_SIZE(A.rows, A.cols);

    mat_transpose<<<dimentions.x, dimentions.w>>>(A.data, result.data, A.rows, A.cols, dimentions.y, dimentions.z);
}

void DMatrixApply(DMATRIX A, DMATRIX result, int op_code) {
    if (A.rows != result.rows ||
        A.cols != result.cols)
        ERR("[ERROR] (DMatrixApply) Invalid matrices for the \"transpose\" operation!") 
    else if (op_code > SQUARE)
        ERR("[ERROR] (DMatrixApply) Invalid opcode for the \"transpose\" operation!");

    dim4 dimentions = KERNEL_BLOCK_SIZE(A.rows, A.cols);

    mat_apply<<<dimentions.x, dimentions.w>>>(A.data, result.data, op_code, A.rows, A.cols, dimentions.y, dimentions.z);
}

DMATRIX DMatrixCreate(size_t rows, size_t cols) {
    DPDOUBLE data_ptr = NULL;

    if (cudaMalloc(&data_ptr, rows*cols*sizeof(double)) != cudaSuccess) {
        ERR("[ERROR] (DMatrixCreate) Could not allocate GPU memory!");
    }

    if (cudaMemset(data_ptr, 0, rows * cols * sizeof(double)) != cudaSuccess) {
        ERR("[ERROR] (DMatrixCreate) Could not set GPU memory!");
    }
    
    return DMATRIX{data_ptr, rows, cols};
}

void DMatrixFree(DMATRIX mat) {
    if (cudaFree(mat.data) != cudaSuccess)
        ERR("[ERROR] (DMatrixFree) Could not deallocate GPU memory!");
}

DMATRIX copyHostToDevice(HPDOUBLE host_data, size_t rows, size_t cols) {
    DMATRIX device_data = DMatrixCreate(rows, cols);

    __hostToDeviceData(host_data, device_data.data, rows, cols);
    

    return device_data;
}

HPDOUBLE copyDeviceToHost(DMATRIX device_data) {
    HPDOUBLE host_data = new double*[device_data.rows];

    for (size_t i = 0; i < device_data.rows; ++i) {
        host_data[i] = new double[device_data.cols];

        if (cudaMemcpy(host_data[i], device_data.data + (i * device_data.cols), device_data.cols * sizeof(double), cudaMemcpyDeviceToHost) != cudaSuccess)
            ERR("[ERROR] (copyDeviceToHost) Could not copy into Host memory!");
    }

    return host_data;
}

void __hostToDeviceData(HPDOUBLE host_data, DPDOUBLE device_data, size_t rows, size_t cols) {
    cudaError_t result;

    for (size_t i = 0; i < rows; ++i) {
        if ((result = cudaMemcpy(device_data + (i * cols), host_data[i], cols * sizeof(double), cudaMemcpyHostToDevice)) != cudaSuccess)
            {  std::cout << result << std::endl; ERR("[ERROR] (hostToDeviceData) Could not copy into GPU memory!"); };
    }
}