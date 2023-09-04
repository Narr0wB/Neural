
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

/*
                                                    ########## MASK MATRIX #########
    The mask matrix is a matrix of threads which is overlaid over one or more matrices similar to a 2D Convolution or Cross-Correlation and allows for 
    faster computation of same-shaped or single matrix matrix operations. The mask matrix moves along the y-axis going down by a stride given by the formula 
    ((gridDim.x * blockDim.x) / 64) where gridDim.x is the number of blocks, blockDim.x the number of threads in a block, and 64 the fixed x dimention of the matrix. 
    Once it has reached the bottom, the mask matrix moves along the x-axis by a stride of 64 and along the y-axis coming back up re-alligning its first row with the
    processed matrix's (or matrices) first row.

    Its composed of n blocks --> 1024 threads each
    m x 64 thread "matrix" where:
        m = (n*1024)/64
    
    A thread's indices i, j in the mask are given by the following formulas:
        i = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64)
        j = threadIdx.x % 64
*/


__global__ void mat_add(DPDOUBLE A, DPDOUBLE B, DPDOUBLE sum, size_t matRows, size_t matCols)
{
    size_t idxC = threadIdx.x % 64;
    size_t idxR = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64);
    size_t strideX = 64;
    size_t strideY = ((gridDim.x * blockDim.x) / 64);
    

    for (size_t i = idxR; i < matRows; i += strideX)
        for (size_t j = idxC; j < matCols; j += strideY)
            sum[ix(i, j, matCols)] = A[ix(i, j, matCols)] + B[ix(i, j, matCols)];
}

__global__ void mat_subtract(DPDOUBLE A, DPDOUBLE B, DPDOUBLE subtracted, size_t matRows, size_t matCols)
{
    size_t idxC = threadIdx.x % 64;
    size_t idxR = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64);
    size_t strideX = 64;
    size_t strideY = ((gridDim.x * blockDim.x) / 64);
    

    for (size_t i = idxR; i < matRows; i += strideX)
        for (size_t j = idxC; j < matCols; j += strideY)
            subtracted[ix(i, j, matCols)] = A[ix(i, j, matCols)] - B[ix(i, j, matCols)];
}

__global__ void mat_dot(DPDOUBLE A, DPDOUBLE B, DPDOUBLE dot, size_t dotRows, size_t dotCols, size_t commonDim)
{
    size_t idxR = blockIdx.x * blockDim.x + threadIdx.x;
    size_t strideX = gridDim.x * blockDim.x;

    for (size_t i = idxR; i < dotRows; i += strideX)
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
    size_t idxC = threadIdx.x % 64;
    size_t idxR = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64);
    size_t strideX = 64;
    size_t strideY = ((gridDim.x * blockDim.x) / 64);
    

    for (size_t i = idxR; i < matRows; i += strideX)
        for (size_t j = idxC; j < matCols; j += strideY)
            multiplied[ix(i, j, matCols)] = A[ix(i, j, matCols)] * B[ix(i, j, matCols)];
}

__global__ void mat_scale(DPDOUBLE A, DPDOUBLE scaled, double s, size_t matRows, size_t matCols)
{
    size_t idxC = threadIdx.x % 64;
    size_t idxR = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64);
    size_t strideX = 64;
    size_t strideY = ((gridDim.x * blockDim.x) / 64);
    

    for (size_t i = idxC; i < matRows; i += strideX)
        for (size_t j = idxR; j < matCols; j += strideY)
            scaled[ix(i, j, matCols)] = A[ix(i, j, matCols)] * s;
}

__global__ void mat_transpose(DPDOUBLE A, DPDOUBLE transposed, size_t A_rows, size_t A_cols)
{
    size_t idxC = threadIdx.x % 64;
    size_t idxR = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64);
    size_t strideX = 64;
    size_t strideY = ((gridDim.x * blockDim.x) / 64);

    for (size_t i = idxC; i < A_cols; i += strideY)
        for (size_t j = idxR; j < A_rows; j += strideX)
            transposed[ix(i, j, A_rows)] = A[ix(j, i, A_cols)];
}

__global__ void mat_apply(DPDOUBLE A, DPDOUBLE applied, int activationIdx, size_t matRows, size_t matCols)
{
    size_t idxC = threadIdx.x % 64;
    size_t idxR = blockIdx.x * (blockDim.x / 64) + (threadIdx.x / 64);
    size_t strideX = 64;
    size_t strideY = ((gridDim.x * blockDim.x) / 64);

    for (size_t i = idxR; i < matRows; i += strideX)
        for (size_t j = idxC; j < matCols; j += strideY)
            applied[ix(i, j, matCols)] = funclist[activationIdx](A[ix(i, j, matCols)]);
}

__global__ void dense_forwardprop(DPDOUBLE Wn, DPDOUBLE An, DPDOUBLE Bn, DPDOUBLE Zm, size_t Wrows, size_t Wcols)
{
    size_t idxR = blockIdx.x * blockDim.x + threadIdx.x;
    size_t strideX = gridDim.x * blockDim.x;

    for (size_t i = idxR; i < Wrows; i += strideX)
    {
        double sum = 0;
        for (size_t k = 0; k < Wcols; k++)
            sum += Wn[ix(i, k, Wcols)] * An[ix(k, 0, 1)];
        Zm[ix(i, 0, 1)] = sum + Bn[ix(i, 0, 1)];
    }
}

__global__ void dense_backprop(DPDOUBLE dY, DPDOUBLE Zm, DPDOUBLE An, DPDOUBLE dBn, DPDOUBLE dWn, size_t Zmrows, size_t Anrows, int derivativeIdx)
{
    size_t idxR = blockIdx.x * blockDim.x + threadIdx.x;
    size_t strideX = gridDim.x * blockDim.x;

    for (size_t i = idxR; i < Zmrows; i += strideX)
    {
        dBn[ix(i, 0, 1)] = funclist[derivativeIdx](Zm[ix(i, 0, 1)]) * dY[ix(i, 0, 1)];
        for (size_t j = 0; j < Anrows; j++)
        {
            dWn[ix(i, j, Anrows)] = dBn[ix(i, 0, 1)] * An[ix(j, 0, 1)];
        }
    }
}

// IN-DEVICE MATRIX OPERATIONS ---------------------------------------------------------------------

DeviceMatrix::DeviceMatrix(Matrix m) : m_alloced(true) {
    m_rows = m.rows();
    m_cols = m.cols();

    auto result = cudaMalloc(&m_Data, (m_rows * m_cols) * sizeof(double));

    if (result != cudaSuccess)
        ERR("[ERROR] (copyHostToDevice) Cannot allocate GPU memory!");

    for (int i = 0; i < m_rows; i++)
    {
        result = cudaMemcpy((void*)(m_Data + (m_cols * i * sizeof(double))), m[i], m_cols * sizeof(double), cudaMemcpyHostToDevice);
        if (result != cudaSuccess)
            ERR("[ERROR] (copyHostToDevice) Cannot could not copy into GPU!");
    }

}

DeviceMatrix DeviceMatrix::operator+(DeviceMatrix& other) {
    if (m_rows != other.m_rows || 
        m_cols != other.m_cols)
        ERR("[ERROR] (DeviceMatrix::operator+) Invalid matrices for the \"+\" operation!")

    DeviceMatrix result(m_rows, m_cols);

    mat_add<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(m_Data, other.m_Data, result.m_Data, m_rows, m_cols);

    return result;
}

DeviceMatrix DeviceMatrix::operator-(DeviceMatrix& other) {
    if (m_rows != other.m_rows || 
        m_cols != other.m_cols)
        ERR("[ERROR] (DeviceMatrix::operator-) Invalid matrices for the \"-\" operation!")

    DeviceMatrix result(m_rows, m_cols);

    mat_subtract<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(m_Data, other.m_Data, result.m_Data, m_rows, m_cols);

    return result;
}

DeviceMatrix DeviceMatrix::operator*(DeviceMatrix& other) {
    if (m_rows != other.m_cols || 
        m_cols != other.m_rows)
        ERR("[ERROR] (DeviceMatrix::operator*) Invalid matrices for the \"*\" operation!")

    DeviceMatrix result(m_rows, other.m_cols);

    mat_dot<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(m_Data, other.m_Data, result.m_Data, m_rows, other.m_cols, m_cols);

    return result;
}

DeviceMatrix DeviceMatrix::operator*(double scalar) {
    DeviceMatrix result(m_rows, m_cols);

    mat_scale<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(m_Data, result.m_Data, scalar, m_rows, m_cols);

    return result;
}

void DeviceMatrix::operator=(DeviceMatrix other)
{
}

DeviceMatrix DeviceMatrix::transpose() {
    DeviceMatrix result(m_cols, m_rows);

    mat_transpose<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(m_Data, result.m_Data, m_rows, m_cols);

    return result;
} 

void DeviceMatrix::apply(DeviceMatrix& result, int op_idx) {
    if (op_idx < 0 || op_idx > 2)
        ERR("[ERROR] (DeviceMatrix::apply) Invalid operation index!")

    if (result.rows() != m_rows ||
        result.cols() != m_cols)
        ERR("[ERROR] (DeviceMatrix::apply) Invalid matrix!")

    mat_apply<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(m_Data, result.m_Data, op_idx, m_rows, m_cols);
} 

DeviceMatrix hadamard(DeviceMatrix mat1, DeviceMatrix mat2) {
    if (mat1.m_rows != mat2.m_rows || 
        mat1.m_cols != mat2.m_cols)
        ERR("[ERROR] (DeviceMatrix::hadamard) Invalid matrices for the hadamard operation!");
    
    DeviceMatrix result(mat1.m_rows, mat2.m_cols);

    mat_multiply<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(mat1.m_Data, mat2.m_Data, result.m_Data, mat1.m_rows, mat1.m_cols);

    return result;
}

HPDOUBLE DeviceMatrix::toHost() {
    HPDOUBLE host_data = new double*[m_rows];

    for (int i = 0; i < m_rows; ++i) {
        host_data[i] = new double[m_cols];

        if (cudaMemcpy(host_data[i], m_Data + (i * m_cols), m_cols, cudaMemcpyDeviceToHost) != cudaSuccess)
            ERR("[ERROR] (DeviceMatrix::toHost) Could not copy data to host!");
    }

    return host_data;
}

void DeviceMatrix::set(Matrix m)
{
    if (m.rows() != m_rows || m.cols() != m_cols)
        ERR("[ERROR] (DeviceMatrix::set) Invalid matrices!");

    for (int i = 0; i < m_rows; ++i) {
        if (cudaMemcpy((void*)(m_Data + (i * m_cols * sizeof(double))), m.getdata()[i], m_cols * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
            ERR("[ERROR] (DeviceMatrix::set) Could not copy the data from the host to the device!")
    }
}

void DeviceMatrix::scale(double scalar) {
    mat_scale<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(m_Data, m_Data, scalar, m_rows, m_cols);
}

void denseForwardProp(DeviceMatrix& Zm, DeviceMatrix Wn, DeviceMatrix An, DeviceMatrix Bn) {
    if (An.cols() != 1 ||
        Bn.cols() != 1 ||
        An.rows() != Wn.cols() ||
        Bn.rows() != Wn.rows())
        ERR("[ERROR] (denseForwardProp) Invlid matrices!")

    if (Zm.rows() != Wn.rows() || Zm.cols() != 1)
        ERR("[ERROR] (denseForwardProp) Invlid matrices!")

    dense_forwardprop<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(Wn.data(), An.data(), Bn.data(), Zm.data(), Wn.rows(), Wn.cols());
}

void denseBackProp(DeviceMatrix& dWn, DeviceMatrix& dBn, DeviceMatrix dY, DeviceMatrix Zm, DeviceMatrix An, int op_idx) {
    if (An.cols() != 1 ||
        Zm.cols() != 1 ||
        dY.cols() != 1 ||
        Zm.rows() != dY.rows())
        ERR("[ERROR] (denseBackProp) Invalid matrices!")

    if (dWn.rows() != Zm.rows() ||
        dWn.cols() != An.rows() ||
        dBn.rows() != Zm.rows() ||
        dBn.cols() != 1)
        ERR("[ERROR] (denseBackProp) Invalid matrices!");

    dense_backprop<<<KERNEL_GRID_SIZE, KERNEL_BLOCK_SIZE>>>(dY.data(), Zm.data(), An.data(), dBn.data(), dWn.data(), Zm.rows(), An.rows(), op_idx);
}

