
#include "cuda/cuda_ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include <cublas_v2.h>

#define ROUND_UP(n, BLOCK) (((n) + (BLOCK) - 1) / (BLOCK))

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status_ = (call);                                        \
        if (status_ != CUBLAS_STATUS_SUCCESS)                                   \
            throw std::runtime_error("cuBLAS error: code " +                    \
                                     std::to_string(static_cast<int>(status_)) +\
                                     " at " + __FILE__ + ":" +                  \
                                     std::to_string(__LINE__));                 \
    } while (0)

constexpr size_t BLOCK      = 256;
constexpr size_t TILE       = 16;
constexpr size_t MAX_GRID_Z = 65535;

namespace {

struct CUBLASHandle {
    cublasHandle_t handle = nullptr;
    CUBLASHandle()  { CUBLAS_CHECK(cublasCreate(&handle)); }
    ~CUBLASHandle() { if (handle) cublasDestroy(handle); }
    CUBLASHandle(const CUBLASHandle&)            = delete;
    CUBLASHandle& operator=(const CUBLASHandle&) = delete;
};

inline cublasHandle_t get_cublas_handle()
{
    static CUBLASHandle instance;
    return instance.handle;
}

struct CudaMemPoolConfig {
    CudaMemPoolConfig()
    {
        int device = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaMemPool_t pool = nullptr;
        CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, device));
        u64 threshold = std::numeric_limits<u64>::max();
        CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }
    CudaMemPoolConfig(const CudaMemPoolConfig&)            = delete;
    CudaMemPoolConfig& operator=(const CudaMemPoolConfig&) = delete;
};

template <Numeric T>
__global__ void fill_kernel(TensorView<T> dst, T value)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < dst.size())
        dst.data[i] = value;
}

struct BroadcastStrides {
    size_t a[MAX_DIMS] = {};
    size_t b[MAX_DIMS] = {};
};

inline BroadcastStrides make_broadcast_strides(const Shape& sa, const Shape& sb, const Shape& so)
{
    BroadcastStrides st;
    size_t sa_acc = 1, sb_acc = 1;
    for (size_t k = so.ndim; k-- > 0;) {
        if (k + sa.ndim >= so.ndim) {
            const size_t d = sa[k + sa.ndim - so.ndim];
            st.a[k] = (d == 1) ? 0 : sa_acc;
            sa_acc *= d;
        }
        if (k + sb.ndim >= so.ndim) {
            const size_t d = sb[k + sb.ndim - so.ndim];
            st.b[k] = (d == 1) ? 0 : sb_acc;
            sb_acc *= d;
        }
    }
    return st;
}

template <Numeric T>
__global__ void add_kernel(TensorView<const T> a, TensorView<const T> b, TensorView<T> out)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < out.size())
        out.data[i] = a.data[i] + b.data[i];
}

template <Numeric T, typename Idx>
__global__ void add_broadcast_kernel(TensorView<const T> a, TensorView<const T> b,
                                     TensorView<T> out, BroadcastStrides st)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= out.size())
        return;

    Idx rem = static_cast<Idx>(i);
    size_t ia = 0, ib = 0;
    for (size_t k = out.shape.ndim; k-- > 1;) {
        const Idx dim   = static_cast<Idx>(out.shape[k]);
        const Idx coord = rem % dim;
        rem /= dim;
        ia += static_cast<size_t>(coord) * st.a[k];
        ib += static_cast<size_t>(coord) * st.b[k];
    }
    ia += static_cast<size_t>(rem) * st.a[0];
    ib += static_cast<size_t>(rem) * st.b[0];
    out.data[i] = a.data[ia] + b.data[ib];
}

template <Numeric T>
__global__ void sub_kernel(TensorView<const T> a, TensorView<const T> b, TensorView<T> out)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < out.size())
        out.data[i] = a.data[i] - b.data[i];
}

template <Numeric T, typename Idx>
__global__ void sub_broadcast_kernel(TensorView<const T> a, TensorView<const T> b,
                                     TensorView<T> out, BroadcastStrides st)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= out.size())
        return;

    Idx rem = static_cast<Idx>(i);
    size_t ia = 0, ib = 0;
    for (size_t k = out.shape.ndim; k-- > 1;) {
        const Idx dim   = static_cast<Idx>(out.shape[k]);
        const Idx coord = rem % dim;
        rem /= dim;
        ia += static_cast<size_t>(coord) * st.a[k];
        ib += static_cast<size_t>(coord) * st.b[k];
    }
    ia += static_cast<size_t>(rem) * st.a[0];
    ib += static_cast<size_t>(rem) * st.b[0];
    out.data[i] = a.data[ia] - b.data[ib];
}

template <Numeric T>
__global__ void transpose_kernel(TensorView<const T> a, TensorView<T> out,
                                 size_t batch_count, size_t R, size_t C)
{
    const size_t col = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t row = static_cast<size_t>(blockIdx.y) * blockDim.y + threadIdx.y;

    if (row >= R || col >= C)
        return;

    const size_t stride = R * C;
    for (size_t batch = blockIdx.z; batch < batch_count; batch += gridDim.z) {
        const T* A  = a.data   + batch * stride;
        T*       AT = out.data + batch * stride;
        AT[col * R + row] = A[row * C + col];
    }
}

template <Numeric T>
__global__ void sgd_step_kernel(TensorView<T> params, TensorView<const T> grad, T lr)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < params.size())
        params.data[i] -= lr * grad.data[i];
}

template <Numeric T>
__global__ void relu_forward_kernel(TensorView<const T> in, TensorView<T> out)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < out.size()) {
        const T v = in.data[i];
        out.data[i] = v > T{0} ? v : T{0};
    }
}

template <Numeric T>
__global__ void relu_backward_kernel(TensorView<const T> in, TensorView<const T> dY, TensorView<T> dX)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < dX.size())
        dX.data[i] = in.data[i] > T{0} ? dY.data[i] : T{0};
}

template <Numeric T>
__global__ void maxpool2d_forward_kernel(TensorView<const T> x, TensorView<T> y,
                                         TensorView<size_t> max_indices, size_t P)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= y.size())
        return;

    const size_t C    = y.shape[1];
    const size_t outH = y.shape[2];
    const size_t outW = y.shape[3];
    const size_t W    = x.shape[3];

    size_t tmp = tid;
    const size_t j = tmp % outW; tmp /= outW;
    const size_t i = tmp % outH; tmp /= outH;
    const size_t c = tmp % C;    tmp /= C;
    const size_t n = tmp;

    const size_t base_row_off = x.shape.offset(n, c, i * P, 0);
    size_t max_idx = base_row_off + j * P;
    T max_val      = x.data[max_idx];

    for (size_t u = 0; u < P; ++u) {
        const size_t row_off = base_row_off + u * W;
        for (size_t v = 0; v < P; ++v) {
            const size_t idx = row_off + j * P + v;
            const T val = x.data[idx];
            if (val > max_val) {
                max_val = val;
                max_idx = idx;
            }
        }
    }

    max_indices.data[tid] = max_idx;
    y.data[tid]           = max_val;
}

template <Numeric T>
__global__ void maxpool2d_backward_kernel(TensorView<const T> dY, TensorView<const size_t> max_indices,
                                          TensorView<T> dX)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < max_indices.size())
        dX.data[max_indices.data[i]] = dY.data[i];
}

template <Numeric T>
__global__ void conv2d_forward_kernel(TensorView<const T> x, TensorView<const T> filters,
                                      TensorView<const T> bias, TensorView<T> y)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= y.size())
        return;

    const size_t O    = filters.shape[0];
    const size_t I    = filters.shape[1];
    const size_t K    = filters.shape[2];
    const size_t outH = y.shape[2];
    const size_t outW = y.shape[3];

    size_t tmp = tid;
    const size_t j   = tmp % outW; tmp /= outW;
    const size_t i   = tmp % outH; tmp /= outH;
    const size_t out = tmp % O;    tmp /= O;
    const size_t n   = tmp;

    T acc = bias.data[out];
    for (size_t c = 0; c < I; ++c) {
        for (size_t u = 0; u < K; ++u) {
            for (size_t v = 0; v < K; ++v) {
                acc += filters(out, c, u, v) * x(n, c, i + u, j + v);
            }
        }
    }
    y.data[tid] = acc;
}

template <Numeric T>
__global__ void conv2d_backward_filters_kernel(TensorView<const T> input, TensorView<const T> dY,
                                               TensorView<T> filters_grad)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= filters_grad.size())
        return;

    const size_t N    = input.shape[0];
    const size_t I    = filters_grad.shape[1];
    const size_t K    = filters_grad.shape[2];
    const size_t outH = dY.shape[2];
    const size_t outW = dY.shape[3];

    size_t tmp = tid;
    const size_t j   = tmp % K; tmp /= K;
    const size_t i   = tmp % K; tmp /= K;
    const size_t in  = tmp % I; tmp /= I;
    const size_t out = tmp;

    T acc = T{0};
    for (size_t n = 0; n < N; ++n) {
        for (size_t u = 0; u < outH; ++u) {
            for (size_t v = 0; v < outW; ++v) {
                acc += dY(n, out, u, v) * input(n, in, i + u, j + v);
            }
        }
    }
    filters_grad.data[tid] += acc;
}

template <Numeric T>
__global__ void conv2d_backward_bias_kernel(TensorView<const T> dY, TensorView<T> bias_grad)
{
    const size_t out = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (out >= bias_grad.size())
        return;

    const size_t N    = dY.shape[0];
    const size_t outH = dY.shape[2];
    const size_t outW = dY.shape[3];

    T acc = T{0};
    for (size_t n = 0; n < N; ++n) {
        for (size_t u = 0; u < outH; ++u) {
            for (size_t v = 0; v < outW; ++v) {
                acc += dY(n, out, u, v);
            }
        }
    }
    bias_grad.data[out] += acc;
}

template <Numeric T>
__global__ void conv2d_backward_input_kernel(TensorView<const T> dY, TensorView<const T> filters,
                                             TensorView<T> dX)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= dX.size())
        return;

    const size_t O    = filters.shape[0];
    const size_t I    = filters.shape[1];
    const size_t K    = filters.shape[2];
    const size_t H    = dX.shape[2];
    const size_t W    = dX.shape[3];
    const size_t outH = dY.shape[2];
    const size_t outW = dY.shape[3];

    size_t tmp = tid;
    const size_t j  = tmp % W; tmp /= W;
    const size_t i  = tmp % H; tmp /= H;
    const size_t in = tmp % I; tmp /= I;
    const size_t n  = tmp;

    const size_t u_begin = i >= K ? i - K + 1 : 0;
    const size_t u_end   = (i + 1 < outH) ? (i + 1) : outH;
    const size_t v_begin = j >= K ? j - K + 1 : 0;
    const size_t v_end   = (j + 1 < outW) ? (j + 1) : outW;

    T acc = T{0};
    for (size_t out = 0; out < O; ++out) {
        for (size_t u = u_begin; u < u_end; ++u) {
            for (size_t v = v_begin; v < v_end; ++v) {
                acc += dY(n, out, u, v) * filters(out, in, i - u, j - v);
            }
        }
    }
    dX.data[tid] = acc;
}

template <Numeric T>
__global__ void cross_entropy_forward_kernel(TensorView<const T> pred, TensorView<const T> target,
                                             TensorView<T> probs, TensorView<T> losses)
{
    const size_t n = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t N = pred.shape[0];
    if (n >= N)
        return;

    const size_t L = pred.shape[1];

    T max_val = pred(n, 0);
    for (size_t i = 1; i < L; ++i) {
        const T v = pred(n, i);
        if (v > max_val)
            max_val = v;
    }

    T sample_loss = T{0};
    T log_arg     = T{0};
    for (size_t i = 0; i < L; ++i) {
        const T shifted = pred(n, i) - max_val;
        const T e       = exp(shifted);
        probs(n, i)     = e;
        sample_loss    -= target(n, i) * shifted;
        log_arg        += e;
    }

    for (size_t i = 0; i < L; ++i)
        probs(n, i) /= log_arg;

    losses(n) = sample_loss + log(log_arg);
}

template <Numeric T>
__global__ void cross_entropy_backward_kernel(TensorView<const T> probs, TensorView<const T> target,
                                              TensorView<T> out, T inv_N)
{
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < out.size())
        out.data[i] = inv_N * (probs.data[i] - target.data[i]);
}

template <ReduceOp op, Numeric T>
__global__ void reduce_kernel(TensorView<const T> x, size_t axis_len, size_t inner_stride, TensorView<T> out)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= out.size())
        return;

    const size_t i = tid / inner_stride;
    const size_t j = tid % inner_stride;

    T acc = T{0};
    for (size_t k = 0; k < axis_len; ++k)
        acc += x.data[(i * axis_len + k) * inner_stride + j];

    if constexpr (op == ReduceOp::SUM)
        out.data[tid] += acc;
    else if constexpr (op == ReduceOp::SUB)
        out.data[tid] -= acc;
}

template <Numeric T>
__global__ void gather_kernel(TensorView<const T> src, TensorView<const size_t> indices,
                              TensorView<T> out, size_t stride)
{
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (tid >= out.size())
        return;

    const size_t b   = tid / stride;
    const size_t rem = tid % stride;
    out.data[tid]    = src.data[indices.data[b] * stride + rem];
}

} // namespace

namespace cuda_ops {

void init_mem_pool()
{
    static CudaMemPoolConfig instance;
    (void)instance;
}

template <Numeric T>
void fill(TensorView<T> dst, T value)
{
    const size_t n = dst.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    fill_kernel<<<grid, static_cast<u32>(BLOCK)>>>(dst, value);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void add(TensorView<const T> a, TensorView<const T> b, TensorView<T> out)
{
    const bool same = (a.shape == b.shape) && (b.shape == out.shape);
    if (!same && !(a.shape.broadcast_to(b.shape) == out.shape))
        throw std::invalid_argument("cuda_ops::add: invalid input shapes");

    const size_t n = out.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    if (same) {
        add_kernel<<<grid, static_cast<u32>(BLOCK)>>>(a, b, out);
    } else {
        const BroadcastStrides st = make_broadcast_strides(a.shape, b.shape, out.shape);
        if (n <= std::numeric_limits<u32>::max())
            add_broadcast_kernel<T, u32><<<grid, static_cast<u32>(BLOCK)>>>(a, b, out, st);
        else
            add_broadcast_kernel<T, size_t><<<grid, static_cast<u32>(BLOCK)>>>(a, b, out, st);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void sub(TensorView<const T> a, TensorView<const T> b, TensorView<T> out)
{
    const bool same = (a.shape == b.shape) && (b.shape == out.shape);
    if (!same && !(a.shape.broadcast_to(b.shape) == out.shape))
        throw std::invalid_argument("cuda_ops::sub: invalid input shapes");

    const size_t n = out.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    if (same) {
        sub_kernel<<<grid, static_cast<u32>(BLOCK)>>>(a, b, out);
    } else {
        const BroadcastStrides st = make_broadcast_strides(a.shape, b.shape, out.shape);
        if (n <= std::numeric_limits<u32>::max())
            sub_broadcast_kernel<T, u32><<<grid, static_cast<u32>(BLOCK)>>>(a, b, out, st);
        else
            sub_broadcast_kernel<T, size_t><<<grid, static_cast<u32>(BLOCK)>>>(a, b, out, st);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void matmul(TensorView<const T> a, TensorView<const T> b, TensorView<T> out,
            bool trans_a, bool trans_b, bool accumulate)
{
    const size_t ndim = a.shape.ndim;
    if (ndim < 2 || b.shape.ndim != ndim || out.shape.ndim != ndim)
        throw std::invalid_argument("cuda_ops::matmul: inputs and output must have matching rank >= 2");

    size_t batch = 1;
    for (size_t i = 0; i + 2 < ndim; ++i) {
        if (a.shape[i] != b.shape[i] || a.shape[i] != out.shape[i])
            throw std::invalid_argument("cuda_ops::matmul: batch dimensions do not match");
        batch *= a.shape[i];
    }

    const size_t M   = trans_a ? a.shape[ndim - 1] : a.shape[ndim - 2];
    const size_t K   = trans_a ? a.shape[ndim - 2] : a.shape[ndim - 1];
    const size_t K_b = trans_b ? b.shape[ndim - 1] : b.shape[ndim - 2];
    const size_t N   = trans_b ? b.shape[ndim - 2] : b.shape[ndim - 1];

    if (K_b != K)
        throw std::invalid_argument("cuda_ops::matmul: inner dimensions do not match");
    if (out.shape[ndim - 2] != M || out.shape[ndim - 1] != N)
        throw std::invalid_argument("cuda_ops::matmul: output matrix shape does not match {..., M, N}");

    if (batch == 0 || M == 0 || K == 0 || N == 0)
        return;

    const cublasHandle_t    handle = get_cublas_handle();
    const cublasOperation_t op_a   = trans_b ? CUBLAS_OP_T : CUBLAS_OP_N;
    const cublasOperation_t op_b   = trans_a ? CUBLAS_OP_T : CUBLAS_OP_N;

    const int m   = static_cast<int>(N);
    const int n   = static_cast<int>(M);
    const int k   = static_cast<int>(K);
    const int lda = static_cast<int>(b.shape[ndim - 1]);
    const int ldb = static_cast<int>(a.shape[ndim - 1]);
    const int ldc = static_cast<int>(N);

    const T alpha = T{1};
    const T beta  = accumulate ? T{1} : T{0};

    if constexpr (std::is_same_v<T, fp32>) {
        if (batch > 1) {
            CUBLAS_CHECK(cublasSgemmStridedBatched(
                handle, op_a, op_b, m, n, k,
                &alpha,
                b.data, lda, static_cast<long long>(K * N),
                a.data, ldb, static_cast<long long>(M * K),
                &beta,
                out.data, ldc, static_cast<long long>(M * N),
                static_cast<int>(batch)));
        } else {
            CUBLAS_CHECK(cublasSgemm(
                handle, op_a, op_b, m, n, k,
                &alpha,
                b.data, lda,
                a.data, ldb,
                &beta,
                out.data, ldc));
        }
    }
    else if constexpr (std::is_same_v<T, fp64>) {
        if (batch > 1) {
            CUBLAS_CHECK(cublasDgemmStridedBatched(
                handle, op_a, op_b, m, n, k,
                &alpha,
                b.data, lda, static_cast<long long>(K * N),
                a.data, ldb, static_cast<long long>(M * K),
                &beta,
                out.data, ldc, static_cast<long long>(M * N),
                static_cast<int>(batch)));
        } else {
            CUBLAS_CHECK(cublasDgemm(
                handle, op_a, op_b, m, n, k,
                &alpha,
                b.data, lda,
                a.data, ldb,
                &beta,
                out.data, ldc));
        }
    }
}

template <Numeric T>
void transpose(TensorView<const T> a, TensorView<T> out)
{
    const size_t ndim = a.shape.ndim;
    if (ndim < 2 || out.shape.ndim != ndim)
        throw std::invalid_argument("cuda_ops::transpose: input and output must have matching rank >= 2");

    size_t batch = 1;
    for (size_t i = 0; i + 2 < ndim; ++i) {
        if (a.shape[i] != out.shape[i])
            throw std::invalid_argument("cuda_ops::transpose: batch dimensions do not match");
        batch *= a.shape[i];
    }

    const size_t R = a.shape[ndim - 2];
    const size_t C = a.shape[ndim - 1];

    if (out.shape[ndim - 2] != C || out.shape[ndim - 1] != R)
        throw std::invalid_argument("cuda_ops::transpose: output matrix shape does not match {..., C, R}");

    if (batch == 0 || R == 0 || C == 0)
        return;

    dim3 block(static_cast<u32>(TILE), static_cast<u32>(TILE));
    dim3 grid(static_cast<u32>(ROUND_UP(C, TILE)),
              static_cast<u32>(ROUND_UP(R, TILE)),
              static_cast<u32>(std::min(batch, MAX_GRID_Z)));

    transpose_kernel<<<grid, block>>>(a, out, batch, R, C);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void sgd_step(TensorView<T> params, TensorView<const T> grad, T lr)
{
    if (!(params.shape == grad.shape))
        throw std::invalid_argument("cuda_ops::sgd_step: params and grad shapes do not match");

    const size_t n = params.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    sgd_step_kernel<<<grid, static_cast<u32>(BLOCK)>>>(params, grad, lr);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void relu_forward(TensorView<const T> in, TensorView<T> out)
{
    if (!(in.shape == out.shape))
        throw std::invalid_argument("cuda_ops::relu_forward: input and output shapes do not match");

    const size_t n = in.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    relu_forward_kernel<<<grid, static_cast<u32>(BLOCK)>>>(in, out);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void relu_backward(TensorView<const T> in, TensorView<const T> dY, TensorView<T> dX)
{
    if (!(in.shape == dY.shape) || !(in.shape == dX.shape))
        throw std::invalid_argument("cuda_ops::relu_backward: input and gradient shapes do not match");

    const size_t n = dX.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    relu_backward_kernel<<<grid, static_cast<u32>(BLOCK)>>>(in, dY, dX);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void maxpool2d_forward(TensorView<const T> x, TensorView<T> y, TensorView<size_t> max_indices, size_t pool_size)
{
    if (pool_size == 0 || x.shape.ndim != 4 || x.shape[2] < pool_size || x.shape[3] < pool_size)
        throw std::invalid_argument("cuda_ops::maxpool2d_forward: expected 4D input with H, W >= pool_size > 0");

    const Shape expected{x.shape[0], x.shape[1], x.shape[2] / pool_size, x.shape[3] / pool_size};
    if (!(y.shape == expected) || !(max_indices.shape == expected))
        throw std::invalid_argument("cuda_ops::maxpool2d_forward: output or max_indices shape mismatch");

    const size_t n = y.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    maxpool2d_forward_kernel<<<grid, static_cast<u32>(BLOCK)>>>(x, y, max_indices, pool_size);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void maxpool2d_backward(TensorView<const T> dY, TensorView<const size_t> max_indices, TensorView<T> dX)
{
    if (!(dY.shape == max_indices.shape))
        throw std::invalid_argument("cuda_ops::maxpool2d_backward: dY and max_indices shapes do not match");

    const size_t n = max_indices.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    maxpool2d_backward_kernel<<<grid, static_cast<u32>(BLOCK)>>>(dY, max_indices, dX);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void conv2d_forward(TensorView<const T> x, TensorView<const T> filters, TensorView<const T> bias, TensorView<T> y)
{
    if (x.shape.ndim != 4 || filters.shape.ndim != 4 || bias.shape.ndim != 1)
        throw std::invalid_argument("cuda_ops::conv2d_forward: invalid tensor ranks");

    const size_t N = x.shape[0];
    const size_t I = x.shape[1];
    const size_t H = x.shape[2];
    const size_t W = x.shape[3];
    const size_t O = filters.shape[0];
    const size_t K = filters.shape[2];

    if (filters.shape[1] != I || filters.shape[3] != K || bias.shape[0] != O || K == 0 || H < K || W < K)
        throw std::invalid_argument("cuda_ops::conv2d_forward: incompatible filter/bias/input shapes");

    const Shape expected{N, O, H - K + 1, W - K + 1};
    if (!(y.shape == expected))
        throw std::invalid_argument("cuda_ops::conv2d_forward: output shape mismatch");

    const size_t n = y.size();
    if (n == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    conv2d_forward_kernel<<<grid, static_cast<u32>(BLOCK)>>>(x, filters, bias, y);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void conv2d_backward(TensorView<const T> input, TensorView<const T> filters, TensorView<const T> dY,
                     TensorView<T> filters_grad, TensorView<T> bias_grad, TensorView<T> dX)
{
    if (input.shape.ndim != 4 || filters.shape.ndim != 4 || dY.shape.ndim != 4 ||
        filters_grad.shape.ndim != 4 || bias_grad.shape.ndim != 1 || dX.shape.ndim != 4)
        throw std::invalid_argument("cuda_ops::conv2d_backward: invalid tensor ranks");

    const size_t N = input.shape[0];
    const size_t I = input.shape[1];
    const size_t H = input.shape[2];
    const size_t W = input.shape[3];
    const size_t O = filters.shape[0];
    const size_t K = filters.shape[2];

    if (filters.shape[1] != I || filters.shape[3] != K || K == 0 || H < K || W < K)
        throw std::invalid_argument("cuda_ops::conv2d_backward: incompatible input/filter shapes");
    if (!(dY.shape == Shape{N, O, H - K + 1, W - K + 1}))
        throw std::invalid_argument("cuda_ops::conv2d_backward: dY shape mismatch");
    if (!(filters_grad.shape == filters.shape) || bias_grad.shape[0] != O || !(dX.shape == input.shape))
        throw std::invalid_argument("cuda_ops::conv2d_backward: output gradient shape mismatch");

    if (const size_t n_filters = filters_grad.size(); n_filters > 0) {
        const u32 grid = static_cast<u32>(ROUND_UP(n_filters, BLOCK));
        conv2d_backward_filters_kernel<<<grid, static_cast<u32>(BLOCK)>>>(input, dY, filters_grad);
        CUDA_CHECK(cudaGetLastError());
    }

    if (const size_t n_bias = bias_grad.size(); n_bias > 0) {
        const u32 grid = static_cast<u32>(ROUND_UP(n_bias, BLOCK));
        conv2d_backward_bias_kernel<<<grid, static_cast<u32>(BLOCK)>>>(dY, bias_grad);
        CUDA_CHECK(cudaGetLastError());
    }

    if (const size_t n_dx = dX.size(); n_dx > 0) {
        const u32 grid = static_cast<u32>(ROUND_UP(n_dx, BLOCK));
        conv2d_backward_input_kernel<<<grid, static_cast<u32>(BLOCK)>>>(dY, filters, dX);
        CUDA_CHECK(cudaGetLastError());
    }
}

template <Numeric T>
void cross_entropy_forward(TensorView<const T> pred, TensorView<const T> target, TensorView<T> probs, TensorView<T> loss)
{
    if (pred.shape.ndim != 2 || !(pred.shape == target.shape) || !(pred.shape == probs.shape) ||
        pred.shape[0] == 0 || pred.shape[1] == 0)
        throw std::invalid_argument("cuda_ops::cross_entropy_forward: expected matching non-empty 2D tensors");
    if (loss.shape.ndim != 1 || loss.shape[0] != pred.shape[0])
        throw std::invalid_argument("cuda_ops::cross_entropy_forward: loss shape must be {N}");

    const size_t N = pred.shape[0];

    const u32 grid = static_cast<u32>(ROUND_UP(N, BLOCK));
    cross_entropy_forward_kernel<<<grid, static_cast<u32>(BLOCK)>>>(pred, target, probs, loss);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void cross_entropy_backward(TensorView<const T> probs, TensorView<const T> target, TensorView<T> out)
{
    if (probs.shape.ndim != 2 || !(probs.shape == target.shape) || !(probs.shape == out.shape) ||
        probs.shape[0] == 0 || probs.shape[1] == 0)
        throw std::invalid_argument("cuda_ops::cross_entropy_backward: expected matching non-empty 2D tensors");

    const size_t N = probs.shape[0];
    const T inv_N  = T{1} / static_cast<T>(N);
    const size_t n = out.size();

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    cross_entropy_backward_kernel<<<grid, static_cast<u32>(BLOCK)>>>(probs, target, out, inv_N);
    CUDA_CHECK(cudaGetLastError());
}

template <ReduceOp op, Numeric T>
void reduce(TensorView<const T> x, size_t axis, TensorView<T> out)
{
    if (axis >= x.shape.ndim)
        throw std::invalid_argument("cuda_ops::reduce: invalid axis");

    size_t outer_stride = 1;
    size_t inner_stride = 1;
    for (size_t i = 0; i < axis; ++i)
        outer_stride *= x.shape[i];
    for (size_t i = axis + 1; i < x.shape.ndim; ++i)
        inner_stride *= x.shape[i];

    if (out.size() != outer_stride * inner_stride)
        throw std::invalid_argument("cuda_ops::reduce: invalid output shape");

    const size_t axis_len = x.shape[axis];
    const size_t n        = out.size();
    if (n == 0 || axis_len == 0)
        return;

    const u32 grid = static_cast<u32>(ROUND_UP(n, BLOCK));
    reduce_kernel<op><<<grid, static_cast<u32>(BLOCK)>>>(x, axis_len, inner_stride, out);
    CUDA_CHECK(cudaGetLastError());
}

template <Numeric T>
void gather(TensorView<const T> src, TensorView<const size_t> indices, TensorView<T> out)
{
    if (src.shape.ndim == 0 || indices.shape.ndim != 1 || out.shape.ndim != src.shape.ndim)
        throw std::invalid_argument("cuda_ops::gather: invalid tensor ranks");
    if (out.shape[0] != indices.shape[0])
        throw std::invalid_argument("cuda_ops::gather: output batch size must match indices length");
    for (size_t i = 1; i < src.shape.ndim; ++i) {
        if (out.shape[i] != src.shape[i])
            throw std::invalid_argument("cuda_ops::gather: trailing dimensions of src and out do not match");
    }

    const size_t n = out.size();
    if (n == 0)
        return;
    if (src.shape[0] == 0)
        throw std::invalid_argument("cuda_ops::gather: cannot gather from an empty src tensor");

    const size_t stride = src.size() / src.shape[0];
    const u32 grid      = static_cast<u32>(ROUND_UP(n, BLOCK));
    gather_kernel<<<grid, static_cast<u32>(BLOCK)>>>(src, indices, out, stride);
    CUDA_CHECK(cudaGetLastError());
}

#define INSTANTIATE_FLOAT_OPS(T)                                                                    \
    template void fill<T>(TensorView<T>, T);                                                        \
    template void add<T>(TensorView<const T>, TensorView<const T>, TensorView<T>);                  \
    template void sub<T>(TensorView<const T>, TensorView<const T>, TensorView<T>);                  \
    template void matmul<T>(TensorView<const T>, TensorView<const T>, TensorView<T>, bool, bool, bool); \
    template void transpose<T>(TensorView<const T>, TensorView<T>);                                 \
    template void sgd_step<T>(TensorView<T>, TensorView<const T>, T);                               \
    template void relu_forward<T>(TensorView<const T>, TensorView<T>);                              \
    template void relu_backward<T>(TensorView<const T>, TensorView<const T>, TensorView<T>);        \
    template void maxpool2d_forward<T>(TensorView<const T>, TensorView<T>, TensorView<size_t>, size_t); \
    template void maxpool2d_backward<T>(TensorView<const T>, TensorView<const size_t>, TensorView<T>);  \
    template void conv2d_forward<T>(TensorView<const T>, TensorView<const T>, TensorView<const T>, TensorView<T>); \
    template void conv2d_backward<T>(TensorView<const T>, TensorView<const T>, TensorView<const T>, \
                                     TensorView<T>, TensorView<T>, TensorView<T>);                  \
    template void cross_entropy_forward<T>(TensorView<const T>, TensorView<const T>, TensorView<T>, TensorView<T>); \
    template void cross_entropy_backward<T>(TensorView<const T>, TensorView<const T>, TensorView<T>); \
    template void reduce<ReduceOp::SUM, T>(TensorView<const T>, size_t, TensorView<T>);             \
    template void reduce<ReduceOp::SUB, T>(TensorView<const T>, size_t, TensorView<T>);             \
    template void gather<T>(TensorView<const T>, TensorView<const size_t>, TensorView<T>);

INSTANTIATE_FLOAT_OPS(fp32)
INSTANTIATE_FLOAT_OPS(fp64)
#undef INSTANTIATE_FLOAT_OPS

template void fill<size_t>(TensorView<size_t>, size_t);

} // namespace cuda_ops
