
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include "tensor_view.hpp"

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstring>
#include <format>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#ifdef __GPU_ACCELERATION_CUDA
#include "cuda/cuda_ops.hpp"
#endif

namespace detail {
inline std::mt19937& rng()
{
    thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}
} // namespace detail


enum class Device {
    CPU,
    CUDA
};

inline constexpr std::string_view device_name(Device d)
{
    switch (d) {
        case Device::CPU:  return "CPU";
        case Device::CUDA: return "CUDA";
    }
    return "unknown";
}

// Copies are shallow: `Tensor b = a;` shares a's buffer, and so do slice(),
// reshape() and flatten(). Use clone() for an independent copy.
template <Numeric T>
class Tensor {
private:
    std::shared_ptr<T[]> data;
    Shape shape_;
    Device device_ = Device::CPU;

    Tensor(std::shared_ptr<T[]> buf, const Shape& s, Device dev) : data(std::move(buf)), shape_(s), device_(dev) {}

public:
    Tensor() = default;

    explicit Tensor(Device dev, const Shape& s) : shape_(s), device_(dev)
    {
        if (device_ == Device::CPU) {
            data = std::make_shared_for_overwrite<T[]>(s.size());
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            T* raw_ptr = nullptr;
            const size_t bytes = size() * sizeof(T);
            if (bytes > 0) {
                cuda_ops::init_mem_pool();
                CUDA_CHECK(cudaMallocAsync(&raw_ptr, bytes, 0));
                data = std::shared_ptr<T[]>(raw_ptr, [](T* p) { if (p) cudaFreeAsync(p, 0); });
            }
#else
            throw std::runtime_error("Tensor: CUDA support not compiled in");
#endif
        }
    }

    template <std::integral... D>
        requires(sizeof...(D) >= 1 && sizeof...(D) <= MAX_DIMS)
    explicit Tensor(Device dev, D... dims) : Tensor(dev, Shape{static_cast<size_t>(dims)...})
    {}

    explicit Tensor(const Shape& s) : Tensor(Device::CPU, s) {}

    template <std::integral... D>
        requires(sizeof...(D) >= 1 && sizeof...(D) <= MAX_DIMS)
    explicit Tensor(D... dims) : Tensor(Shape{static_cast<size_t>(dims)...})
    {}

    [[nodiscard]] static Tensor zeros(Device dev, const Shape& s)
    {
        Tensor out(dev, s);
        out.fill(T{0});
        return out;
    }

    [[nodiscard]] static Tensor zeros(const Shape& s)
    {
        return zeros(Device::CPU, s);
    }

    [[nodiscard]] static Tensor ones(Device dev, const Shape& s)
    {
        Tensor out(dev, s);
        out.fill(T{1});
        return out;
    }

    [[nodiscard]] static Tensor ones(const Shape& s)
    {
        return ones(Device::CPU, s);
    }

    [[nodiscard]] static Tensor randn(Device dev, const Shape& s) requires std::floating_point<T>
    {
        Tensor t(s);
        std::normal_distribution<T> dist(T{0}, T{1});
        for (size_t i = 0; i < t.size(); ++i)
            t.data[i] = dist(detail::rng());
        return t.to(dev);
    }

    [[nodiscard]] static Tensor randn(const Shape& s) requires std::floating_point<T>
    {
        return randn(Device::CPU, s);
    }

    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim; }
    size_t dim(size_t i) const { return shape_[i]; }
    size_t size() const { return shape_.size(); }
    size_t bytes() const { return shape_.size() * sizeof(T); }

    Device device() const { return device_; }

    T* raw() { return data.get(); }
    const T* raw() const { return data.get(); }

    template <std::integral... I>
    T& operator()(I... idx)
    {
        assert(device_ == Device::CPU);
        return data[shape_.offset(idx...)];
    }

    template <std::integral... I>
    const T& operator()(I... idx) const
    {
        assert(device_ == Device::CPU);
        return data[shape_.offset(idx...)];
    }

    [[nodiscard]] TensorView<T> view() { return {data.get(), shape_}; }
    [[nodiscard]] TensorView<const T> view() const { return {data.get(), shape_}; }
    [[nodiscard]] TensorView<const T> cview() const { return {data.get(), shape_}; }

    [[nodiscard]] Tensor<T> to(Device dev) const
    {
        if (dev == device_) return *this;
#ifdef __GPU_ACCELERATION_CUDA

        Tensor<T> out(dev, shape_);

        if (bytes() > 0) {
            switch (dev) {
                case Device::CPU:  CUDA_CHECK(cudaMemcpy(out.raw(), data.get(), bytes(), cudaMemcpyDeviceToHost)); break;
                case Device::CUDA: CUDA_CHECK(cudaMemcpy(out.raw(), data.get(), bytes(), cudaMemcpyHostToDevice)); break;
            }
        }

        return out;
#else
        throw std::runtime_error("Tensor::to: CUDA support not compiled in");
#endif
    }

    // Index i along the first dimension, dropping it: for {N, C, H, W},
    // slice(n).slice(c) is the H x W plane.
    [[nodiscard]] Tensor slice(size_t i) const
    {
        if (shape_.ndim < 2)
            throw std::invalid_argument("Tensor::slice: needs at least 2 dimensions, got " + shape_.str());
        if (i >= shape_[0])
            throw std::out_of_range("Tensor::slice: index " + std::to_string(i) + " out of range for " + shape_.str());

        Shape sub;
        sub.ndim = shape_.ndim - 1;
        for (size_t k = 0; k < sub.ndim; ++k)
            sub.dims[k] = shape_[k + 1];

        // Aliasing constructor: points into the buffer while sharing ownership
        // of all of it, so the parent stays alive as long as the slice does.
        return Tensor(std::shared_ptr<T[]>(data, data.get() + i * sub.size()), sub, device_);
    }

    [[nodiscard]] Tensor reshape(const Shape& s) const
    {
        if (s.size() != size())
            throw std::invalid_argument("Tensor::reshape: cannot reshape " + shape_.str() + " to " + s.str());
        return Tensor(data, s, device_);
    }

    [[nodiscard]] Tensor flatten() const { return reshape({size()}); }

    [[nodiscard]] Tensor clone() const
    {
        if (device_ == Device::CUDA) {
#ifdef __GPU_ACCELERATION_CUDA
            Tensor t(Device::CUDA, shape_);
            const size_t bytes = size() * sizeof(T);
            if (bytes > 0)
                CUDA_CHECK(cudaMemcpy(t.raw(), data.get(), bytes, cudaMemcpyDeviceToDevice));
            return t;
#else
            throw std::runtime_error("Tensor::clone: CUDA support not compiled in");
#endif
        }
        Tensor t(shape_);
        if (bytes() > 0)
            std::memcpy(t.raw(), data.get(), bytes());
        return t;
    }

    void fill(T value)
    {
        if (device_ == Device::CPU) {
            std::fill_n(data.get(), shape_.size(), value);
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            if (bytes() == 0) return;

            if constexpr (sizeof(T) == 1)
                CUDA_CHECK(cudaMemset(data.get(), static_cast<unsigned char>(value), bytes()));
            else if (value == T{0}) 
                CUDA_CHECK(cudaMemset(data.get(), static_cast<unsigned char>(0), bytes()));
            else
                cuda_ops::fill<T>(view(), value);
#else
            throw std::runtime_error("Tensor::fill: only supported on CPU tensors");
#endif
        }
    }

    template <typename F>
        requires std::is_invocable_r_v<T, F, T>
    void apply(F f)
    {
        if (device_ != Device::CPU)
            throw std::runtime_error("Tensor::apply: only supported on CPU tensors");
        for (size_t i = 0; i < size(); ++i)
            data[i] = f(data[i]);
    }
};

namespace detail {
template <Numeric T, Numeric... Ts>
void require_same_device(const char* op, const Tensor<T>& first, const Tensor<Ts>&... rest)
{
    auto check = [&](const auto& t) {
        if (t.device() != first.device())
            throw std::invalid_argument(std::string(op) + ": tensors are on different devices (" +
                                        std::string(device_name(first.device())) + " vs " +
                                        std::string(device_name(t.device())) + ")");
    };
    (check(rest), ...);
}

template <Numeric T>
void require_cpu(const char* op, const Tensor<T>& a)
{
    if (a.device() != Device::CPU)
        throw std::runtime_error(std::string(op) + ": CUDA backend not implemented yet");
}

template <Numeric T, Numeric... Ts>
void require_same_shape(const char* op, const Tensor<T>& first, const Tensor<Ts>&... rest)
{
    auto check = [&](const auto& t) {
        if (!(first.shape() == t.shape()))
            throw std::invalid_argument(std::string(op) + ": shapes " + first.shape().str() + " and " +
                                        t.shape().str() + " do not match");
    };
    (check(rest), ...);
}

template <Numeric T>
void require_2d(const char* op, const Tensor<T>& a)
{
    if (a.ndim() != 2)
        throw std::invalid_argument(std::string(op) + ": expected a 2D tensor, got " + a.shape().str());
}

// Unsafe: caller must verify matching devices (CPU) and shapes.
template <Numeric T, typename F>
void elementwise(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out, F f)
{
    const T* pa = a.raw();
    const T* pb = b.raw();
    T* po = out.raw();
    for (size_t i = 0; i < out.size(); ++i)
        po[i] = f(pa[i], pb[i]);
}

// Unsafe: caller must verify matching devices (CPU) and out.shape() == a.shape().broadcast_to(b.shape()).
template <Numeric T, typename F>
void elementwise_broadcast(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out, F f)
{
    const size_t n = out.size();
    if (n == 0)
        return;

    const Shape& sa = a.shape();
    const Shape& sb = b.shape();
    const Shape& so = out.shape();
    const size_t ndim = so.ndim;

    size_t a_stride[MAX_DIMS] = {};
    size_t b_stride[MAX_DIMS] = {};
    size_t sa_acc = 1, sb_acc = 1;
    for (size_t k = ndim; k-- > 0;) {
        if (k + sa.ndim >= ndim) {
            const size_t d = sa[k + sa.ndim - ndim];
            a_stride[k] = (d == 1) ? 0 : sa_acc;
            sa_acc *= d;
        }
        if (k + sb.ndim >= ndim) {
            const size_t d = sb[k + sb.ndim - ndim];
            b_stride[k] = (d == 1) ? 0 : sb_acc;
            sb_acc *= d;
        }
    }

    const T* pa = a.raw();
    const T* pb = b.raw();
    T* po = out.raw();

    const size_t inner  = so[ndim - 1];
    const size_t outer  = n / inner;
    const bool a_contig = (a_stride[ndim - 1] == 1);
    const bool b_contig = (b_stride[ndim - 1] == 1);

    for (size_t row = 0; row < outer; ++row) {
        size_t rem = row, ia = 0, ib = 0;
        for (size_t k = ndim - 1; k-- > 1;) {
            const size_t coord = rem % so[k];
            rem /= so[k];
            ia += coord * a_stride[k];
            ib += coord * b_stride[k];
        }
        if (ndim > 1) {
            ia += rem * a_stride[0];
            ib += rem * b_stride[0];
        }

        const T* pa_row = pa + ia;
        const T* pb_row = pb + ib;
        T*       po_row = po + row * inner;

        if (a_contig && b_contig) {
            for (size_t j = 0; j < inner; ++j)
                po_row[j] = f(pa_row[j], pb_row[j]);
        } else if (a_contig) {
            const T bv = *pb_row;
            for (size_t j = 0; j < inner; ++j)
                po_row[j] = f(pa_row[j], bv);
        } else if (b_contig) {
            const T av = *pa_row;
            for (size_t j = 0; j < inner; ++j)
                po_row[j] = f(av, pb_row[j]);
        } else {
            std::fill_n(po_row, inner, f(*pa_row, *pb_row));
        }
    }
}
} // namespace detail

template <Numeric T>
void add_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out)
{
    detail::require_same_device("add_into", a, b, out);

    const bool same = (a.shape() == b.shape()) && (b.shape() == out.shape());
    if (!same && !(a.shape().broadcast_to(b.shape()) == out.shape()))
        throw std::invalid_argument("add_into: output shape " + out.shape().str() +
                                    " does not match broadcast of " + a.shape().str() + " and " + b.shape().str());

    if (a.device() == Device::CPU) {
        auto op = [](T x, T y) { return x + y; };
        if (same) detail::elementwise(a, b, out, op);
        else      detail::elementwise_broadcast(a, b, out, op);
    } else {
#ifdef __GPU_ACCELERATION_CUDA
        cuda_ops::add(a.view(), b.view(), out.view());
#else
        detail::require_cpu("add_into", a);
#endif
    }
}

template <Numeric T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b)
{
    Tensor<T> out(a.device(), a.shape().broadcast_to(b.shape()));
    add_into(a, b, out);
    return out;
}

template <Numeric T>
void sub_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out)
{
    detail::require_same_device("sub_into", a, b, out);

    const bool same = (a.shape() == b.shape()) && (b.shape() == out.shape());
    if (!same && !(a.shape().broadcast_to(b.shape()) == out.shape()))
        throw std::invalid_argument("sub_into: output shape " + out.shape().str() +
                                    " does not match broadcast of " + a.shape().str() + " and " + b.shape().str());

    if (a.device() == Device::CPU) {
        auto op = [](T x, T y) { return x - y; };
        if (same) detail::elementwise(a, b, out, op);
        else      detail::elementwise_broadcast(a, b, out, op);
    } else {
#ifdef __GPU_ACCELERATION_CUDA
        cuda_ops::sub(a.view(), b.view(), out.view());
#else
        detail::require_cpu("sub_into", a);
#endif
    }
}

template <Numeric T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b)
{
    Tensor<T> out(a.device(), a.shape().broadcast_to(b.shape()));
    sub_into(a, b, out);
    return out;
}

template <Numeric T>
void hadamard_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out)
{
    detail::require_same_device("hadamard_into", a, b, out);
    detail::require_cpu("hadamard_into", a);

    const bool same = (a.shape() == b.shape()) && (b.shape() == out.shape());
    if (!same && !(a.shape().broadcast_to(b.shape()) == out.shape()))
        throw std::invalid_argument("hadamard_into: output shape " + out.shape().str() +
                                    " does not match broadcast of " + a.shape().str() + " and " + b.shape().str());

    auto op = [](T x, T y) { return x * y; };
    if (same) detail::elementwise(a, b, out, op);
    else      detail::elementwise_broadcast(a, b, out, op);
}

template <Numeric T>
Tensor<T> hadamard(const Tensor<T>& a, const Tensor<T>& b)
{
    Tensor<T> out(a.device(), a.shape().broadcast_to(b.shape()));
    hadamard_into(a, b, out);
    return out;
}

template <Numeric T>
void scale_into(const Tensor<T>& a, T s, Tensor<T>& out)
{
    detail::require_same_device("scale_into", a, out);
    detail::require_same_shape("scale_into", a, out);
    detail::require_cpu("scale_into", a);

    const T* pa = a.raw();
    T* po = out.raw();
    for (size_t i = 0; i < a.size(); ++i)
        po[i] = pa[i] * s;
}

template <Numeric T>
Tensor<T> scale(const Tensor<T>& a, T s)
{
    Tensor<T> out(a.device(), a.shape());
    scale_into(a, s, out);
    return out;
}

template <ReduceOp op = ReduceOp::SUM, Numeric T>
void reduce_into(const Tensor<T>& x, size_t axis, Tensor<T>& out)
{
    detail::require_same_device("reduce_into", x, out);

    if (x.device() == Device::CUDA) {
#ifdef __GPU_ACCELERATION_CUDA
        cuda_ops::reduce<op>(x.view(), axis, out.view());
        return;
#else
        detail::require_cpu("reduce_into", x);
#endif
    }

    const Shape in_shape = x.shape();
    if (axis >= in_shape.ndim)
        throw std::invalid_argument("reduce_into: invalid axis");

    size_t outer_stride = 1;
    size_t inner_stride = 1;
    for (size_t i = 0; i < axis; i++)
        outer_stride *= in_shape[i];
    for (size_t i = axis + 1; i < in_shape.ndim; i++)
        inner_stride *= in_shape[i];

    if (out.size() != outer_stride * inner_stride)
        throw std::invalid_argument("reduce_into: invalid output shape");

    const size_t axis_len = in_shape[axis];
    const T* px = x.raw();
    T*       po = out.raw();

    if (inner_stride == 1) {
        for (size_t i = 0; i < outer_stride; ++i) {
            const T* px_row = px + i * axis_len;
            T acc = T{0};
            for (size_t k = 0; k < axis_len; ++k)
                acc += px_row[k];
            if constexpr (op == ReduceOp::SUM)
                po[i] += acc;
            else if constexpr (op == ReduceOp::SUB)
                po[i] -= acc;
        }
    } else {
        for (size_t i = 0; i < outer_stride; ++i) {
            T* po_row = po + i * inner_stride;
            for (size_t k = 0; k < axis_len; ++k) {
                const T* px_row = px + (i * axis_len + k) * inner_stride;
                for (size_t j = 0; j < inner_stride; ++j) {
                    if constexpr (op == ReduceOp::SUM)
                        po_row[j] += px_row[j];
                    else if constexpr (op == ReduceOp::SUB)
                        po_row[j] -= px_row[j];
                }
            }
        }
    }
}


template <ReduceOp op = ReduceOp::SUM, Numeric T>
Tensor<T> reduce(const Tensor<T>& x, size_t axis)
{
    Tensor<T> out = Tensor<T>::zeros(x.device(), x.shape().reduce(axis));
    reduce_into<op>(x, axis, out);
    return out;
}


template <Numeric T>
size_t argmax(const Tensor<T>& a)
{
    detail::require_cpu("argmax", a);
    if (a.size() == 0)
        throw std::invalid_argument("argmax: tensor is empty");
    const T* p = a.raw();
    size_t best = 0;
    for (size_t i = 1; i < a.size(); ++i)
        if (p[i] > p[best])
            best = i;
    return best;
}

// No tensor * tensor operator on purpose: use matmul() or hadamard().
template <Numeric T>
Tensor<T> operator+(const Tensor<T>& a, const Tensor<T>& b) { return add(a, b); }

template <Numeric T>
Tensor<T> operator-(const Tensor<T>& a, const Tensor<T>& b) { return sub(a, b); }

template <Numeric T>
Tensor<T> operator*(const Tensor<T>& a, T s) { return scale(a, s); }

template <Numeric T>
Tensor<T> operator*(T s, const Tensor<T>& a) { return scale(a, s); }

// @out must not alias @a or @b.
template <Numeric T>
void matmul_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out,
                 bool trans_a = false, bool trans_b = false, bool accumulate = true)
{
    detail::require_same_device("matmul_into", a, b, out);

    if (a.device() == Device::CUDA) {
#ifdef __GPU_ACCELERATION_CUDA
        cuda_ops::matmul(a.view(), b.view(), out.view(), trans_a, trans_b, accumulate);
        return;
#else
        detail::require_cpu("matmul_into", a);
#endif
    }

    const size_t ndim = a.ndim();
    if (ndim < 2 || b.ndim() != ndim || out.ndim() != ndim)
        throw std::invalid_argument("matmul_into: inputs and output must have matching rank >= 2");

    size_t batch = 1;
    for (size_t i = 0; i + 2 < ndim; ++i) {
        if (a.dim(i) != b.dim(i) || a.dim(i) != out.dim(i))
            throw std::invalid_argument("matmul_into: batch dimensions of " + a.shape().str() + ", " +
                                        b.shape().str() + " and " + out.shape().str() + " do not match");
        batch *= a.dim(i);
    }

    const size_t M   = trans_a ? a.dim(ndim - 1) : a.dim(ndim - 2);
    const size_t K   = trans_a ? a.dim(ndim - 2) : a.dim(ndim - 1);
    const size_t K_b = trans_b ? b.dim(ndim - 1) : b.dim(ndim - 2);
    const size_t N   = trans_b ? b.dim(ndim - 2) : b.dim(ndim - 1);
    if (K != K_b)
        throw std::invalid_argument("matmul_into: inner dimensions of " + a.shape().str() + " and " +
                                    b.shape().str() + " do not match");
    if (out.dim(ndim - 2) != M || out.dim(ndim - 1) != N)
        throw std::invalid_argument("matmul_into: output matrix shape does not match {..., " +
                                    std::to_string(M) + ", " + std::to_string(N) + "}, got " + out.shape().str());

    if (batch == 0 || M == 0 || K == 0 || N == 0)
        return;

    const size_t a_stride = M * K;
    const size_t b_stride = K * N;
    const size_t c_stride = M * N;

    if (!trans_b) {
        const size_t a_rs = trans_a ? 1 : K, a_cs = trans_a ? M : 1;
        // i-k-j order keeps the inner loop unit-stride in both out and b for SIMD vectorization.
        #pragma omp parallel for collapse(2) if(batch * M * K * N >= 200000)
        for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            for (size_t i = 0; i < M; ++i) {
                const T* pa     = a.raw()   + batch_idx * a_stride;
                const T* pb     = b.raw()   + batch_idx * b_stride;
                T*       po_row = out.raw() + batch_idx * c_stride + i * N;
                if (!accumulate)
                    std::fill_n(po_row, N, T{0});
                for (size_t p = 0; p < K; ++p) {
                    const T  a_ip   = pa[i * a_rs + p * a_cs];
                    const T* pb_row = pb + p * N;
                    for (size_t j = 0; j < N; ++j)
                        po_row[j] += a_ip * pb_row[j];
                }
            }
        }
    } else if (!trans_a) {
        // i-j-k dot product keeps both a and b^T unit-stride in the inner loop.
        #pragma omp parallel for collapse(2) if(batch * M * K * N >= 200000)
        for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            for (size_t i = 0; i < M; ++i) {
                const T* pa_row = a.raw()   + batch_idx * a_stride + i * K;
                const T* pb     = b.raw()   + batch_idx * b_stride;
                T*       po_row = out.raw() + batch_idx * c_stride + i * N;
                for (size_t j = 0; j < N; ++j) {
                    const T* pb_row = pb + j * K;
                    T acc = accumulate ? po_row[j] : T{0};
                    for (size_t p = 0; p < K; ++p)
                        acc += pa_row[p] * pb_row[p];
                    po_row[j] = acc;
                }
            }
        }
    } else {
        // Both transposed: i-j-k keeps b^T unit-stride in the inner loop.
        #pragma omp parallel for collapse(2) if(batch * M * K * N >= 200000)
        for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            for (size_t i = 0; i < M; ++i) {
                const T* pa     = a.raw()   + batch_idx * a_stride;
                const T* pb     = b.raw()   + batch_idx * b_stride;
                T*       po_row = out.raw() + batch_idx * c_stride + i * N;
                for (size_t j = 0; j < N; ++j) {
                    const T* pb_row = pb + j * K;
                    T acc = accumulate ? po_row[j] : T{0};
                    for (size_t p = 0; p < K; ++p)
                        acc += pa[p * M + i] * pb_row[p];
                    po_row[j] = acc;
                }
            }
        }
    }
}

template <Numeric T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b,
                 bool trans_a = false, bool trans_b = false)
{
    const size_t ndim = a.ndim();
    if (ndim < 2 || b.ndim() != ndim)
        throw std::invalid_argument("matmul: inputs must have matching rank >= 2");

    Shape out_shape = a.shape();
    out_shape.dims[ndim - 2] = trans_a ? a.dim(ndim - 1) : a.dim(ndim - 2);
    out_shape.dims[ndim - 1] = trans_b ? b.dim(ndim - 2) : b.dim(ndim - 1);

    Tensor<T> out(a.device(), out_shape);
    matmul_into(a, b, out, trans_a, trans_b, /*accumulate=*/false);
    return out;
}

// Writes a^T into @out, swapping the last two dimensions: {..., R, C} -> {..., C, R}.
// @out must not alias @a.
template <Numeric T>
void transpose_into(const Tensor<T>& a, Tensor<T>& out)
{
    detail::require_same_device("transpose_into", a, out);

    const size_t ndim = a.ndim();
    if (ndim < 2 || out.ndim() != ndim)
        throw std::invalid_argument("transpose_into: input and output must have matching rank >= 2");

    size_t batch = 1;
    for (size_t i = 0; i + 2 < ndim; ++i) {
        if (a.dim(i) != out.dim(i))
            throw std::invalid_argument("transpose_into: batch dimensions of " + a.shape().str() +
                                        " and " + out.shape().str() + " do not match");
        batch *= a.dim(i);
    }

    const size_t R = a.dim(ndim - 2);
    const size_t C = a.dim(ndim - 1);
    if (out.dim(ndim - 2) != C || out.dim(ndim - 1) != R)
        throw std::invalid_argument("transpose_into: output matrix shape does not match {..., " +
                                    std::to_string(C) + ", " + std::to_string(R) + "}, got " + out.shape().str());

    if (batch == 0 || R == 0 || C == 0)
        return;

    if (a.device() == Device::CPU) {
        const size_t stride = R * C;
        for (size_t batch_idx = 0; batch_idx < batch; ++batch_idx) {
            const T* pa = a.raw()   + batch_idx * stride;
            T*       po = out.raw() + batch_idx * stride;
            for (size_t i = 0; i < R; ++i)
                for (size_t j = 0; j < C; ++j)
                    po[j * R + i] = pa[i * C + j];
        }
    }
    else {
#ifdef __GPU_ACCELERATION_CUDA
        cuda_ops::transpose(a.view(), out.view());
#else
        detail::require_cpu("transpose_into", a);
#endif
    }
}

template <Numeric T>
Tensor<T> transpose(const Tensor<T>& a)
{
    const size_t ndim = a.ndim();
    if (ndim < 2)
        throw std::invalid_argument("transpose: expected tensor of rank >= 2, got " + a.shape().str());

    Shape out_shape = a.shape();
    std::swap(out_shape.dims[ndim - 2], out_shape.dims[ndim - 1]);

    Tensor<T> out(a.device(), out_shape);
    transpose_into(a, out);
    return out;
}

#endif // TENSOR_HPP
