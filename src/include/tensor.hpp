
#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <cassert>
#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#ifdef __GPU_ACCELERATION_CUDA
#include <cuda_runtime.h>
#endif

// Only defined as __host__ __device__ when compiled as CUDA by nvcc.
#ifdef __CUDACC__
#define NN_HOST_DEVICE __host__ __device__
#else
#define NN_HOST_DEVICE
#endif

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

typedef float  fp32;
typedef double fp64;

inline constexpr size_t MAX_DIMS = 4;

// Plain array instead of std::array so it can be used inside CUDA kernels.
struct Shape {
    size_t dims[MAX_DIMS] = {};
    size_t ndim = 0;

    Shape() = default;

    Shape(std::initializer_list<size_t> d)
    {
        if (d.size() > MAX_DIMS)
            throw std::invalid_argument("Shape: at most " + std::to_string(MAX_DIMS) + " dimensions");
        for (size_t v : d)
            dims[ndim++] = v;
    }

    NN_HOST_DEVICE size_t operator[](size_t i) const { return dims[i]; }

    NN_HOST_DEVICE size_t size() const
    {
        if (ndim == 0)
            return 0;
        size_t n = 1;
        for (size_t i = 0; i < ndim; ++i)
            n *= dims[i];
        return n;
    }

    // Row-major: ((i0 * d1 + i1) * d2 + i2) * d3 + i3.
    template <std::integral... I>
    NN_HOST_DEVICE size_t offset(I... idx) const
    {
        assert(sizeof...(I) == ndim);
        size_t o = 0, k = 0;
        ((o = o * dims[k++] + static_cast<size_t>(idx)), ...);
        return o;
    }

    NN_HOST_DEVICE bool operator==(const Shape& o) const
    {
        if (ndim != o.ndim)
            return false;
        for (size_t i = 0; i < ndim; ++i)
            if (dims[i] != o.dims[i])
                return false;
        return true;
    }

    std::string str() const
    {
        std::string s = "[";
        for (size_t i = 0; i < ndim; ++i)
            s += (i ? ", " : "") + std::to_string(dims[i]);
        return s + "]";
    }
};

// Must not outlive the tensor it came from.
template <Numeric T>
struct TensorView {
    T* data = nullptr;
    Shape shape;

    NN_HOST_DEVICE size_t size() const { return shape.size(); }

    template <std::integral... I>
    NN_HOST_DEVICE T& operator()(I... idx) const { return data[shape.offset(idx...)]; }
};

namespace detail {
inline std::mt19937& rng()
{
    thread_local std::mt19937 gen{std::random_device{}()};
    return gen;
}
} // namespace detail

// Copies are shallow: `Tensor b = a;` shares a's buffer, and so do slice(),
// reshape() and flatten(). Use clone() for an independent copy.
template <Numeric T>
class Tensor {
private:
    std::shared_ptr<T[]> data;
    Shape shape_;

    Tensor(std::shared_ptr<T[]> buf, const Shape& s) : data(std::move(buf)), shape_(s) {}

public:
    Tensor() = default;

    // Zero-filled.
    explicit Tensor(const Shape& s) : data(std::make_shared<T[]>(s.size())), shape_(s) {}

    template <std::integral... D>
        requires(sizeof...(D) >= 1 && sizeof...(D) <= MAX_DIMS)
    explicit Tensor(D... dims) : Tensor(Shape{static_cast<size_t>(dims)...})
    {
    }

    static Tensor zeros(const Shape& s) { return Tensor(s); }

    static Tensor ones(const Shape& s)
    {
        Tensor t(s);
        t.fill(T{1});
        return t;
    }

    static Tensor randn(const Shape& s) requires std::floating_point<T>
    {
        Tensor t(s);
        std::normal_distribution<T> dist(T{0}, T{1});
        for (size_t i = 0; i < t.size(); ++i)
            t.data[i] = dist(detail::rng());
        return t;
    }

    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim; }
    size_t dim(size_t i) const { return shape_[i]; }
    size_t size() const { return shape_.size(); }

    T* raw() { return data.get(); }
    const T* raw() const { return data.get(); }

    template <std::integral... I>
    T& operator()(I... idx) { return data[shape_.offset(idx...)]; }

    template <std::integral... I>
    const T& operator()(I... idx) const { return data[shape_.offset(idx...)]; }

    TensorView<T> view() { return {data.get(), shape_}; }
    TensorView<const T> view() const { return {data.get(), shape_}; }

    // Index i along the first dimension, dropping it: for {N, C, H, W},
    // slice(n).slice(c) is the H x W plane.
    Tensor slice(size_t i) const
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
        return Tensor(std::shared_ptr<T[]>(data, data.get() + i * sub.size()), sub);
    }

    Tensor reshape(const Shape& s) const
    {
        if (s.size() != size())
            throw std::invalid_argument("Tensor::reshape: cannot reshape " + shape_.str() + " to " + s.str());
        return Tensor(data, s);
    }

    Tensor flatten() const { return reshape({size()}); }

    Tensor clone() const
    {
        Tensor t(shape_);
        for (size_t i = 0; i < size(); ++i)
            t.data[i] = data[i];
        return t;
    }

    void fill(T value)
    {
        for (size_t i = 0; i < size(); ++i)
            data[i] = value;
    }

    template <typename F>
        requires std::is_invocable_r_v<T, F, T>
    void apply(F f)
    {
        for (size_t i = 0; i < size(); ++i)
            data[i] = f(data[i]);
    }
};

namespace detail {
template <Numeric T>
void require_same_shape(const Tensor<T>& a, const Tensor<T>& b, const char* op)
{
    if (!(a.shape() == b.shape()))
        throw std::invalid_argument(std::string(op) + ": shapes " + a.shape().str() + " and " +
                                    b.shape().str() + " do not match");
}

template <Numeric T>
void require_2d(const Tensor<T>& a, const char* op)
{
    if (a.ndim() != 2)
        throw std::invalid_argument(std::string(op) + ": expected a 2D tensor, got " + a.shape().str());
}

template <Numeric T, typename F>
void elementwise(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out, const char* op, F f)
{
    require_same_shape(a, b, op);
    require_same_shape(a, out, op);
    const T* pa = a.raw();
    const T* pb = b.raw();
    T* po = out.raw();
    for (size_t i = 0; i < a.size(); ++i)
        po[i] = f(pa[i], pb[i]);
}
} // namespace detail

template <Numeric T>
void add_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out)
{
    detail::elementwise(a, b, out, "add_into", [](T x, T y) { return x + y; });
}

template <Numeric T>
Tensor<T> add(const Tensor<T>& a, const Tensor<T>& b)
{
    detail::require_same_shape(a, b, "add");
    Tensor<T> out(a.shape());
    add_into(a, b, out);
    return out;
}

template <Numeric T>
void sub_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out)
{
    detail::elementwise(a, b, out, "sub_into", [](T x, T y) { return x - y; });
}

template <Numeric T>
Tensor<T> sub(const Tensor<T>& a, const Tensor<T>& b)
{
    detail::require_same_shape(a, b, "sub");
    Tensor<T> out(a.shape());
    sub_into(a, b, out);
    return out;
}

template <Numeric T>
void hadamard_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out)
{
    detail::elementwise(a, b, out, "hadamard_into", [](T x, T y) { return x * y; });
}

template <Numeric T>
Tensor<T> hadamard(const Tensor<T>& a, const Tensor<T>& b)
{
    detail::require_same_shape(a, b, "hadamard");
    Tensor<T> out(a.shape());
    hadamard_into(a, b, out);
    return out;
}

template <Numeric T>
void scale_into(const Tensor<T>& a, T s, Tensor<T>& out)
{
    detail::require_same_shape(a, out, "scale_into");
    const T* pa = a.raw();
    T* po = out.raw();
    for (size_t i = 0; i < a.size(); ++i)
        po[i] = pa[i] * s;
}

template <Numeric T>
Tensor<T> scale(const Tensor<T>& a, T s)
{
    Tensor<T> out(a.shape());
    scale_into(a, s, out);
    return out;
}

template <Numeric T>
T sum(const Tensor<T>& a)
{
    T total{};
    for (size_t i = 0; i < a.size(); ++i)
        total += a.raw()[i];
    return total;
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

// Results are accumulated (+=) into @out. @out must not alias @a or @b.
template <Numeric T>
void matmul_into(const Tensor<T>& a, const Tensor<T>& b, Tensor<T>& out)
{
    detail::require_2d(a, "matmul_into");
    detail::require_2d(b, "matmul_into");
    detail::require_2d(out, "matmul_into");
    const size_t n = a.dim(0), k = a.dim(1), m = b.dim(1);
    if (k != b.dim(0))
        throw std::invalid_argument("matmul_into: inner dimensions of " + a.shape().str() + " and " +
                                    b.shape().str() + " do not match");
    if (!(out.shape() == Shape{n, m}))
        throw std::invalid_argument("matmul_into: expected output shape " + Shape{n, m}.str() +
                                    ", got " + out.shape().str());

    const T* pa = a.raw();
    const T* pb = b.raw();
    T* po = out.raw();
    // i-k-j order keeps the inner loop on contiguous memory in b and out.
    for (size_t i = 0; i < n; ++i)
        for (size_t p = 0; p < k; ++p) {
            const T a_ip = pa[i * k + p];
            for (size_t j = 0; j < m; ++j)
                po[i * m + j] += a_ip * pb[p * m + j];
        }
}

template <Numeric T>
Tensor<T> matmul(const Tensor<T>& a, const Tensor<T>& b)
{
    detail::require_2d(a, "matmul");
    detail::require_2d(b, "matmul");
    const size_t n = a.dim(0), k = a.dim(1), m = b.dim(1);
    if (k != b.dim(0))
        throw std::invalid_argument("matmul: inner dimensions of " + a.shape().str() + " and " +
                                    b.shape().str() + " do not match");

    Tensor<T> out(n, m);
    matmul_into(a, b, out);
    return out;
}

// Writes a^T into @out. @out must not alias @a.
template <Numeric T>
void transpose_into(const Tensor<T>& a, Tensor<T>& out)
{
    detail::require_2d(a, "transpose_into");
    const size_t r = a.dim(0), c = a.dim(1);
    if (!(out.shape() == Shape{c, r}))
        throw std::invalid_argument("transpose_into: expected output shape " + Shape{c, r}.str() +
                                    ", got " + out.shape().str());

    const T* pa = a.raw();
    T* po = out.raw();
    for (size_t i = 0; i < r; ++i)
        for (size_t j = 0; j < c; ++j)
            po[j * r + i] = pa[i * c + j];
}

template <Numeric T>
Tensor<T> transpose(const Tensor<T>& a)
{
    detail::require_2d(a, "transpose");
    Tensor<T> out(a.dim(1), a.dim(0));
    transpose_into(a, out);
    return out;
}

#ifdef __GPU_ACCELERATION_CUDA

// Kernel launches return nothing: call CUDA_CHECK(cudaGetLastError()) after one.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(err_) + " at " +        \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)

// Pass view() to kernels, never the DeviceTensor itself.
template <Numeric T>
class DeviceTensor {
private:
    T* data_ = nullptr;
    Shape shape_;

    size_t bytes() const { return shape_.size() * sizeof(T); }

    void release()
    {
        // Destructors must not throw, so errors from cudaFree are ignored.
        if (data_)
            cudaFree(data_);
        data_ = nullptr;
    }

public:
    DeviceTensor() = default;

    // Uninitialised, unlike Tensor. Use zeros() for a zero-filled one.
    explicit DeviceTensor(const Shape& s) : shape_(s)
    {
        if (bytes() > 0)
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), bytes()));
    }

    ~DeviceTensor() { release(); }

    DeviceTensor(const DeviceTensor&) = delete;
    DeviceTensor& operator=(const DeviceTensor&) = delete;

    DeviceTensor(DeviceTensor&& o) noexcept
        : data_(std::exchange(o.data_, nullptr)), shape_(std::exchange(o.shape_, Shape{}))
    {
    }

    DeviceTensor& operator=(DeviceTensor&& o) noexcept
    {
        if (this != &o) {
            release();
            data_ = std::exchange(o.data_, nullptr);
            shape_ = std::exchange(o.shape_, Shape{});
        }
        return *this;
    }

    static DeviceTensor zeros(const Shape& s)
    {
        DeviceTensor d(s);
        if (d.bytes() > 0)
            CUDA_CHECK(cudaMemset(d.data_, 0, d.bytes()));
        return d;
    }

    static DeviceTensor from_host(const Tensor<T>& host)
    {
        DeviceTensor d(host.shape());
        d.copy_from_host(host);
        return d;
    }

    void copy_from_host(const Tensor<T>& host)
    {
        if (!(host.shape() == shape_))
            throw std::invalid_argument("DeviceTensor::copy_from_host: shapes " + host.shape().str() +
                                        " and " + shape_.str() + " do not match");
        if (bytes() > 0)
            CUDA_CHECK(cudaMemcpy(data_, host.raw(), bytes(), cudaMemcpyHostToDevice));
    }

    Tensor<T> to_host() const
    {
        Tensor<T> h(shape_);
        if (bytes() > 0)
            CUDA_CHECK(cudaMemcpy(h.raw(), data_, bytes(), cudaMemcpyDeviceToHost));
        return h;
    }

    DeviceTensor clone() const
    {
        DeviceTensor d(shape_);
        if (bytes() > 0)
            CUDA_CHECK(cudaMemcpy(d.data_, data_, bytes(), cudaMemcpyDeviceToDevice));
        return d;
    }

    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim; }
    size_t dim(size_t i) const { return shape_[i]; }
    size_t size() const { return shape_.size(); }

    T* data() { return data_; }
    const T* data() const { return data_; }

    TensorView<T> view() { return {data_, shape_}; }
    TensorView<const T> view() const { return {data_, shape_}; }

    TensorView<T> view(const Shape& s)
    {
        if (s.size() != size())
            throw std::invalid_argument("DeviceTensor::view: cannot view " + shape_.str() + " as " + s.str());
        return {data_, s};
    }
};

#endif // __GPU_ACCELERATION_CUDA

#endif // TENSOR_HPP
