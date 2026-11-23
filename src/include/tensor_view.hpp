
#ifndef TENSOR_VIEW_HPP
#define TENSOR_VIEW_HPP

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef __GPU_ACCELERATION_CUDA
#include <cuda_runtime.h>

// Kernel launches return nothing: call CUDA_CHECK(cudaGetLastError()) after one.
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err_ = (call);                                              \
        if (err_ != cudaSuccess)                                                \
            throw std::runtime_error(std::string("CUDA error: ") +              \
                                     cudaGetErrorString(err_) + " at " +        \
                                     __FILE__ + ":" + std::to_string(__LINE__)); \
    } while (0)
#endif

// Only defined as __host__ __device__ when compiled as CUDA by nvcc.
#ifdef __CUDACC__
#define NN_HOST_DEVICE __host__ __device__
#else
#define NN_HOST_DEVICE
#endif

template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

typedef float    fp32;
typedef double   fp64;

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t   i8;
typedef int16_t  i16;
typedef int32_t  i32;
typedef int64_t  i64;

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

    TensorView() = default;
    NN_HOST_DEVICE TensorView(T* d, Shape s) : data(d), shape(s) {}

    template <typename U>
        requires std::is_const_v<T> && std::same_as<std::remove_const_t<T>, U>
    NN_HOST_DEVICE TensorView(const TensorView<U>& other) : data(other.data), shape(other.shape) {}

    NN_HOST_DEVICE size_t size() const { return shape.size(); }

    template <std::integral... I>
    NN_HOST_DEVICE T& operator()(I... idx) const { return data[shape.offset(idx...)]; }
};

#endif // TENSOR_VIEW_HPP
