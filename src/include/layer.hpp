
#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

template <Numeric T>
struct Param {
    Tensor<T> value;
    Tensor<T> grad;
};

template <Numeric T>
struct Layer {
    virtual Tensor<T> forward(const Tensor<T>& x) = 0;
    virtual Tensor<T> backward(const Tensor<T>& dY) = 0;

    virtual std::vector<Param<T>> params() { return {}; }  // {value, grad} pairs for the optimizer
    virtual ~Layer() = default;
};


// 2D convolution, stride 1, no padding ("valid").
//   x       : {N, C_in,  H,         W        }
//   y       : {N, C_out, H - K + 1, W - K + 1}
//   filters : {C_out, C_in, K, K}
//   bias    : {C_out}  (one scalar per output channel)
struct ConvLayer : Layer<fp32> {
    Tensor<fp32> filters;
    Tensor<fp32> filters_grad;
    Tensor<fp32> bias;
    Tensor<fp32> bias_grad;

    ConvLayer(size_t in_channels, size_t out_channels, size_t kernel_size)
    {
        if (in_channels == 0 || out_channels == 0 || kernel_size == 0)
            throw std::invalid_argument("ConvLayer: channels and kernel size must be non-zero");

        const Shape fs{out_channels, in_channels, kernel_size, kernel_size};

        // He initialisation: N(0, 2 / fan_in), fan_in = C_in * K * K.
        const fp32 stddev = std::sqrt(2.0f / static_cast<fp32>(in_channels * kernel_size * kernel_size));
        filters = Tensor<fp32>::randn(fs);
        filters.apply([stddev](fp32 v) { return v * stddev; });
        filters_grad = Tensor<fp32>(fs);

        bias      = Tensor<fp32>(Shape{out_channels});
        bias_grad = Tensor<fp32>(Shape{out_channels});
    }

    size_t in_channels() const { return filters.dim(1); }
    size_t out_channels() const { return filters.dim(0); }
    size_t kernel_size() const { return filters.dim(2); }

    // y[n, o, i, j] = bias[o] + sum_{c, u, v} x[n, c, i+u, j+v] * filters[o, c, u, v]
    // Caches x for backward().
    Tensor<fp32> forward(const Tensor<fp32>& x) override;

    // Given dY = dL/dy, accumulates (+=):
    //   filters_grad[o, c, u, v] += sum_{n, i, j} dY[n, o, i, j] * x[n, c, i+u, j+v]
    //   bias_grad[o]             += sum_{n, i, j} dY[n, o, i, j]
    // and returns dX = dL/dx, same shape as x:
    //   dX[n, c, p, q] = sum_{o, u, v} dY[n, o, p-u, q-v] * filters[o, c, u, v]   (out-of-range dY terms are 0)
    Tensor<fp32> backward(const Tensor<fp32>& dY) override;

    std::vector<Param<fp32>> params() override { return {{filters, filters_grad}, {bias, bias_grad}}; }

private:
    Tensor<fp32> input; // cached by forward()
};


// 2D max pooling, non-overlapping P x P windows (stride = P, floor division).
//   x : {N, C, H,     W    }
//   y : {N, C, H / P, W / P}
struct MaxPoolLayer : Layer<fp32> {
    explicit MaxPoolLayer(size_t pool_size = 2) : pool_size_(pool_size)
    {
        if (pool_size == 0)
            throw std::invalid_argument("MaxPoolLayer: pool size must be non-zero");
    }

    size_t pool_size() const { return pool_size_; }

    // y[n, c, i, j] = max_{0 <= u, v < P} x[n, c, i*P + u, j*P + v]
    // Caches input_shape and winning positions in max_indices for backward().
    Tensor<fp32> forward(const Tensor<fp32>& x) override;

    // Routes each dY[n, c, i, j] back to the winning element of its P x P window
    // in dX (same shape as input_shape); all other entries in dX are 0.
    Tensor<fp32> backward(const Tensor<fp32>& dY) override;

private:
    size_t pool_size_;
    Shape input_shape;          // cached by forward() (needed when H or W isn't a multiple of P)
    Tensor<size_t> max_indices; // {N, C, H/P, W/P}: winning index per pooling window
};


// Elementwise ReLU activation (works on any tensor shape).
//   y[k] = max(0, x[k])
struct ReLULayer : Layer<fp32> {
    // y[k] = x[k] > 0 ? x[k] : 0
    // Caches x for backward().
    Tensor<fp32> forward(const Tensor<fp32>& x) override;

    // dX[k] = input[k] > 0 ? dY[k] : 0
    Tensor<fp32> backward(const Tensor<fp32>& dY) override;

private:
    Tensor<fp32> input; // cached by forward()
};


// Simple matmul + bias
//  x :       {N, A}
//  y :       {N, B}
//  weights : {B, A}
//  bias:     {B}
struct DenseLayer : Layer<fp32> {
    Tensor<fp32> weights;
    Tensor<fp32> weights_grad;
    Tensor<fp32> bias;
    Tensor<fp32> bias_grad;

    size_t input_shape() const { return weights.shape()[1]; }
    size_t output_shape() const { return weights.shape()[0]; }

    // y[n, i] = bias[i] + sum_{j} weighs[i, j] * x[j]
    Tensor<fp32> forward(const Tensor<fp32>& x) override;

    Tensor<fp32> backward(const Tensor<fp32>& dY) override;

private:
    Tensor<fp32> input; // cached by forward()
};

#endif // LAYER_HPP