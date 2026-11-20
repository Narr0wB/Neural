
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


struct MaxPoolLayer : Layer<fp32> {

};


struct ReLULayer : Layer<fp32> {

};


struct DenseLayer : Layer<fp32> {

};

#endif // LAYER_HPP