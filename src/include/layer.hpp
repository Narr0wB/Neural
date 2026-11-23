
#ifndef LAYER_HPP
#define LAYER_HPP

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <concepts>
#include <stdexcept>
#include <vector>

template <std::floating_point T = fp32>
struct Param {
    Tensor<T> value;
    Tensor<T> grad;
};

template <std::floating_point T = fp32>
struct Layer {
    virtual Tensor<T> forward(const Tensor<T>& x) = 0;
    virtual Tensor<T> backward(const Tensor<T>& dY) = 0;

    virtual void to(Device) {}
    virtual std::vector<Param<T>> params() { return {}; }
    virtual ~Layer() = default;
};


// 2D convolution, stride 1, no padding ("valid").
//   x       : {N, C_in,  H,         W        }
//   y       : {N, C_out, H - K + 1, W - K + 1}
//   filters : {C_out, C_in, K, K}
//   bias    : {C_out}  (one scalar per output channel)
template <std::floating_point T = fp32>
struct ConvLayer : Layer<T> {
    Tensor<T> filters;
    Tensor<T> filters_grad;
    Tensor<T> bias;
    Tensor<T> bias_grad;

    ConvLayer(size_t in_channels, size_t out_channels, size_t kernel_size)
    {
        if (in_channels == 0 || out_channels == 0 || kernel_size == 0)
            throw std::invalid_argument("ConvLayer: channels and kernel size must be non-zero");

        const Shape fs{out_channels, in_channels, kernel_size, kernel_size};

        const T stddev = std::sqrt(T{2} / static_cast<T>(in_channels * kernel_size * kernel_size));
        filters = Tensor<T>::randn(fs);
        filters.apply([stddev](T v) { return v * stddev; });
        filters_grad = Tensor<T>::zeros(fs);

        bias      = Tensor<T>::zeros(Shape{out_channels});
        bias_grad = Tensor<T>::zeros(Shape{out_channels});
    }

    size_t in_channels() const { return filters.dim(1); }
    size_t out_channels() const { return filters.dim(0); }
    size_t kernel_size() const { return filters.dim(2); }

    // y[n, o, i, j] = bias[o] + sum_{c, u, v} x[n, c, i+u, j+v] * filters[o, c, u, v]
    Tensor<T> forward(const Tensor<T>& x) override
    {
        detail::require_same_device("ConvLayer::forward", filters, filters_grad, bias, bias_grad, x);

        const Shape in_shape = x.shape();
        if (in_shape.ndim != 4)
            throw std::invalid_argument("ConvLayer::forward: expected 4D input {N, C_in, H, W}, got " + in_shape.str());

        const size_t K = kernel_size();
        const size_t I = in_channels();
        const size_t N = in_shape[0];
        const size_t O = out_channels();
        const size_t H = in_shape[2];
        const size_t W = in_shape[3];

        if (in_shape[1] != I)
            throw std::invalid_argument("ConvLayer::forward: expected " + std::to_string(I) +
                                        " input channels, got " + std::to_string(in_shape[1]));
        if (H < K || W < K)
            throw std::invalid_argument("ConvLayer::forward: input spatial size " + std::to_string(H) + "x" +
                                        std::to_string(W) + " is smaller than kernel size " + std::to_string(K));

        input = x;

        const size_t outH = H - K + 1;
        const size_t outW = W - K + 1;

        Tensor<T> y(x.device(), N, O, outH, outW);

        if (x.device() == Device::CPU) {
            #pragma omp parallel for collapse(2)
            for (size_t n = 0; n < N; n++) {
                for (size_t out = 0; out < O; out++) {
                    for (size_t i = 0; i < outH; i++) {
                        for (size_t j = 0; j < outW; j++) {
                            T acc = bias(out);
                            for (size_t c = 0; c < I; c++) {
                                for (size_t u = 0; u < K; u++) {
                                    for (size_t v = 0; v < K; v++) {
                                        acc += filters(out, c, u, v) * x(n, c, i + u, j + v);
                                    }
                                }
                            }
                            y(n, out, i, j) = acc;
                        }
                    }
                }
            }
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            cuda_ops::conv2d_forward(x.view(), filters.cview(), bias.cview(), y.view());
#else
            detail::require_cpu("ConvLayer::forward", x);
#endif
        }


        return y;
    }

    //   filters_grad[o, c, u, v] += sum_{n, i, j} dY[n, o, i, j] * x[n, c, i+u, j+v]
    //   bias_grad[o]             += sum_{n, i, j} dY[n, o, i, j]
    //   dX[n, c, p, q] = sum_{o, u, v} dY[n, o, p-u, q-v] * filters[o, c, u, v]   (out-of-range dY terms are 0)
    Tensor<T> backward(const Tensor<T>& dY) override
    {
        detail::require_same_device("ConvLayer::backward", dY, input, filters, filters_grad, bias, bias_grad);

        const size_t N = input.shape()[0];
        const size_t H = input.shape()[2];
        const size_t W = input.shape()[3];
        const size_t K = kernel_size();
        const size_t O = out_channels();
        const size_t I = in_channels();

        const size_t outH = H - K + 1;
        const size_t outW = W - K + 1;

        if (!(dY.shape() == Shape{N, O, outH, outW}))
            throw std::invalid_argument("ConvLayer::backward: input gradient shape must match output shape");

        Tensor<T> dX(dY.device(), N, I, H, W);

        if (dY.device() == Device::CPU) {
            #pragma omp parallel for collapse(2)
            for (size_t out = 0; out < O; out++) {
                for (size_t in = 0; in < I; in++) {
                    for (size_t i = 0; i < K; i++) {
                        for (size_t j = 0; j < K; j++) {
                            T acc = T{0};
                            for (size_t k = 0; k < N; k++) {
                                for (size_t u = 0; u < outH; u++) {
                                    for (size_t v = 0; v < outW; v++) {
                                        acc += dY(k, out, u, v) * input(k, in, i + u, j + v);
                                    }
                                }
                            }
                            filters_grad(out, in, i, j) += acc;
                        }
                    }
                }
            }

            for (size_t out = 0; out < O; out++) {
                T acc = T{0};
                for (size_t k = 0; k < N; k++) {
                    for (size_t u = 0; u < outH; u++) {
                        for (size_t v = 0; v < outW; v++) {
                            acc += dY(k, out, u, v);
                        }
                    }
                }

                bias_grad(out) += acc;
            }

            #pragma omp parallel for collapse(4)
            for (size_t n = 0; n < N; n++) {
                for (size_t in = 0; in < I; in++) {
                    for (size_t i = 0; i < H; i++) {
                        for (size_t j = 0; j < W; j++) {
                            T acc = T{0};
                            for (size_t out = 0; out < O; out++) {
                                const size_t u_begin = i >= K ? i - K + 1 : 0;
                                const size_t u_end   = std::min(i + 1, outH);
                                const size_t v_begin = j >= K ? j - K + 1 : 0;
                                const size_t v_end   = std::min(j + 1, outW);

                                for (size_t u = u_begin; u < u_end; u++) {
                                    for (size_t v = v_begin; v < v_end; v++) {
                                        acc += dY(n, out, u, v) * filters(out, in, i - u, j - v);
                                    }
                                }
                            }
                            dX(n, in, i, j) = acc;
                        }
                    }
                }
            }
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            cuda_ops::conv2d_backward(input.cview(), filters.cview(), dY.view(), 
                filters_grad.view(), bias_grad.view(), dX.view());
#else
            detail::require_cpu("ConvLayer::backward", dY);
#endif
        }

        return dX;
    }

    void to(Device dev) override
    {
        filters      = filters.to(dev);
        filters_grad = filters_grad.to(dev);
        bias         = bias.to(dev);
        bias_grad    = bias_grad.to(dev);
    }

    std::vector<Param<T>> params() override { return {{filters, filters_grad}, {bias, bias_grad}}; }

private:
    Tensor<T> input;
};


// 2D max pooling, non-overlapping P x P windows (stride = P, floor division).
//   x : {N, C, H,     W    }
//   y : {N, C, H / P, W / P}
template <std::floating_point T = fp32>
struct MaxPoolLayer : Layer<T> {
    explicit MaxPoolLayer(size_t pool_size = 2) : pool_size_(pool_size)
    {
        if (pool_size == 0)
            throw std::invalid_argument("MaxPoolLayer: pool size must be non-zero");
    }

    size_t pool_size() const { return pool_size_; }

    // y[n, c, i, j] = max_{0 <= u, v < P} x[n, c, i*P + u, j*P + v]
    Tensor<T> forward(const Tensor<T>& x) override
    {
        const Shape in_shape = x.shape();
        const size_t P = pool_size_;
        const size_t N = in_shape[0];
        const size_t C = in_shape[1];
        const size_t H = in_shape[2];
        const size_t W = in_shape[3];

        if (in_shape.ndim != 4 || H < P || W < P)
            throw std::invalid_argument("MaxPoolLayer::forward: expected 4D input with H, W >= P");

        input_shape = in_shape;
        const size_t outH = H / P;
        const size_t outW = W / P;

        Tensor<T> y(x.device(), N, C, outH, outW);
        max_indices = Tensor<size_t>(x.device(), N, C, outH, outW);

        if (x.device() == Device::CPU) {
            for (size_t n = 0; n < N; n++) {
                for (size_t c = 0; c < C; c++) {
                    for (size_t i = 0; i < outH; i++) {
                        const size_t base_row_off = input_shape.offset(n, c, i * P, 0);

                        for (size_t j = 0; j < outW; j++) {
                            size_t max_idx = base_row_off + j * P;
                            T max          = x.raw()[max_idx];

                            for (size_t u = 0; u < P; u++) {
                                const size_t row_off = base_row_off + u * W;
                                const T* row         = &x.raw()[row_off];

                                for (size_t v = 0; v < P; v++) {
                                    const size_t col_off = j * P + v;
                                    T val = row[col_off];

                                    if (val > max) {
                                        max = val;
                                        max_idx = row_off + col_off;
                                    }
                                }
                            }

                            max_indices(n, c, i, j) = max_idx;
                            y(n, c, i, j) = max;
                        }
                    }
                }
            }
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            cuda_ops::maxpool2d_forward(x.view(), y.view(), max_indices.view(), P);
#else
            detail::require_cpu("MaxPoolLayer::forward", x);
#endif
        }


        return y;
    }

    // Routes each dY[n, c, i, j] back to the winning element of its P x P window
    Tensor<T> backward(const Tensor<T>& dY) override
    {
        detail::require_same_device("MaxPoolLayer::backward", dY, max_indices);

        const size_t P = pool_size_;
        const size_t N = input_shape[0];
        const size_t C = input_shape[1];
        const size_t H = input_shape[2];
        const size_t W = input_shape[3];

        const size_t outH = H / P;
        const size_t outW = W / P;

        if (!(dY.shape() == Shape{N, C, outH, outW}))
            throw std::invalid_argument("MaxPoolLayer::backward: gradient tensor shape does not match cached output shape");

        Tensor<T> dX = Tensor<T>::zeros(dY.device(), {N, C, H, W});

        if (dY.device() == Device::CPU) {
            auto indices = max_indices.flatten();
            auto flat_dY = dY.flatten();
            auto flat_dX = dX.flatten();
            for (size_t i = 0; i < indices.size(); i++)
                flat_dX(indices(i)) = flat_dY(i);
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            cuda_ops::maxpool2d_backward(dY.view(), max_indices.cview(), dX.view());
#else
            detail::require_cpu("MaxPoolLayer::backward", dY);
#endif
        }

        return dX;
    }

private:
    size_t pool_size_;
    Shape input_shape;
    Tensor<size_t> max_indices; // {N, C, H/P, W/P}: passing index per pooling window
};


// Elementwise ReLU activation (works on any tensor shape).
//   y[k] = max(0, x[k])
template <std::floating_point T = fp32>
struct ReLULayer : Layer<T> {
    // y[k] = x[k] > 0 ? x[k] : 0
    Tensor<T> forward(const Tensor<T>& x) override
    {
        input = x;

        Tensor<T> y(x.device(), x.shape());

        if (x.device() == Device::CPU) {
            auto flat_x = x.flatten();
            auto flat_y = y.flatten();
            for (size_t i = 0; i < x.size(); i++) {
                flat_y(i) = std::max(T{0}, flat_x(i));
            }
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            cuda_ops::relu_forward(x.view(), y.view());
#else
            detail::require_cpu("ReLULayer::forward", x);
#endif
        }

        return y;
    }

    // dX[k] = input[k] > 0 ? dY[k] : 0
    Tensor<T> backward(const Tensor<T>& dY) override
    {
        detail::require_same_device("ReLULayer::backward", dY, input);

        if (!(dY.shape() == input.shape()))
            throw std::invalid_argument("ReLULayer::backward: gradient shape " + dY.shape().str() +
                                        " does not match cached input shape " + input.shape().str());

        Tensor<T> dX(input.device(), input.shape());

        if (dY.device() == Device::CPU) {
            auto flat_dX    = dX.flatten();
            auto flat_dY    = dY.flatten();
            auto flat_input = input.flatten();
            for (size_t i = 0; i < flat_dX.size(); i++) {
                flat_dX(i) = flat_input(i) > T{0} ? flat_dY(i) : T{0};
            }
        }
        else {
#ifdef __GPU_ACCELERATION_CUDA
            cuda_ops::relu_backward(input.cview(), dY.view(), dX.view());
#else
            detail::require_cpu("ReLULayer::backward", dY);
#endif
        }

        return dX;
    }

private:
    Tensor<T> input;
};


// Simple matmul + bias
//  x :       {N, A}
//  y :       {N, B}
//  weights : {A, B}
//  bias:     {B}
template <std::floating_point T = fp32>
struct DenseLayer : Layer<T> {
    Tensor<T> weights;
    Tensor<T> weights_grad;
    Tensor<T> bias;
    Tensor<T> bias_grad;

    DenseLayer(size_t A, size_t B)
    {
        if (A == 0 || B == 0)
            throw std::invalid_argument("DenseLayer: input and output sizes must be non-zero");

        const Shape ws{A, B};

        // He initialisation: N(0, 2 / fan_in), fan_in = A.
        const T stddev = std::sqrt(T{2} / static_cast<T>(A));
        weights = Tensor<T>::randn(ws);
        weights.apply([stddev](T v) { return v * stddev; });
        weights_grad = Tensor<T>::zeros(ws);

        bias      = Tensor<T>::zeros(Shape{B});
        bias_grad = Tensor<T>::zeros(Shape{B});
    }

    size_t input_shape() const { return weights.shape()[0]; }
    size_t output_shape() const { return weights.shape()[1]; }

    // y[n, i] = bias[i] + sum_{j} weights[j, i] * x[n, j]
    Tensor<T> forward(const Tensor<T>& x) override
    {
        const Shape in_shape = x.shape();
        const size_t A = input_shape();
        if (in_shape.ndim != 2 || in_shape[1] != A)
            throw std::invalid_argument("DenseLayer::forward: expected 2D input {N, " + std::to_string(A) +
                                        "}, got " + in_shape.str());

        input = x;
        const size_t N = in_shape[0];

        auto y = matmul(x, weights);
        for (size_t n = 0; n < N; n++) {
            auto y_n = y.slice(n);
            add_into(y_n, bias, y_n);
        }

        return y;
    }

    // dW[i, j] = sum_{n} dY[n, j] * x[n, i]
    // dB[i]    = sum_{n} dY[n, i]
    // dX[n, i] = sum_{j} dY[n, j] * weights[i, j]
    Tensor<T> backward(const Tensor<T>& dY) override
    {
        const size_t N = input.shape()[0];
        const size_t B = output_shape();

        const Shape expected{N, B};
        if (!(dY.shape() == expected))
            throw std::invalid_argument("DenseLayer::backward: expected gradient shape " + expected.str() +
                                        ", got " + dY.shape().str());

        Tensor<T> dX = Tensor<T>::zeros(input.device(), input.shape());

        matmul_into(transpose(input), dY, weights_grad);
        for (size_t n = 0; n < N; n++)
            add_into(bias_grad, dY.slice(n), bias_grad);
        matmul_into(dY, transpose(weights), dX);

        return dX;
    }

    void to(Device dev) override
    {
        weights      = weights.to(dev);
        weights_grad = weights_grad.to(dev);
        bias         = bias.to(dev);
        bias_grad    = bias_grad.to(dev);
    }

    std::vector<Param<T>> params() override { return {{weights, weights_grad}, {bias, bias_grad}}; }

private:
    Tensor<T> input;
};


// Flattens all dimensions after the batch axis (zero-copy view).
//   x : {N, d1, ...}
//   y : {N, d1 * ...}
template <std::floating_point T = fp32>
struct FlattenLayer : Layer<T> {
    Tensor<T> forward(const Tensor<T>& x) override
    {
        const Shape in_shape = x.shape();
        if (in_shape.ndim < 2 || in_shape[0] == 0)
            throw std::invalid_argument("FlattenLayer::forward: expected at least 2D input with N > 0, got " +
                                        in_shape.str());

        input_shape = in_shape;
        const size_t N = in_shape[0];
        return x.reshape(Shape{N, x.size() / N});
    }

    Tensor<T> backward(const Tensor<T>& dY) override
    {
        const size_t N = input_shape[0];
        const Shape expected{N, input_shape.size() / N};
        if (!(dY.shape() == expected))
            throw std::invalid_argument("FlattenLayer::backward: expected gradient shape " + expected.str() +
                                        ", got " + dY.shape().str());

        return dY.reshape(input_shape);
    }

private:
    Shape input_shape; // cached by forward()
};

#endif // LAYER_HPP
