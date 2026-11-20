
#include "layer.hpp"

Tensor<fp32> ConvLayer::forward(const Tensor<fp32>& x)
{
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

    Tensor<fp32> y({N, O, outH, outW});

    for (size_t n = 0; n < N; n++) {
        for (size_t out = 0; out < O; out++) {
            for (size_t i = 0; i < outH; i++) {
                for (size_t j = 0; j < outW; j++) {
                    fp32 acc = bias(out);
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

    return y;
}

Tensor<fp32> ConvLayer::backward(const Tensor<fp32>& dY)
{
    const size_t N = input.shape()[0];
    const size_t H = input.shape()[2];
    const size_t W = input.shape()[3];
    const size_t K = kernel_size();
    const size_t O = out_channels();
    const size_t I = in_channels();

    const size_t outH = H - K + 1;
    const size_t outW = W - K + 1;

    if (!(dY.shape() == Shape{N, O, outH, outW}))
        throw std::invalid_argument("ConvLayer::backward: input grandient shape must match output shape");

    // We first compute the gradient for the filters
    for (size_t out = 0; out < O; out++) {
        for (size_t in = 0; in < I; in++) {
            for (size_t i = 0; i < K; i++) {
                for (size_t j = 0; j < K; j++) {
                    fp32 acc = 0;
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
}