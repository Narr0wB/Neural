
#include "layer.hpp"

#include <algorithm>
#include <limits>

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

    Tensor<fp32> y(N, O, outH, outW);

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

    for (size_t out = 0; out < O; out++) {
        fp32 acc = 0;
        for (size_t k = 0; k < N; k++) {
            for (size_t u = 0; u < outH; u++) {
                for (size_t v = 0; v < outW; v++) {
                    acc += dY(k, out, u, v);
                }
            }
        }

        bias_grad(out) += acc;
    }

    Tensor<fp32> dX(N, I, H, W);

    for (size_t n = 0; n < N; n++) {
        for (size_t in = 0; in < I; in++) {
            for (size_t i = 0; i < H; i++) {
                for (size_t j = 0; j < W; j++) {
                    fp32 acc = 0;
                    for (size_t out = 0; out < O; out++) {
                        const size_t u_begin = i >= K ? i - K + 1 : 0;
                        const size_t u_end   = std::min(i+1, outH);
                        const size_t v_begin = j >= K ? j - K + 1 : 0;
                        const size_t v_end   = std::min(j+1, outW);

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

    return dX;
}


Tensor<fp32> MaxPoolLayer::forward(const Tensor<fp32>& x)
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

    Tensor<fp32> y(N, C, outH, outW);
    max_indices = Tensor<size_t>(N, C, outH, outW);

    for (size_t n = 0; n < N; n++) {
        for (size_t c = 0; c < C; c++) {
            for (size_t i = 0; i < outH; i++) {
                const size_t base_row_off = input_shape.offset(n, c, i * P, 0);

                for (size_t j = 0; j < outW; j++) {
                    size_t max_idx = base_row_off + j * P;
                    fp32 max       = x.raw()[max_idx];

                    for (size_t u = 0; u < P; u++) {
                        const size_t row_off = base_row_off + u * W;
                        const fp32* row = &x.raw()[row_off];

                        for (size_t v = 0; v < P; v++) {
                            const size_t col_off = j * P + v;
                            fp32 val = row[col_off];

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

    return y;
}

Tensor<fp32> MaxPoolLayer::backward(const Tensor<fp32>& dY)
{
    const size_t P = pool_size_;
    const size_t N = input_shape[0];
    const size_t C = input_shape[1];
    const size_t H = input_shape[2];
    const size_t W = input_shape[3];

    const size_t outH = H / P;
    const size_t outW = W / P;

    if (!(dY.shape() == Shape{N, C, outH, outW}))
        throw std::invalid_argument("MaxPoolLayer::backward: gradient tensor shape does not match cached output shape");

    Tensor<fp32> dX(N, C, H, W);

    auto indices = max_indices.flatten();
    auto flat_dY = dY.flatten();
    auto flat_dX = dX.flatten();
    for (size_t i = 0; i < indices.size(); i++) {
        flat_dX(indices(i)) = flat_dY(i);
    }

    return dX;
}

Tensor<fp32> ReLULayer::forward(const Tensor<fp32>& x)
{
    input = x;

    Tensor<fp32> y(x.shape());

    auto flat_x = x.flatten();
    auto flat_y = y.flatten();
    for (size_t i = 0; i < x.size(); i++) {
        flat_y(i) = std::max(0.0f, flat_x(i));
    }

    return y;
}

Tensor<fp32> ReLULayer::backward(const Tensor<fp32>& dY)
{
    if (!(dY.shape() == input.shape()))
        throw std::invalid_argument("ReLULayer::backward: gradient shape " + dY.shape().str() +
                                    " does not match cached input shape " + input.shape().str());

    Tensor<fp32> dX(input.shape());

    auto flat_dX = dX.flatten();
    auto flat_dY = dY.flatten();
    auto flat_input = input.flatten();
    for (size_t i = 0; i < flat_dX.size(); i++) {
        flat_dX(i) = flat_input(i) > 0.0f ? flat_dY(i) : 0.0f;
    }

    return dX;
}

Tensor<fp32> DenseLayer::forward(const Tensor<fp32>& x)
{
    const Shape in_shape = x.shape();
    const size_t N = in_shape[0];
    const size_t A = input_shape();
    const size_t B = output_shape();

    if (in_shape[1] != A)
        throw std::invalid_argument("");

    Tensor<fp32> y(N, B);

    for (size_t n = 0; n < N; n++) {
        auto currentX = x.slice(n);
        auto currentY = y.slice(n);
        __matmul(weights, currentX, currentY);
        __add(bias, currentY, currentY);
    }

    return y;
}

Tensor<fp32> DenseLayer::backward(const Tensor<fp32>& dY)
{
    
}
