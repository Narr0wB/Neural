
#include "layer.hpp"

Tensor<fp32> ConvLayer::forward(const Tensor<fp32>& x)
{
    input = x;

    const Shape in_shape = x.shape();
    const size_t K = kernel_size();
    const size_t I = in_channels();
    const size_t N = in_shape[0];
    const size_t H = in_shape[2];
    const size_t O = out_channels();
    const size_t W = in_shape[3];

    Tensor<fp32> y({N, O, H, W});

    for (size_t n = 0; n < N; n++) {
        for (size_t o = 0; o < O; o++) {
            for (size_t kx = 0; kx < W - K + 1; kx++) {
                for (size_t ky = 0; ky < H - K + 1; ky++) {
                    y(n, o) = bias(o);
                    for (size_t i = 0; i < I; i++) {
                        y(n, o) += hadamard(x(n, i), filters(o));
                    }
                }
            }
        }
    }
}