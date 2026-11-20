
#include "layer.hpp"

Tensor<fp32> ConvLayer::forward(const Tensor<fp32>& x)
{
    input = x;

    const Shape in_shape = x.shape();
    Tensor<fp32> y({in_shape[0], out_channels(), in_shape[2], in_shape(3)});


}