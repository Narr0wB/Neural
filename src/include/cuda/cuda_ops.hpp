
#ifndef CUDA_OPS_HPP
#define CUDA_OPS_HPP

#include "tensor_view.hpp"

namespace cuda_ops {

void init_mem_pool();

template <Numeric T>
void fill(TensorView<T> out, T value);

template <Numeric T>
void add(TensorView<const T> a, TensorView<const T> b, TensorView<T> out);

template <Numeric T>
void sub(TensorView<const T> a, TensorView<const T> b, TensorView<T> out);

template <Numeric T>
void matmul(TensorView<const T> a, TensorView<const T> b, TensorView<T> out,
            bool trans_a = false, bool trans_b = false, bool accumulate = true);

template <Numeric T>
void transpose(TensorView<const T> a, TensorView<T> out);

template <Numeric T>
void sgd_step(TensorView<T> params, TensorView<const T> grad, T lr);

template <Numeric T>
void relu_forward(TensorView<const T> in, TensorView<T> out);

template <Numeric T>
void relu_backward(TensorView<const T> in, TensorView<const T> dY, TensorView<T> dX);

template <Numeric T>
void maxpool2d_forward(TensorView<const T> x, TensorView<T> y, TensorView<size_t> max_indices, size_t pool_size);

template <Numeric T>
void maxpool2d_backward(TensorView<const T> dY, TensorView<const size_t> max_indices, TensorView<T> dX);

template <Numeric T>
void conv2d_forward(TensorView<const T> x, TensorView<const T> filters, TensorView<const T> bias, TensorView<T> y);

template <Numeric T>
void conv2d_backward(TensorView<const T> input, TensorView<const T> filters, TensorView<const T> dY,
                     TensorView<T> filters_grad, TensorView<T> bias_grad, TensorView<T> dX);

template <Numeric T>
void cross_entropy_forward(TensorView<const T> pred, TensorView<const T> target, TensorView<T> probs, TensorView<T> loss);

template <Numeric T>
void cross_entropy_backward(TensorView<const T> probs, TensorView<const T> target, TensorView<T> out);

template <ReduceOp op, Numeric T>
void reduce(TensorView<const T> x, size_t axis, TensorView<T> out);

template <Numeric T>
void gather(TensorView<const T> src, TensorView<const size_t> indices, TensorView<T> out);

} // namespace cuda_ops

#endif // CUDA_OPS_HPP
