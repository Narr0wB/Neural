
#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"
#include "layer.hpp"

#include <vector>

template <std::floating_point T = fp32>
struct Optimizer {
    explicit Optimizer(std::vector<Param<T>> params) : params_(std::move(params)) {}

    virtual void optimize() = 0;
    virtual void zero_grads()
    {
        for (auto& param : params_)
            param.grad.fill(T{0});
    }

    virtual ~Optimizer() = default;

protected:
    std::vector<Param<T>> params_;
};

template <std::floating_point T = fp32>
struct SGDOptimizer : Optimizer<T> {
    SGDOptimizer(std::vector<Param<T>> params, T learning_rate) : Optimizer<T>(params), lr(learning_rate) {}

    void optimize() override
    {
        for (auto& param : params_) {
            detail::require_same_device("SGDOptimizer::optimize", param.value, param.grad);
            detail::require_same_shape("SGDOptimizer::optimize", param.value, param.grad);

            if (param.value.device() == Device::CPU) {
                auto flat_value = param.value.flatten();
                auto flat_grad  = param.grad.flatten();
                for (size_t i = 0; i < flat_value.size(); i++) {
                    flat_value(i) -= lr * flat_grad(i);
                }
            }
            else {
#ifdef __GPU_ACCELERATION_CUDA
                cuda_ops::sgd_step(param.value.view(), param.grad.cview(), lr);
#else
                detail::require_cpu("SGDOptimizer::optimize", param.value);
#endif
            }
        }
    }

private:
    using Optimizer<T>::params_;
    T lr;
};

#endif // OPTIMIZER_HPP