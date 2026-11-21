
#ifndef OPTIMIZER_HPP
#define OPTIMIZER_HPP

#include "tensor.hpp"
#include "layer.hpp"

#include <vector>

template <Numeric T>
struct Optimizer {
    explicit Optimizer(std::vector<Param<T>> params) : params_(std::move(params)) {};

    virtual void optimize() = 0;
    virtual void zero_grads() = 0;

    virtual ~Optimizer() = default;

protected:
    std::vector<Param<T>> params_;
};

template <Numeric T>
struct SGDOptimizer : Optimizer<T> {
    SGDOptimizer(std::vector<Param<T>> params, T learning_rate) : Optimizer<T>(params), lr(learning_rate) {}

    void optimize() override
    {
        for (auto& param : params_) {
            auto flat_value = param.value.flatten();
            auto flat_grad  = param.grad.flatten();
            for (size_t i = 0; i < flat_value.size(); i++) {
                flat_value(i) -= lr * flat_grad(i);
            }
        }
    }

    void zero_grads() override
    {
        for (auto& param : params_) {
            param.grad.fill(T{0});
        }
    }

private:
    using Optimizer<T>::params_;
    T lr;
};

#endif // OPTIMIZER_HPP