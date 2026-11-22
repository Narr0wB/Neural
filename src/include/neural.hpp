
#ifndef NEURAL_HPP
#define NEURAL_HPP

#include "tensor.hpp"
#include "layer.hpp"

#include <vector>
#include <memory>

template <Numeric T>
struct Model {
    explicit Model() = default;

    void add(std::unique_ptr<Layer<T>> l) 
    { 
        auto p = l->params();
        params_.insert(params_.end(), p.begin(), p.end());
        layers_.push_back(std::move(l)); 
    }

    Tensor<T> forward(Tensor<T> x)
    {
        for (auto& l : layers_)
            x = l->forward(x);
        return x;
    }

    void backward(Tensor<T> dY)
    {
        for (auto it = layers_.rbegin(); it < layers_.rend(); it++)
            dY = (*it)->backward(dY);
    }

    std::vector<Param<T>>& params() { return params_; }

private:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
    std::vector<Param<T>> params_;
};

#endif // NEURAL_HPP