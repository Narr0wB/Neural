
#ifndef NEURAL_HPP
#define NEURAL_HPP

#include "tensor.hpp"
#include "layer.hpp"

#include <vector>
#include <memory>

template <std::floating_point T = fp32>
struct Model {
    explicit Model() = default;

    void add(std::unique_ptr<Layer<T>> l) { layers_.push_back(std::move(l)); }

    void to(Device dev) 
    {
        for (auto& l : layers_)
            l->to(dev);
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

    std::vector<Param<T>> params() 
    { 
        std::vector<Param<T>> out;
        for (auto& l : layers_) {
            auto l_params = l->params();
            out.insert(out.end(), l_params.begin(), l_params.end());
        }
        return out; 
    }

private:
    std::vector<std::unique_ptr<Layer<T>>> layers_;
};

#endif // NEURAL_HPP