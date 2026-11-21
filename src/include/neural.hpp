
#ifndef NEURAL_HPP
#define NEURAL_HPP

#include "tensor.hpp"
#include "layer.hpp"

#include <vector>

template <Numeric T>
struct Model {


private:
    std::vector<Layer<T>> layers_;
};

#endif // NEURAL_HPP