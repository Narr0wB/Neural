
#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"


template <Numeric T, typename Target = Tensor<T>>
struct Loss {
    virtual T forward(const Tensor<T> pred, const Target& target) = 0;
    virtual Tensor<T> backward(T loss) = 0;
    virtual ~Loss() = default;
};

struct CrossEntropyLoss : Loss<fp32, Tensor<u8>> {
    explicit CrossEntropyLoss() = default;

    fp32 forward(const Tensor<fp32> pred, const Tensor<u8>& target) override
    {

    }

    Tensor<fp32> backward(fp32 loss) override
    {

    }

private:
    Tensor<fp32> input;
};


#endif // LOSS_HPP