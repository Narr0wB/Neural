
#ifndef LOSS_HPP
#define LOSS_HPP

#include "tensor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>


template <std::floating_point T = fp32, typename Target = Tensor<T>>
struct Loss {
    virtual T forward(const Tensor<T>& pred, const Target& target) = 0;
    virtual Tensor<T> backward() = 0;
    virtual ~Loss() = default;
};


template <std::floating_point T = fp32, typename Target = Tensor<T>>
struct CrossEntropyLoss : Loss<T, Target> {
    explicit CrossEntropyLoss() = default;

    T forward(const Tensor<T>& pred, const Target& target) override
    {
        if (!(pred.shape() == target.shape()) || pred.shape().ndim != 2 || pred.dim(0) == 0 || pred.dim(1) == 0)
            throw std::invalid_argument("CrossEntropyLoss::forward: prediction must have the same shape as the target");

        target_ = target;

        const Shape in_shape = pred.shape();
        const size_t N = in_shape[0];
        const size_t L = in_shape[1];

        T out = 0;

        probs_ = Tensor<T>(N, L);

        for (size_t n = 0; n < N; n++) {
            auto current_pred   = pred.slice(n);
            auto current_target = target.slice(n);

            T max = std::numeric_limits<T>::lowest();
            for (size_t i = 0; i < L; i++)
                max = std::max(max, current_pred(i));

            T log_arg = 0;
            for (size_t i = 0; i < L; i++) {
                T e = std::exp(current_pred(i) - max);
                probs_(n, i) = e;
                out -= current_target(i) * (current_pred(i) - max);
                log_arg += e;
            }
            for (size_t i = 0; i < L; i++)
                probs_(n, i) /= log_arg;

            out += std::log(log_arg);
        }

        return out / static_cast<T>(N);
    }

    Tensor<T> backward() override
    {
        const size_t N = probs_.dim(0);
        const size_t L = probs_.dim(1);
        const T inv_N = T{1} / static_cast<T>(N);

        Tensor<T> out(probs_.shape());

        for (size_t n = 0; n < N; n++) {
            for (size_t i = 0; i < L; i++) {
                out(n, i) = inv_N * (probs_(n, i) - target_(n, i));
            }
        }

        return out;
    }

private:
    Tensor<T> probs_;
    Target    target_;
};


#endif // LOSS_HPP