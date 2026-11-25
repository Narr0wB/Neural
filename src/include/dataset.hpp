
#ifndef DATASET_HPP
#define DATASET_HPP

#include "tensor.hpp"

#include <cstddef>
#include <vector>

// Generic batched dataset (any tensor rank >= 1, batch along axis 0):
//   inputs  : {N, ...}
//   targets : {N, ...}
struct Dataset {
    Tensor<fp32> inputs;
    Tensor<fp32> targets;

    size_t size() const { return inputs.ndim() > 0 ? inputs.dim(0) : 0; }
};

// Copies up to @batch_size samples from @ds at @indices[start .. start + B)
// into a contiguous mini-batch Dataset (works for any input/target tensor rank).
Dataset gather_batch(const Dataset& ds, const Tensor<size_t>& indices, size_t start, size_t batch_size);

// Returns a randomly shuffled permutation of [0, n) on @dev.
Tensor<size_t> shuffled_indices(size_t n, Device dev = Device::CPU);

#endif // DATASET_HPP
