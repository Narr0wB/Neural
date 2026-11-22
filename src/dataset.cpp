
#include "dataset.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

Dataset gather_batch(const Dataset& ds, const std::vector<size_t>& indices, size_t start, size_t batch_size)
{
    if (ds.size() == 0)
        throw std::invalid_argument("gather_batch: dataset is empty");
    if (ds.targets.ndim() == 0 || ds.targets.dim(0) != ds.size())
        throw std::invalid_argument("gather_batch: inputs and targets batch sizes do not match");
    if (start >= indices.size() || batch_size == 0)
        throw std::invalid_argument("gather_batch: invalid start index or zero batch size");

    const size_t B = std::min(batch_size, indices.size() - start);
    const size_t in_stride  = ds.inputs.size()  / ds.size();
    const size_t tgt_stride = ds.targets.size() / ds.size();

    Shape batch_in_shape  = ds.inputs.shape();
    Shape batch_tgt_shape = ds.targets.shape();
    batch_in_shape.dims[0]  = B;
    batch_tgt_shape.dims[0] = B;

    Dataset batch;
    batch.inputs  = Tensor<fp32>(batch_in_shape);
    batch.targets = Tensor<fp32>(batch_tgt_shape);

    const fp32* src_in  = ds.inputs.raw();
    const fp32* src_tgt = ds.targets.raw();
    fp32* dst_in  = batch.inputs.raw();
    fp32* dst_tgt = batch.targets.raw();

    for (size_t b = 0; b < B; ++b) {
        const size_t idx = indices[start + b];
        if (idx >= ds.size())
            throw std::out_of_range("gather_batch: index " + std::to_string(idx) + " out of range");

        std::memcpy(dst_in  + b * in_stride,  src_in  + idx * in_stride,  in_stride  * sizeof(fp32));
        std::memcpy(dst_tgt + b * tgt_stride, src_tgt + idx * tgt_stride, tgt_stride * sizeof(fp32));
    }

    return batch;
}

std::vector<size_t> shuffled_indices(size_t n)
{
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), detail::rng());
    return idx;
}
