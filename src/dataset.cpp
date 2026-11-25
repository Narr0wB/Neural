
#include "dataset.hpp"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>

Dataset gather_batch(const Dataset& ds, const Tensor<size_t>& indices, size_t start, size_t batch_size)
{
    if (ds.size() == 0)
        throw std::invalid_argument("gather_batch: dataset is empty");
    detail::require_same_device("gather_batch", ds.inputs, ds.targets, indices);
    if (ds.targets.ndim() == 0 || ds.targets.dim(0) != ds.size())
        throw std::invalid_argument("gather_batch: inputs and targets batch sizes do not match");
    if (indices.ndim() != 1)
        throw std::invalid_argument("gather_batch: indices must be a 1D tensor");
    if (start >= indices.size() || batch_size == 0)
        throw std::invalid_argument("gather_batch: invalid start index or zero batch size");

    const size_t B = std::min(batch_size, indices.size() - start);

    Shape batch_in_shape  = ds.inputs.shape();
    Shape batch_tgt_shape = ds.targets.shape();
    batch_in_shape.dims[0]  = B;
    batch_tgt_shape.dims[0] = B;

    const Device dev = ds.inputs.device();
    Dataset batch;
    batch.inputs  = Tensor<fp32>(dev, batch_in_shape);
    batch.targets = Tensor<fp32>(dev, batch_tgt_shape);

    if (dev == Device::CPU) {
        const size_t* idx_ptr = indices.raw() + start;
        for (size_t b = 0; b < B; ++b) {
            if (idx_ptr[b] >= ds.size())
                throw std::out_of_range("gather_batch: index " + std::to_string(idx_ptr[b]) + " out of range");
        }

        const size_t in_stride  = ds.inputs.size()  / ds.size();
        const size_t tgt_stride = ds.targets.size() / ds.size();

        const fp32* src_in  = ds.inputs.raw();
        const fp32* src_tgt = ds.targets.raw();
        fp32* dst_in  = batch.inputs.raw();
        fp32* dst_tgt = batch.targets.raw();

        for (size_t b = 0; b < B; ++b) {
            const size_t idx = idx_ptr[b];
            std::memcpy(dst_in  + b * in_stride,  src_in  + idx * in_stride,  in_stride  * sizeof(fp32));
            std::memcpy(dst_tgt + b * tgt_stride, src_tgt + idx * tgt_stride, tgt_stride * sizeof(fp32));
        }
    }
    else {
#ifdef __GPU_ACCELERATION_CUDA
        TensorView<const size_t> batch_idx(indices.raw() + start, Shape{B});
        cuda_ops::gather(ds.inputs.cview(),  batch_idx, batch.inputs.view());
        cuda_ops::gather(ds.targets.cview(), batch_idx, batch.targets.view());
#else
        detail::require_cpu("gather_batch", ds.inputs);
#endif
    }

    return batch;
}

Tensor<size_t> shuffled_indices(size_t n, Device dev)
{
    Tensor<size_t> idx(n);
    std::iota(idx.raw(), idx.raw() + n, size_t{0});
    std::shuffle(idx.raw(), idx.raw() + n, detail::rng());
    return idx.to(dev);
}
