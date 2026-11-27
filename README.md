# NNeural

A small C++20 library for building and training convolutional neural networks (CNNs), with OpenMP CPU parallelization and optional NVIDIA CUDA/cuDNN GPU acceleration.

## Features

- **Tensors**: N-dimensional `Tensor<T>` with seamless `Device::CPU` / `Device::CUDA` transfers via `.to(...)`.
- **Sequential Model**: `Model<T>` container (`add`, `to`, `forward`, `backward`, `params`).
- **Layers**: `ConvLayer`, `MaxPoolLayer`, `ReLULayer`, `FlattenLayer`, `DenseLayer`.
- **Loss & Optimization**: `CrossEntropyLoss` and `SGDOptimizer`.
- **Datasets**: Batching helpers (`shuffled_indices`, `gather_batch`) and an MNIST CSV loader (`load_mnist_csv`).

## Requirements

- CMake ≥ 3.20
- C++20-compatible compiler
- OpenMP *(optional — automatically detected for multi-threaded CPU execution)*
- [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) & cuDNN *(optional — only needed when enabling CUDA)*

## Building

### CPU (default)

```bash
cmake -B build
cmake --build build
```

### With CUDA GPU Acceleration

```bash
cmake -B build -DUSE_CUDA=ON
cmake --build build
```

If cuDNN is installed in a non-standard location, pass `-DCUDNN_PATH=/path/to/cudnn` (or set `CUDNN_INCLUDE_DIR` and `CUDNN_LIBRARY`).

This builds the reusable static library `nneural_core` and the example MNIST training binary `build/nneural`.

## Example Usage

```cpp
#include <iostream>
#include <memory>

#include "nneural.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "datasets/mnist.hpp"

int main() {
    auto train = load_mnist_csv("datasets/mnist_train.csv");

    Model<fp32> cnn;
    cnn.add(std::make_unique<ConvLayer<fp32>>(1, 8, 3));
    cnn.add(std::make_unique<ReLULayer<fp32>>());
    cnn.add(std::make_unique<MaxPoolLayer<fp32>>(2));
    cnn.add(std::make_unique<ConvLayer<fp32>>(8, 16, 3));
    cnn.add(std::make_unique<ReLULayer<fp32>>());
    cnn.add(std::make_unique<MaxPoolLayer<fp32>>(2));
    cnn.add(std::make_unique<FlattenLayer<fp32>>());
    cnn.add(std::make_unique<DenseLayer<fp32>>(400, 64));
    cnn.add(std::make_unique<ReLULayer<fp32>>());
    cnn.add(std::make_unique<DenseLayer<fp32>>(64, 10));

#ifdef __GPU_ACCELERATION_CUDA
    cnn.to(Device::CUDA);
    train.inputs  = train.inputs.to(Device::CUDA);
    train.targets = train.targets.to(Device::CUDA);
#endif

    SGDOptimizer<fp32> optimizer(cnn.params(), 0.05f);
    CrossEntropyLoss<fp32> loss;

    const size_t batch_size = 64;
    const size_t epochs     = 5;

    for (size_t e = 0; e < epochs; ++e) {
        auto indices = shuffled_indices(train.size(), train.inputs.device());

        for (size_t i = 0; i < train.size(); i += batch_size) {
            auto batch = gather_batch(train, indices, i, batch_size);

            auto out = cnn.forward(batch.inputs);
            loss.forward(out, batch.targets);
            cnn.backward(loss.backward());

            optimizer.optimize();
            optimizer.zero_grads();
        }
        std::cout << "epoch " << e << " | loss: " << loss.loss() << '\n';
    }
}
```

See [src/main.cpp](src/main.cpp) for a complete training and evaluation example on MNIST.
