
#include <iostream>
#include <format>
#include <numeric>

#include "neural.hpp"
#include "tensor.hpp"
#include "layer.hpp"
#include "loss.hpp"
#include "optimizer.hpp"

#include "dataset.hpp"
#include "datasets/mnist.hpp"

int main(int argc, char** argv) {
    auto mnist_train = load_mnist_csv("datasets/mnist_train.csv");
    auto mnist_eval  = load_mnist_csv("datasets/mnist_eval.csv");

    Model<fp32> cnn = Model<fp32>();
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
#endif
    
    auto optimizer = SGDOptimizer<fp32>(cnn.params(), 0.05f);
    auto loss      = CrossEntropyLoss<fp32>();

    const size_t batch_size = 32;
    const size_t epochs     = 5;

    for (size_t e = 0; e < epochs; e++) {
        auto indices = shuffled_indices(mnist_train.size());

        for (size_t i = 0; i < mnist_train.size(); i += batch_size) {
            auto batch = gather_batch(mnist_train, indices, i, batch_size);
#ifdef __GPU_ACCELERATION_CUDA
            batch.inputs = batch.inputs.to(Device::CUDA);
            batch.targets = batch.targets.to(Device::CUDA);
#endif

            auto out = cnn.forward(batch.inputs);
            auto l = loss.forward(out, batch.targets);
            cnn.backward(loss.backward());

            optimizer.optimize();
            optimizer.zero_grads();

            std::cout << std::format("batch {} | CELoss: {}", i, l) << std::endl;
        }
    }

    std::vector<size_t> eval_idx(mnist_eval.size());
    std::iota(eval_idx.begin(), eval_idx.end(), 0);

    size_t correct = 0;

#ifdef __GPU_ACCELERATION_CUDA
    cnn.to(Device::CPU);
#endif

    for (size_t i = 0; i < mnist_eval.size(); i += batch_size) {
        auto batch = gather_batch(mnist_eval, eval_idx, i, batch_size);
        auto out   = cnn.forward(batch.inputs);
        for (size_t n = 0; n < batch.size(); n++) {
            if (argmax(out.slice(n)) == argmax(batch.targets.slice(n)))
                correct++;
        }
    }
    std::cout << std::format("acc {:.2f}%", (fp32(correct) / fp32(mnist_eval.size())) * 100.0f) << std::endl;

    return 0;
}