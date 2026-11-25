
#include <chrono>
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
    (void)argc;
    (void)argv;
    using clock = std::chrono::steady_clock;
    auto secs = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double>(b - a).count();
    };

    const auto t_load_start = clock::now();
    auto mnist_train = load_mnist_csv("datasets/mnist_train.csv");
    auto mnist_eval  = load_mnist_csv("datasets/mnist_eval.csv");
    const auto t_load_end = clock::now();

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

    const size_t batch_size = 64;
    const size_t epochs     = 5;

#ifdef __GPU_ACCELERATION_CUDA
    mnist_train.inputs  = mnist_train.inputs.to(Device::CUDA);
    mnist_train.targets = mnist_train.targets.to(Device::CUDA);
#endif

    std::cout << std::format("Starting training on device: {}. batch_size: {}, epochs: {}",
                             device_name(mnist_train.inputs.device()), batch_size, epochs) << std::endl;

    const auto t_train_start = clock::now();
    for (size_t e = 0; e < epochs; e++) {
        auto indices = shuffled_indices(mnist_train.size(), mnist_train.inputs.device());

        for (size_t i = 0; i < mnist_train.size(); i += batch_size) {
            auto batch = gather_batch(mnist_train, indices, i, batch_size);

            auto out = cnn.forward(batch.inputs);
            loss.forward(out, batch.targets);
            cnn.backward(loss.backward());

            optimizer.optimize();
            optimizer.zero_grads();
        }
        auto l = loss.loss();
        std::cout << std::format("epoch {} | CELoss: {}", e, l) << std::endl;
    }
    const auto t_train_end = clock::now();

    Tensor<size_t> eval_idx(mnist_eval.size());
    std::iota(eval_idx.raw(), eval_idx.raw() + mnist_eval.size(), size_t{0});

    size_t correct = 0;

    const auto t_eval_start = clock::now();
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
    const auto t_eval_end = clock::now();

    std::cout << std::format("acc {:.2f}%", (fp32(correct) / fp32(mnist_eval.size())) * 100.0f) << std::endl;
    std::cout << std::format("time | load: {:.2f}s | train: {:.2f}s | eval: {:.2f}s | total: {:.2f}s",
                             secs(t_load_start, t_load_end),
                             secs(t_train_start, t_train_end),
                             secs(t_eval_start, t_eval_end),
                             secs(t_load_start, t_eval_end)) << std::endl;

    return 0;
}