
#ifndef DATASETS_MNIST_HPP
#define DATASETS_MNIST_HPP

#include "dataset.hpp"

#include <cstddef>
#include <iosfwd>
#include <string>

inline constexpr size_t MNIST_CHANNELS = 1;
inline constexpr size_t MNIST_HEIGHT   = 28;
inline constexpr size_t MNIST_WIDTH    = 28;
inline constexpr size_t MNIST_CLASSES  = 10;

// Loads an MNIST CSV file (header + rows of `label,p0,p1,...,p783`) into a Dataset:
//   inputs  : {N, 1, 28, 28}  normalized to [0, 1]
//   targets : {N, 10}         one-hot encoded
// Pass max_samples = 0 to load the entire file.
Dataset load_mnist_csv(const std::string& path, size_t max_samples = 0);

// Renders sample @idx as ASCII art along with its ground-truth label.
void print_mnist_image(std::ostream& os, const Dataset& ds, size_t idx);

#endif // DATASETS_MNIST_HPP
