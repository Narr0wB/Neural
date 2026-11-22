
#include "datasets/mnist.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <fstream>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace {

int parse_int(const char*& p, const char* end, const char* what)
{
    int value = 0;
    auto [next, ec] = std::from_chars(p, end, value);
    if (ec != std::errc{})
        throw std::runtime_error(std::string("load_mnist_csv: failed to parse ") + what);
    p = next;
    if (p < end && *p == ',')
        ++p;
    return value;
}

} // namespace

Dataset load_mnist_csv(const std::string& path, size_t max_samples)
{
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("load_mnist_csv: cannot open file '" + path + "'");

    // Skip CSV header.
    file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');

    std::vector<std::string> lines;
    if (max_samples > 0)
        lines.reserve(max_samples);

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty())
            continue;
        if (line.back() == '\r')
            line.pop_back();
        lines.push_back(std::move(line));
        if (max_samples > 0 && lines.size() == max_samples)
            break;
    }

    const size_t N = lines.size();
    const size_t pixels_per_img = MNIST_HEIGHT * MNIST_WIDTH;

    Dataset ds;
    ds.inputs  = Tensor<fp32>(N, MNIST_CHANNELS, MNIST_HEIGHT, MNIST_WIDTH);
    ds.targets = Tensor<fp32>(N, MNIST_CLASSES);

    fp32* in_data  = ds.inputs.raw();
    fp32* tgt_data = ds.targets.raw();

    for (size_t n = 0; n < N; ++n) {
        const char* p   = lines[n].data();
        const char* end = p + lines[n].size();

        const int label = parse_int(p, end, "label");
        if (label < 0 || static_cast<size_t>(label) >= MNIST_CLASSES)
            throw std::runtime_error("load_mnist_csv: label out of range: " + std::to_string(label));
        tgt_data[n * MNIST_CLASSES + static_cast<size_t>(label)] = 1.0f;

        fp32* dst = in_data + n * pixels_per_img;
        for (size_t i = 0; i < pixels_per_img; ++i) {
            const int px = parse_int(p, end, "pixel");
            dst[i] = static_cast<fp32>(px) / 255.0f;
        }
    }

    return ds;
}

void print_mnist_image(std::ostream& os, const Dataset& ds, size_t idx)
{
    if (idx >= ds.size())
        throw std::out_of_range("print_mnist_image: index out of range");

    static constexpr char ramp[] = ".,-~:;=!*#$@";
    static constexpr size_t ramp_max = sizeof(ramp) - 2; // exclude '\0'

    for (size_t r = 0; r < ds.inputs.dim(2); ++r) {
        for (size_t c = 0; c < ds.inputs.dim(3); ++c) {
            const fp32 v = std::clamp(ds.inputs(idx, 0, r, c), 0.0f, 1.0f);
            const size_t level = static_cast<size_t>(std::lround(v * static_cast<fp32>(ramp_max)));
            os << ramp[level] << ' ';
        }
        os << '\n';
    }

    size_t label = 0;
    for (size_t k = 1; k < ds.targets.dim(1); ++k)
        if (ds.targets(idx, k) > ds.targets(idx, label))
            label = k;
    os << "Label: " << label << '\n';
}
