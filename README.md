# Neural
GPU-Accelerated (NVIDIA gpus only) simple ANN made for the mnist dataset (handwritten digits)

To use this model you will need the csv versions of the mnist datasets available at [Here](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) and put them in the same dir of the source files

In order to train the network you will need the make command [Make](https://stat545.com/make-windows.html) and the NVIDIA CUDA Toolkit [CUDA](https://developer.nvidia.com/cuda-toolkit)

Once you have installed both make and the CUDA toolkit you can run the current example (in main.cpp) by typing in the terminal `make`

NOTE: If you have a NVIDIA GPU define before including the modules __GPU_ACCELERATION_CUDA to train on the GPU

If you just want to test the accuracy of the pretrained modes, namely "testNeMk0pLR.nn", or train using the CPU you will not need the cuda toolkit but just a C/C++ compiler


## Usage

Load the train and test datasets by using the function in the Image module `ImageList csv_to_image(const char* path, int n_of_imgs)`
Then create a SimpleNeuralNetwork object and either pass the constructor the path of a pretrained module (.nn files) or use the train constructor `SimpleNeuralNetwork(ImageList data_set, size_t epochs, double learn_rate, bool verbose = false, size_t batch_size = 100)`

NOTE: If you have trained a model you can save its data by using the function `void SimpleNeuralNetwork::save(const char* path)` and making sure to add .nn as the file's extension

Once trained / loaded the model, you can either evaluate the accuracy of the model using the member function `double evaluate(ImageList data_set, bool verbose)` passing in the train dataset or run a single image by using `void SimpleNeuralNetwork::run_visual(Image* i)`

NOTE: the ImageList objects have the [] operator overloaded, meaning that they can return an Image* if given a valid index (e.g `Image* test = train[1]`), which in turn can be used with `void SimpleNeuralNetwork::run_visual(Image* i)`

## Examples

```cpp
#include <iostream>
#include "neural.h"
#include "cudalinear/linalg.h"
#include "image.h"

int main(int argc, char** argv) {

    ImageList train = csv_to_image("mnist_train.csv", 10000); // MAX IMGS -> 60k
    ImageList test = csv_to_image("mnist_test.csv", 1000); // MAX IMGS -> 10k

    SimpleNeuralNetwork e("test10e60k0p25.nn"); // Load constructor
    // SimpleNeuralNetwork e(train, 2, 0.25, true) // Train constructor
    
    printf("##### Model accuracy: %.2f%c #####\n\n", e.evaluate(test, true) * 100, '%'); // Evaluate model
    
    return 0;
   }
```
