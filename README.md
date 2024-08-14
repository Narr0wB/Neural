
# Neural
GPU-Accelerated (NVIDIA gpus only) simple ANN made for the mnist dataset (handwritten digits)

## Setting up
To build and test out this project you will need to follow a couple of steps:
- First, make sure you have installed the following essential tools:
```bash
>$ make
>$ g++
```
- Unzip the mnist_test.zip file datasets present in the ```datasets/``` directory 

- In order to train/use the neural network using the GPU you will need to install the [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)


## Build
Once you have installed the required tools and set up the environment you are now ready to build the project. Just navigate to the build directory and run the Makefile:
```bash
>$ cd build/
>$ make __CUDA=1 # or set to 0 to disable GPU
```

## Usage

Load the train and test datasets by using the function `ImageList csv_to_image(const char* path, int n_of_imgs)`,
then create a SimpleNeuralNetwork object and use one of the two contructors, the train constructor:

```cpp
SimpleNeuralNetwork(std::vector<Image> data_set, size_t epochs, double learn_rate, bool verbose = false, size_t batch_size = 100
```

Or the pretrained module constructor:

```cpp
SimpleNeuralNetwork(const char* path)
```
(Pretrained modules are available in the ```pretrained/``` directory)
##
After having loaded the datasets and created the NN all is left is to test it!
You can either test the model accuracy over a specific dataset:

```cpp
SimpleNeuralNetwork snn("../pretrained/model20e60k0p28.nn");
std::vector<Image> test = csv_to_image("../datasets/mnist_test.csv", 1000);

printf("##### Model accuracy: %.2f%% #####\n\n",  snn.evaluate(test,  true)  *  100);
```

Or you can pass it a single image and see the model's prediction:

```cpp
SimpleNeuralNetwork snn("../pretrained/model20e60k0p28.nn");
std::vector<Image> test = csv_to_image("../datasets/mnist_test.csv", 1000);

snn.run_visual(test[0])

/*
Expected output:
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
. . . . . . : * ! ! ~ - . . . . . . . . . . . . . . . . 
. . . . . . $ @ @ @ @ $ # # # # # # # # ! - . . . . . . 
. . . . . . ~ ; ~ ; ! $ @ $ @ @ @ @ $ @ @ = . . . . . . 
. . . . . . . . . . . , ~ , ~ ~ ~ ~ , $ @ ; . . . . . . 
. . . . . . . . . . . . . . . . . . : @ # , . . . . . . 
. . . . . . . . . . . . . . . . . , $ @ : . . . . . . . 
. . . . . . . . . . . . . . . . . = @ $ - . . . . . . . 
. . . . . . . . . . . . . . . . ~ @ @ ~ . . . . . . . . 
. . . . . . . . . . . . . . . . = @ * . . . . . . . . . 
. . . . . . . . . . . . . . . . # @ ~ . . . . . . . . . 
. . . . . . . . . . . . . . . ; @ * . . . . . . . . . . 
. . . . . . . . . . . . . . ~ @ $ - . . . . . . . . . . 
. . . . . . . . . . . . . , $ @ ! . . . . . . . . . . . 
. . . . . . . . . . . . . # @ # - . . . . . . . . . . . 
. . . . . . . . . . . . - @ @ ~ . . . . . . . . . . . . 
. . . . . . . . . . . , $ @ ; . . . . . . . . . . . . . 
. . . . . . . . . . . = @ @ - . . . . . . . . . . . . . 
. . . . . . . . . . ~ $ @ @ - . . . . . . . . . . . . . 
. . . . . . . . . . ; @ @ # - . . . . . . . . . . . . . 
. . . . . . . . . . ; @ # , . . . . . . . . . . . . . . 
. . . . . . . . . . . . . . . . . . . . . . . . . . . . 
Label: 7

The model predicted: 7
*/
```


Lastly, once you have trained a model you have the possibility to save the parameters to disk:
```cpp
std::vector<Image> train = csv_to_image("../datasets/mnist_train.csv", 10000);
    
SimpleNeuralNetwork snn(train, 1, 0.28, true, 100); // Train constructor

snn.save("/path/to/save.nn");
```
##
Here is a simple main.cpp :

main.cpp:
```cpp
#include <iostream>
#include "neural.h"

int main(int argc, char** argv) {

    std::vector<Image> train = csv_to_image("../datasets/mnist_train.csv", 10000); // MAX IMGS -> 60,000
    std::vector<Image> test = csv_to_image("../datasets/mnist_test.csv", 1000); // MAX IMGS -> 10,000

    // TRAIN ----------------------------------------------------------------------------------
    
    SimpleNeuralNetwork e(train, 1, 0.28, true, 100); // Train constructor
	printf("##### Model accuracy: %.2f%% #####\n\n",  snn.evaluate(test,  true)  *  100);
}
```
