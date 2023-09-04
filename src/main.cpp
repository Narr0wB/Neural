#include <iostream>
#include "neural.h"

int main(int argc, char** argv) {

    std::vector<Image> train = csv_to_image("../mnist_train.csv", 10000); // MAX IMGS -> 60k
    std::vector<Image> test = csv_to_image("../mnist_test.csv", 10000); // MAX IMGS -> 10k

    //SimpleNeuralNetwork e("test5e60k0p23.nn"); // Load constructor
    
    // To run a single image and get the output of the network (Loaded model) where the index of the list is taken from the user (argv arguments)
    // SimpleNeuralNetwork e("test5e60k0p23.nn"); 

    SimpleNeuralNetwork e(train, 1, 0.1, true); // Train constructor

    printf("##### Model accuracy: %.2f%c #####\n\n", e.evaluate(test, true) * 100, '%');

    //ConvNN e(train, 1, 0.1, 3, true);

    

    return 0;
}