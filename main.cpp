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


    // To run a single image and get the output of the network (Loaded model) where the index of the list is taken from the user (argv arguments)
    // SimpleNeuralNetwork e("test5e60k0p23.nn"); 

    // e.run_visual(test[atoi(argv[1])]);

    return 0;
}