#include <iostream>
#include "neural.h"
#include "cudalinear/linalg.h"
#include "image.h"

int main(int argc, char** argv) {

    ImageList train = csv_to_image("mnist_train.csv", 60000);
    ImageList test = csv_to_image("mnist_test.csv", 100);

    SimpleNeuralNetwork e("test10e60k0p25.nn");
    
    printf("##### Model accuracy: %.2f%c #####\n\n", e.evaluate(test, true) * 100, '%');

    // SimpleNeuralNetwork e("test5e60k0p23.nn");

    // e.run_visual(test[atoi(argv[1])]);

    return 0;
}