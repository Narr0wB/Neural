#include <iostream>
#include "neural.h"
#include "cudalinear/linalg.h"
#include "image.h"

int main(int argc, char** argv) {

    ImageList train = csv_to_image("mnist_train.csv", 10000);
    ImageList test = csv_to_image("mnist_test.csv", 1000);

    SimpleNeuralNetwork e(train, 10, 200, 0.02, true);
    e.save("test10e10k0p02.nn");

    printf("##### Model accuracy: %.2f%c #####\n", e.evaluate(test, true) * 100, '%');

    return 0;
}