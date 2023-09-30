#include <iostream>
#include "neural.h"
#include "linalg/linalg.h"

int main(int argc, char** argv) {

    std::vector<Image> train = csv_to_image("../datasets/mnist_train.csv", 10000); // MAX IMGS -> 60,000
    std::vector<Image> test = csv_to_image("../datasets/mnist_test.csv", 1000); // MAX IMGS -> 10,000

    // TRAIN ----------------------------------------------------------------------------------
    
    SimpleNeuralNetwork e(train, 1, 0.28, true, 100); // Train constructor

    // To save a trained model
    //e.save("testmodel.nn");

    // LOAD -----------------------------------------------------------------------------------

    //SimpleNeuralNetwork e("../pretrained/model20e60k0p28.nn"); // Load constructor
    
    // To run a single image and get the output of the network (Loaded model) where the index of the list is taken from the user (argv arguments)
    // e.run_visual(test[argv[1]])

    

    

    // Test the accuracy of the model
    printf("##### Model accuracy: %.2f%c #####\n\n", e.evaluate(test, true) * 100, '%');

    

    return 0;
}