
#include "convneural.h"

void ConvNN::train(ImageList data_set, size_t epochs, double alpha, bool verbose, size_t batch_size)
{
    auto train_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start;
    size_t batch_time;
    for (int j = 0; j < epochs; j++)
    {
        printf("\n");

        int counter = 0;
        for (size_t i = 0; i < data_set.length; i++, counter++)
        {   
            if (counter == 1)
                start = std::chrono::steady_clock::now();

            // Convolve the image matrix with the kernels
            MatrixList data(*(data_set[i]->image_data));
            MatrixList C0 = Conv2D(data, kernels, VALID, 1);
            Matrix Z0 = flatten(C0, COLUMN_FLATTEN);
            Matrix A0 = Z0.apply(sigmoid);
            
            // Propagate to the first layer (Hidden layer)
            Matrix Z1 = (W0 * A0) + B0;
            Matrix A1 = Z1.apply(sigmoid);
            
            // Propagate to the second layer (Output layer)
            Matrix Z2 = (W1 * A1) + B1;
            Matrix A2 = Z2.apply(sigmoid);

            // Calculate error
            Matrix yhat(10, 1);
            yhat[data_set[i]->label][0] = 1.0;

            if (counter == batch_size && verbose)
            {
                counter = 0;
                auto stop = std::chrono::steady_clock::now();
                batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                printf("\rE: %d/%zd I: %zd/%zd Error: %f (MSE) | speed: %zdms/batch (batch size: %zd)", j+1, epochs, i / std::min(batch_size, data_set.length), data_set.length / std::min(batch_size, data_set.length), mean((A2 - yhat).apply([](double in){return in * in;})), batch_time, batch_size);
            }
            
            Matrix dB1 = hadamard(Z2.apply(sigmoid_d), (A2 - yhat));
            Matrix dW1 = dB1 * A1.transpose();  
            Matrix dB0 = hadamard(Z1.apply(sigmoid_d), W1.transpose() * (A2 - yhat));          
            Matrix dW0 = dB0 * A0.transpose();
            Matrix dZ0 = hadamard(Z0.apply(sigmoid_d), W0.transpose() * (W1.transpose() * (A2 - yhat)));
            MatrixList dKernels = Conv2D(data, dflatten(dZ0, TensorShape(26, 26), kernels.size(), COLUMN_FLATTEN), VALID, 1);

            for (int kj = 0; kj < kernels.size(); kj++)
                kernels[kj] = kernels[kj] + (-0.25 * dKernels[kj]);
            W0 = W0 + (-alpha * dW0);
            W1 = W1 + (-alpha * dW1);
            B0 = B0 + (-alpha * dB0);
            B1 = B1 + (-alpha * dB1);
        }
    }
    auto train_stop = std::chrono::steady_clock::now();
    double epoch_time = (double) std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count() / epochs;
    if (verbose)
        printf("\n\nTime elapsed: %zds, ~%.2fs per epoch\n", std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count(), epoch_time);
}

double ConvNN::evaluate(ImageList data_set, bool verbose)
{
    double ncorrect = 0;
    int counter = 1;
    size_t msAvg = 0;
    std::chrono::steady_clock::time_point start;

    printf("\n");
    for (size_t i = 0; i < data_set.length; i++, counter++)
    {   
        if (counter == 1)
            start = std::chrono::steady_clock::now();
        
        Matrix prediction = run(data_set[i]);
        if (data_set[i]->label == prediction.argmax(COLUMN, 0))
            ncorrect++;
        
        if (counter == std::min((int)data_set.length, 100))
        {
            counter = 0;
            auto stop = std::chrono::steady_clock::now();
            auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
            if (verbose)
                printf("\r%zd/%zd %zdms/%dsteps", i+1, data_set.length, timeElapsed, std::min((int)data_set.length, 100));
            msAvg += timeElapsed;
        }
    }
    if (verbose)
        printf("\nTime elapsed: %zds, avg batch timing --> %zdms/%dsteps\n -------------------------------------------\n", msAvg/1000, msAvg / (data_set.length / std::min((int)data_set.length, 100)), std::min((int)data_set.length, 100));
    return 1.0 * ncorrect / data_set.length;
}

int ConvNN::predict(Image* i)
{
    return (int) run(i).argmax(COLUMN, 0);
}

Matrix ConvNN::run(Image* i)
{
    // Convolve the image matrix with the kernels
    MatrixList C0 = Conv2D(MatrixList(*(i->image_data)), kernels, VALID, 1);
    Matrix Z0 = flatten(C0, COLUMN_FLATTEN);
    Matrix A0 = Z0.apply(sigmoid);
    
    // Propagate to the first layer (Hidden layer)
    Matrix Z1 = (W0 * A0) + B0;
    Matrix A1 = Z1.apply(sigmoid);
    
    // Propagate to the second layer (Output layer)
    Matrix Z2 = (W1 * A1) + B1;
    Matrix A2 = Z2.apply(sigmoid);
    A2 = softmax(A2);
    
    return A2;
}