
#include "neural.h"

void SimpleNeuralNetwork::train(ImageList data_set, size_t epochs, double alpha, bool verbose, size_t batch_size)
{
    auto train_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start;
    size_t batch_time;
    for (int j = 0; j < epochs; j++)
    {
        printf("\n");
        #ifdef __GPU_ACCELERATION_CUDA

        DMATRIX d_W0 = copyHostToDevice(W0.getdata(), W0.rows, W0.cols);
        DMATRIX d_B0 = copyHostToDevice(B0.getdata(), B0.rows, B0.cols);
        DMATRIX d_W1 = copyHostToDevice(W1.getdata(), W1.rows, W1.cols);
        DMATRIX d_B1 = copyHostToDevice(B1.getdata(), B1.rows, B1.cols);

        int counter = 0;
        
        for (size_t i = 0; i < data_set.length; i++, counter++)
        {   
            if (counter == 1)
                start = std::chrono::steady_clock::now();
            
            // Flatten image data into a big one-dimensional vector
            DMATRIX tmpA0 = copyHostToDevice((*data_set[i]->image_data).flatten(ROW_FLATTEN).transpose().getdata(), B0.cols, W0.cols);
            DMATRIX d_A0 = dMatTranspose(tmpA0);
            dMatFree(tmpA0);

            // Propagate to the first layer (Hidden layer)
            DMATRIX d_Z1 = dDenseForwardProp(d_W0, d_A0, d_B0);
            DMATRIX d_A1 = dMatApply(d_Z1, 0);
            
            // Propagate to the second layer (Output layer)
            DMATRIX d_Z2 = dDenseForwardProp(d_W1, d_A1, d_B1);
            DMATRIX d_A2 = dMatApply(d_Z2, 0);

            // Calculate error
            Matrix yhat(10, 1);
            yhat[data_set[i]->label][0] = 1.0;
            DMATRIX d_yhat = copyHostToDevice(yhat.getdata(), yhat.rows, yhat.cols);
            
            if (counter == batch_size && verbose)
            {
                counter = 0;
                auto stop = std::chrono::steady_clock::now();
                batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                printf("\rE: %d/%zd I: %.0f/%zd Error: %.8f (MSE) Alpha: %.2f | speed: %zdms/batch (batch size: %zd)", j+1, epochs, ceil((double) i / std::min(batch_size, data_set.length)), data_set.length / std::min(batch_size, data_set.length), MSE(yhat, d_A2), alpha, batch_time, batch_size);
            }
            
            DMATRIX subtracted = dMatSubtract(d_A2, d_yhat);
            DMATRIX* d1 = dDenseBackProp(subtracted, d_Z2, d_A1, 1);
            DMATRIX d_dB1 = d1[0];
            DMATRIX d_dW1 = d1[1];
            
            delete[] d1;

            DMATRIX W1transposed = dMatTranspose(d_W1);
            DMATRIX dotted = dMatDot(W1transposed, subtracted);
            DMATRIX* d0 = dDenseBackProp(dotted, d_Z1, d_A0, 1);
            DMATRIX d_dB0 = d0[0];
            DMATRIX d_dW0 = d0[1];
            
            delete[] d0;
            dMatFree(d_Z1);
            dMatFree(d_Z2);
            dMatFree(d_A0);
            dMatFree(d_A1);
            dMatFree(d_A2);
            dMatFree(subtracted);
            dMatFree(W1transposed);
            dMatFree(dotted);
            
            DMATRIX scaled = dMatScale(d_dW0, alpha);
            subtracted = dMatSubtract(d_W0, scaled);
            
            dMatFree(d_W0);
            d_W0 = subtracted;
            
            dMatFree(scaled);
            
            scaled = dMatScale(d_dB0, alpha);
            subtracted = dMatSubtract(d_B0, scaled);

            dMatFree(d_B0);
            d_B0 = subtracted;

            dMatFree(scaled);

            scaled = dMatScale(d_dW1, alpha);
            subtracted = dMatSubtract(d_W1, scaled);

            dMatFree(d_W1);
            d_W1 = subtracted;

            dMatFree(scaled);

            scaled = dMatScale(d_dB1, alpha);
            subtracted = dMatSubtract(d_B1, scaled);

            dMatFree(d_B1);
            d_B1 = subtracted;

            dMatFree(scaled);

            dMatFree(d_dW0);
            dMatFree(d_dB0);
            dMatFree(d_dW1);
            dMatFree(d_dB1);

        }
        
        W0.setdata(copyDeviceToHost(d_W0), d_W0.rows, d_W0.cols);
        B0.setdata(copyDeviceToHost(d_B0), d_B0.rows, d_B0.cols);
        W1.setdata(copyDeviceToHost(d_W1), d_W1.rows, d_W1.cols);
        B1.setdata(copyDeviceToHost(d_B1), d_B1.rows, d_B1.cols);

        dMatFree(d_W0);
        dMatFree(d_B0);
        dMatFree(d_W1);
        dMatFree(d_B1);

        #else

        int counter = 0;
        for (size_t i = 0; i < data_set.length; i++, counter++)
        {   
            if (counter == 1)
                start = std::chrono::steady_clock::now();

            // Flatten image data into a big one-dimensional vector
            Matrix A0 = (*data_set[i]->image_data).flatten(ROW_FLATTEN);
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

            Matrix dW1 = hadamard(Z2.apply(sigmoid_d), (A2 - yhat)) * A1.transpose();
            Matrix dB1 = hadamard(Z2.apply(sigmoid_d), (A2 - yhat));
            Matrix dW0 = hadamard(Z1.apply(sigmoid_d), W1.transpose() * (A2 - yhat)) * (*data_set[i]->image_data).flatten(ROW_FLATTEN).transpose();
            Matrix dB0 = hadamard(Z1.apply(sigmoid_d), W1.transpose() * (A2 - yhat));
            
            W0 = W0 - (alpha * dW0);
            W1 = W1 - (alpha * dW1);
            B0 = B0 - (alpha * dB0);
            B1 = B1 - (alpha * dB1);
        }

        #endif
    }
    auto train_stop = std::chrono::steady_clock::now();
    double epoch_time = (double) std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count() / epochs;
    if (verbose)
        printf("\n\nTime elapsed: %zds, ~%.2fs per epoch\n", std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count(), epoch_time);
}

double SimpleNeuralNetwork::evaluate(ImageList data_set, bool verbose)
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
                printf("\r%zd/%zd %zdms/batch (batch size: %d)", i+1, data_set.length, timeElapsed, std::min((int)data_set.length, 100));
            msAvg += timeElapsed;
        }
    }
    if (verbose)
        printf("\nTime elapsed: %zds, avg batch timing --> %zdms/batch (batch size %d)\n -------------------------------------------\n", msAvg/1000, msAvg / (data_set.length / std::min((int)data_set.length, 100)), std::min((int)data_set.length, 100));
    return 1.0 * ncorrect / data_set.length;
}

int SimpleNeuralNetwork::predict(Image* i)
{
    return (int) run(i).argmax(COLUMN, 0);
}

Matrix SimpleNeuralNetwork::run(Image* i)
{
    // Flatten image data into a big one-dimensional vector
    Matrix A0 = (*i->image_data).flatten(ROW_FLATTEN);

    // Propagate to the first layer (Hidden layer)
    Matrix Z1 = (W0 * A0) + B0;
    Matrix A1 = Z1.apply(sigmoid);
    
    // Propagate to the second layer (Output layer)
    Matrix Z2 = (W1 * A1) + B1;
    Matrix A2 = Z2.apply(sigmoid);
    A2 = softmax(A2);
    
    return A2;
}

void SimpleNeuralNetwork::run_visual(Image* i)
{
    Matrix prediction = run(i);
    std::cout << *i << std::endl;
    std::cout << "Prediction: " << prediction.argmax(COLUMN, 0) << std::endl;
}

void SimpleNeuralNetwork::save(const char* path)
{
    std::ofstream file(path, std::ios::binary);
    for (int i = 0; i < W0.rows; i++)
    {
        for (int c = 0; c < W0.cols; c++)
        {
            file.write((char*) &W0[i][c], 8);
        }
    }

    for (int i = 0; i < B0.rows; i++)
    {
        file.write((char*) &B0[i][0], 8);
    }

    for (int i = 0; i < W1.rows; i++)
    {
        for (int c = 0; c < W1.cols; c++)
        {
            file.write((char*) &W1[i][c], 8);
        }
    }

    for (int i = 0; i < B1.rows; i++)
    {
        file.write((char*) &B1[i][0], 8);
    }
    file.close();
}

void SimpleNeuralNetwork::load(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    for (int i = 0; i < W0.rows; i++)
    {
        for (int c = 0; c < W0.cols; c++)
        {
            file.read((char*)&W0[i][c], 8);
        }
    }

    for (int i = 0; i < B0.rows; i++)
    {
        file.read((char*)&B0[i][0], 8);
    }

    for (int i = 0; i < W1.rows; i++)
    {
        for (int c = 0; c < W1.cols; c++)
        {
            file.read((char*)&W1[i][c], 8);
        }
    }

    for (int i = 0; i < B1.rows; i++)
    {
        file.read((char*)&B1[i][0], 8);
    }
    file.close();
}

#ifdef __GPU_ACCELERATION_CUDA
double MSE(Matrix A, DMATRIX B)
{
    Matrix hB(10, 1);

    hB.setdata(copyDeviceToHost(B), B.rows, B.cols);
    return mean((hB - A).apply([](double in) {return pow(in, 2);}));
}
#endif

Matrix one_hot(int label)
{
    Matrix one_hot_l(10, 1);

    *one_hot_l[label] = 1.0;

    return one_hot_l;
}

inline double ReLU(double in)
{
    return std::max(0.0, in);
}

double ReLU_d(double in)
{
    return (double) in > 0;
}

double sigmoid(double input) 
{
	return 1.0 / (1 + exp(-1 * input));
}

Matrix sigmoid_d(const Matrix& in) 
{
    Matrix ones(in.rows, in.cols);
    ones.fill(1.0);

    return hadamard(in, (ones - in));
}

double sigmoid_d(double in)
{
    return sigmoid(in) * (1 - sigmoid(in));
}

double tanh_d(double in)
{
    return 1 - pow(tanh(in), 2);
}

double mish(double in)
{
    return in * tanh(log(1 + exp(in)));
}

double mish_d(double in)
{
    return 0;
}

double softplus(double in)
{
    return log(1 + exp(in));
}

double softplus_d(double in)
{
    return sigmoid(in);
}

double mean(Matrix in)
{
    double sum = 0.0;
    for (int i = 0; i < in.rows; i++)
        for (int j = 0; j < in.cols; j++)
            sum += in[i][j];
    
    return sum / (in.rows * in.cols);
}
