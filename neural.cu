
#include "neural.h"

void SimpleNeuralNetwork::train(ImageList data_set, size_t epochs, size_t batch_size, double alpha, bool verbose)
{
    auto train_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start;
    size_t batch_time;
    for (int j = 0; j < epochs; j++)
    {
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
            DMATRIX dotZ1 = dMatDot(d_W0, d_A0);
            DMATRIX d_Z1 = dMatAdd(dotZ1, d_B0);
            DMATRIX d_A1 = dMatApply(d_Z1, 0);
            
            // Propagate to the second layer (Output layer)
            DMATRIX dotZ2 = dMatDot(d_W1, d_A1);
            DMATRIX d_Z2 = dMatAdd(dotZ2, d_B1);
            DMATRIX d_A2 = dMatApply(d_Z2, 0);

            dMatFree(dotZ1);
            dMatFree(dotZ2);

            // Calculate error
            Matrix yhat(10, 1);
            yhat[data_set[i]->label][0] = 1.0;
            DMATRIX d_yhat = copyHostToDevice(yhat.getdata(), yhat.rows, yhat.cols);

            if (counter == batch_size && verbose)
            {
                counter = 0;
                auto stop = std::chrono::steady_clock::now();
                batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                printf("E: %d/%zd I: %zd/%zd Error: %f (MSE) | speed: %zdms/batch (batch size: %zd)\n", j+1, epochs, i / std::min(batch_size, data_set.length), data_set.length / std::min(batch_size, data_set.length), MSE(yhat, d_A2), batch_time, batch_size);
            }

            DMATRIX subtracted = dMatSubtract(d_A2, d_yhat);
            DMATRIX applied = dMatApply(d_Z2, 1);
            DMATRIX d_dB1 = dMatMultiply(applied, subtracted);
            DMATRIX transposed = dMatTranspose(d_A1);
            DMATRIX d_dW1 = dMatDot(d_dB1, transposed);
            
            dMatFree(applied);
            dMatFree(transposed);
            dMatFree(d_yhat);

            DMATRIX W1transposed = dMatTranspose(d_W1);
            DMATRIX dotted = dMatDot(W1transposed, subtracted);
            applied = dMatApply(d_Z1, 1);
            DMATRIX d_dB0 = dMatMultiply(applied, dotted);
            transposed = dMatTranspose(d_A0);
            DMATRIX d_dW0 = dMatDot(d_dB0, transposed);

            dMatFree(d_Z1);
            dMatFree(d_Z2);
            dMatFree(d_A0);
            dMatFree(d_A1);
            dMatFree(d_A2);
            dMatFree(subtracted);
            dMatFree(applied);
            dMatFree(transposed);
            dMatFree(W1transposed);
            dMatFree(dotted);

            // Matrix dW1 = hadamard(sigmoid_d(A2), (A2 - y_hat)) * A1.transpose();
            // Matrix dB1 = hadamard(sigmoid_d(A2), (A2 - y_hat));
            // Matrix dW0 = hadamard(Z1.apply(softplus_d), W1.transpose() * (A2 - y_hat)) * (*data_set[i]->image_data).flatten(ROW_FLATTEN).transpose();
            // Matrix dB0 = hadamard(Z1.apply(softplus_d), W1.transpose() * (A2 - y_hat));
            
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
            // W0 = W0 - (alpha * dW0);
            // W1 = W1 - (alpha * dW1);
            // B0 = B0 - (alpha * dB0);
            // B1 = B1 - (alpha * dB1);
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
                printf("E: %d/%zd I: %zd/%zd Error: %f (MSE) | speed: %zdms/batch (batch size: %zd)\n", j+1, epochs, i / std::min(batch_size, data_set.length), data_set.length / std::min(batch_size, data_set.length), mean((A2 - yhat).apply([](double in){return in * in;})), batch_time, batch_size);
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
    size_t epoch_time = std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count() / epochs;
    if (verbose)
        printf("\n\nTime elapsed: %zds, ~%zds per epoch\n", std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count(), epoch_time);
}

double SimpleNeuralNetwork::evaluate(ImageList data_set, bool verbose)
{
    double ncorrect = 0;
    int counter = 1;
    size_t msAvg = 0;
    std::chrono::steady_clock::time_point start;

    for (size_t i = 0; i < data_set.length; i++, counter++)
    {   
        if (counter == 1)
            start = std::chrono::steady_clock::now();

        Matrix prediction = run(data_set[i]);
        if (data_set[i]->label == prediction.argmax(COLUMN, 0))
            ncorrect++;

        if (counter == min((int)data_set.length, 100))
        {
            counter = 0;
            auto stop = std::chrono::steady_clock::now();
            auto timeElapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
            if (verbose)
                printf("%zd/%zd %zdms/batch (batch size: %d)\n", i+1, data_set.length, timeElapsed, min((int)data_set.length, 100));
            msAvg += timeElapsed;
        }
        if (counter == data_set.length)
            auto stop = std::chrono::steady_clock::now();
    }
    if (verbose)
        printf("avg batch timing --> %zdms/batch (batch size %d)\n -------------------------------------------\n",  msAvg / (data_set.length / min((int)data_set.length, 100)), min((int)data_set.length, 100));
    return 1.0 * ncorrect / data_set.length;
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
    std::cout << "Prediction " << prediction.argmax(COLUMN, 0) << std::endl;
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

double MSE(Matrix A, DMATRIX B)
{
    Matrix hB(10, 1);
    HPDOUBLE hBData = copyDeviceToHost(B);

    // hBData = new double*[1];
    // hBData[0] = new double[10];

    // cudaMemcpy(hBData[0], B.dataPtr, 80, cudaMemcpyDeviceToHost);

    hB.setdata(hBData, B.rows, B.cols);
    //std::cout << A << hB;
    return mean((hB - A).apply([](double in) {return pow(in, 2);}));
}

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
