
#include "neural.h"

void SimpleNeuralNetwork::train(std::vector<Image> data_set, size_t epochs, double alpha, bool verbose, size_t batch_size)
{
    auto train_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start;
    size_t batch_time;

    #ifdef __GPU_ACCELERATION_CUDA
    // Layer 0 INPUT ------------------
    DeviceMatrix _A0;
    DeviceMatrix _W0(W0);
    DeviceMatrix _B0(B0);

    DeviceMatrix _dW0(W0);
    DeviceMatrix _dB0(B0);

    // Layer 1 ------------------------
    DeviceMatrix _A1(W1.cols(), 1);
    DeviceMatrix _W1(W1);
    DeviceMatrix _B1(B1);

    DeviceMatrix _Z1(W1.cols(), 1);
    DeviceMatrix _dW1(W1);
    DeviceMatrix _dB1(B1);

    // Layer 2 OUTPUT -----------------
    DeviceMatrix _A2(W1.rows(), 1);
    
    DeviceMatrix _Z2(W1.rows(), 1);
    DeviceMatrix _label;

    std::vector<DeviceImage> batch;
    size_t batches = data_set.size() / batch_size;
    if (data_set.size() % batch_size != 0) batches++;

    #endif // __GPU_ACCELERATION_CUDA

    for (size_t j = 0; j < epochs; j++)
    {
        printf("\n");

    #ifdef __GPU_ACCELERATION_CUDA
        for (size_t b = 0; b < batches; ++b) {
            batch = allocBatch(data_set, batches * batch_size, batch_size);
            for (size_t i = 0; i < batch_size; i++)
            {              
                // Flatten image data into a big one-dimensional vector
                _A0.set(batch[i].image, 28 * 28, 1);

                // Propagate to the first layer (Hidden layer)
                denseForwardProp(_Z1, _W0, _A0, _B0);
                _Z1.apply(_A1, SIGMOID);
                
                // Propagate to the second layer (Output layer)
                denseForwardProp(_A2, _W1, _A1, _B1);
                _Z2.apply(_A2, SIGMOID);

                _label.set(batch[i].label, 10, 1);
                _A2.subtract(_label);
                      
                denseBackProp(_dW1, _dB1, _A2, _Z2, _A1, SIGMOID_DERIVATIVE);

                DeviceMatrix dotted = _W1.transpose() * _A2;
                denseBackProp(_dW0, _dB0, dotted, _Z1, _A1, SIGMOID_DERIVATIVE);

                _dW0.scale(alpha);
                _W0.subtract(_dW0);

                _dB0.scale(alpha);
                _B0.subtract(_dB0);

                _dW1.scale(alpha);
                _W1.subtract(_dW1);

                _dB1.scale(alpha);
                _B1.subtract(_dB1);

            }
            if (verbose)
            {
                auto stop = std::chrono::steady_clock::now();
                batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                printf("\repoch: %zd/%zd | batch: %zd/%zd | Error: %f (MSE) | speed: %zdms/batch (batch size: %zd)", j+1, epochs, batch, batches, MSE(_A2), batch_time, batch_size);
            }
            freeBatch(batch[0]);
        }
        
        
        W0.setdata(_W0.toHost(), _W0.rows(), _W0.cols());
        B0.setdata(_B0.toHost(), _B0.rows(), _B0.cols());
        W1.setdata(_W1.toHost(), _W1.rows(), _W1.cols());
        B1.setdata(_B1.toHost(), _B1.rows(), _B1.cols());

        // dMatFree(d_W0);
        // dMatFree(d_B0);
        // dMatFree(d_W1);
        // dMatFree(d_B1);
        // dMatFree(velW0);
        // dMatFree(velB0);
        // dMatFree(velW1);
        // dMatFree(velB1);

        #else

        size_t counter = 0;
        for (size_t i = 0; i < data_set.size(); i++, counter++)
        {   
            if (counter == 1)
                start = std::chrono::steady_clock::now();

            // Flatten image data into a big one-dimensional vector
            Matrix A0 = (data_set[i].image_data).flatten(ROW_FLATTEN);

            // Propagate to the first layer (Hidden layer)
            Matrix Z1 = (W0 * A0) + B0;
            Matrix A1 = Z1.apply(sigmoid);

            // Propagate to the second layer (Output layer)
            Matrix Z2 = (W1 * A1) + B1;
            Matrix A2 = Z2.apply(sigmoid);

            // Calculate error
            Matrix yhat = one_hot(data_set[i].label);

            if (counter == batch_size && verbose)
            {
                counter = 0;
                auto stop = std::chrono::steady_clock::now();
                batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                printf("\repoch: %zd/%zd | batch: %zd/%zd | Error: %f (MSE) | speed: %zdms/batch (batch size: %zd)", j+1, epochs, i / batch_size, data_set.size() / batch_size, mean((A2 - yhat).apply([](double in){return in * in;})), batch_time, batch_size);
            }

            Matrix dW1 = hadamard(Z2.apply(sigmoid_d), (A2 - yhat)) * A1.transpose();
            Matrix dB1 = hadamard(Z2.apply(sigmoid_d), (A2 - yhat));
            Matrix dW0 = hadamard(Z1.apply(sigmoid_d), W1.transpose() * (A2 - yhat)) * A0.transpose();
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

#ifdef __GPU_ACCELERATION_CUDA
std::vector<DeviceImage> SimpleNeuralNetwork::allocBatch(std::vector<Image> data_set, size_t offset, size_t batch_size)
{
    if (offset > data_set.size())
        ERR("[ERROR] (SimpleNeuralNetwork::allocBatch) Invalid batch numbers!");

    batch_size = std::min(batch_size, data_set.size() - offset);
    DPDOUBLE batch_data;
    
    std::vector<DeviceImage> device_images(batch_size);

    if (cudaMalloc(&batch_data, (28 * 28 + 10) * sizeof(double) * batch_size) != cudaSuccess)
        ERR("[ERROR] (SimpleNeuralNetwork::allocBatch) Could not allocate GPU memory!");

    for (size_t i = 0; i < batch_size; ++i) {
        DPDOUBLE image_data = batch_data + ((28 * 28 + 10) * i);
        DPDOUBLE label_data = batch_data + ((28 * 28 + 10) * i) + 28 * 28;

        if (cudaMemcpy(image_data, data_set[offset + i].image_data.flatten(COLUMN_FLATTEN).getdata()[0], 28 * 28, cudaMemcpyHostToDevice) != cudaSuccess)
            ERR("[ERROR] (SimpleNeuralNetwork::allocBatch) Could not copy from Host to Device memory!");
        
        if (cudaMemcpy(label_data, one_hot(data_set[offset + i].label).flatten(COLUMN_FLATTEN).getdata()[0], 10, cudaMemcpyHostToDevice) != cudaSuccess)
            ERR("[ERROR] (SimpleNeuralNetwork::allocBatch) Could not copy from Host to Device memory!");

        device_images.push_back({.image = image_data, .label = label_data});
    }

    return device_images;
}

void SimpleNeuralNetwork::freeBatch(DeviceImage first)
{
    if (cudaFree(first.image) != cudaSuccess)
        ERR("[ERROR] (SimpleNeuralNetwork::freeBatch) Could not free GPU memory!");
}
#endif // __GPU_ACCELERATION_CUDA

double SimpleNeuralNetwork::evaluate(std::vector<Image> data_set, bool verbose, size_t batch_size)
{   
    int correct = 0;
    size_t counter = 1;
    size_t time_elapsed = 0;
    std::chrono::steady_clock::time_point start;

    for (size_t i = 0; i < data_set.size(); i++, counter++)
    {   
        
        if (counter == 1)
            start = std::chrono::steady_clock::now();

        auto prediction = predict(data_set[i]);
        if (data_set[i].label == prediction)
            correct++;

        if (counter == std::min(data_set.size(), batch_size))
        {
            counter ^= counter;

            auto stop = std::chrono::steady_clock::now();
            auto batch_time_elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();

            if (verbose)
                printf("\r%zd/%zd %zdms/batch (batch size: %zd)", i+1, data_set.size(), batch_time_elapsed, batch_size);
                
            time_elapsed += batch_time_elapsed;
        }
    }
    if (verbose)
        printf("\nTime elapsed: %zds, avg batch timing --> %.0f ms/batch (batch size %zd)\n -------------------------------------------\n", time_elapsed/1000, (float) time_elapsed / (data_set.size() / batch_size), batch_size);
    return 1.0 * ((double) correct / data_set.size());
}


int SimpleNeuralNetwork::predict(Image i)
{
    return (int) run(i).argmax(COLUMN, 0);
}

Matrix SimpleNeuralNetwork::run(Image i)
{
    
    // Flatten image data into a big one-dimensional vector
    Matrix A0 = i.image_data.flatten(ROW_FLATTEN);

    // Propagate to the first layer (Hidden layer)
    Matrix Z1 = (W0 * A0) + B0;
    Matrix A1 = Z1.apply(sigmoid);
    
    // Propagate to the second layer (Output layer)
    Matrix Z2 = (W1 * A1) + B1;
    Matrix A2 = Z2.apply(sigmoid);
    A2 = softmax(A2);

    return A2;
}

void SimpleNeuralNetwork::run_visual(Image i)
{
    Matrix prediction = run(i);
    std::cout << i << std::endl;
    std::cout << "The model predicted: " << prediction.argmax(COLUMN, 0) << std::endl;
}

void SimpleNeuralNetwork::save(const char* path)
{
    std::ofstream file(path, std::ios::binary);
    for (size_t i = 0; i < W0.rows(); i++)
    {
        for (size_t c = 0; c < W0.cols(); c++)
        {
            file.write((char*) &W0[i][c], 8);
        }
    }

    for (size_t i = 0; i < B0.rows(); i++)
    {
        file.write((char*) &B0[i][0], 8);
    }

    for (size_t i = 0; i < W1.rows(); i++)
    {
        for (size_t c = 0; c < W1.cols(); c++)
        {
            file.write((char*) &W1[i][c], 8);
        }
    }

    for (size_t i = 0; i < B1.rows(); i++)
    {
        file.write((char*) &B1[i][0], 8);
    }
    file.close();
}

void SimpleNeuralNetwork::load(const char* path)
{
    std::ifstream file(path, std::ios::binary);
    for (size_t i = 0; i < W0.rows(); i++)
    {
        for (size_t c = 0; c < W0.cols(); c++)
        {
            file.read((char*)&W0[i][c], 8);
        }
    }

    for (size_t i = 0; i < B0.rows(); i++)
    {
        file.read((char*)&B0[i][0], 8);
    }

    for (size_t i = 0; i < W1.rows(); i++)
    {
        for (size_t c = 0; c < W1.cols(); c++)
        {
            file.read((char*)&W1[i][c], 8);
        }
    }

    for (size_t i = 0; i < B1.rows(); i++)
    {
        file.read((char*)&B1[i][0], 8);
    }
    file.close();
}

#ifdef __GPU_ACCELERATION_CUDA
double MSE(DeviceMatrix subtracted)
{
    Matrix hB(10, 1);

    hB.setdata(subtracted.toHost(), subtracted.rows(), subtracted.cols());

    return mean(hB.apply([](double in) {return pow(in, 2);}));
}
#endif

Matrix one_hot(int label)
{
    Matrix one_hot_l(10, 1);

    *one_hot_l[label] = 1.0;

    return one_hot_l;
}

double ReLU(double in)
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
    Matrix ones(in.rows(), in.cols());
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
    for (size_t i = 0; i < in.rows(); i++)
        for (size_t j = 0; j < in.cols(); j++)
            sum += in[i][j];
    
    return sum / (in.rows() * in.cols());
}
