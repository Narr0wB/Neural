
#include "neural.h"

void SimpleNeuralNetwork::train(std::vector<Image> data_set, size_t epochs, double alpha, bool verbose, size_t batch_size)
{
    auto train_start = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point start;
    size_t batch_time;

    if (data_set.size() < batch_size)
    {
        batch_size = data_set.size();
    }

    #ifdef __GPU_ACCELERATION_CUDA
    // Layer 0 INPUT ------------------
    DMATRIX _A0 = DMatrixCreate(28 * 28, 1);;
    DMATRIX _W0 = copyHostToDevice(W0.getdata(), W0.rows(), W0.cols());
    DMATRIX _B0 = copyHostToDevice(B0.getdata(), B0.rows(), B0.cols());

    DMATRIX _dW0 = DMatrixCreate(W0.rows(), W0.cols());
    DMATRIX _dB0 = DMatrixCreate(B0.rows(), B0.cols());

    // Layer 1 HIDDEN -----------------
    DMATRIX _A1 = DMatrixCreate(W1.cols(), 1);
    DMATRIX _Z1 = DMatrixCreate(W1.cols(), 1);
    DMATRIX _W1 = copyHostToDevice(W1.getdata(), W1.rows(), W1.cols());
    DMATRIX _B1 = copyHostToDevice(B1.getdata(), B1.rows(), B1.cols());;

    DMATRIX _dW1 = DMatrixCreate(W1.rows(), W1.cols());
    DMATRIX _dB1 = DMatrixCreate(B1.rows(), B1.cols());

    // Layer 2 OUTPUT -----------------
    DMATRIX _A2 = DMatrixCreate(W1.rows(), 1);
    DMATRIX _Z2 = DMatrixCreate(W1.rows(), 1);

    // TEMP VARS ----------------------
    DMATRIX _temp_W1_transposed = DMatrixCreate(W1.cols(), W1.rows());
    DMATRIX _temp_A1 = DMatrixCreate(W1.cols(), 1);

    
    DPDOUBLE batch_memory = allocBatch(batch_size);
    std::vector<DIMAGE> batch;

    #else

    Matrix E;

    #endif // __GPU_ACCELERATION_CUDA

    size_t batches = data_set.size() / batch_size;
    if (data_set.size() % batch_size != 0) batches++;


    for (size_t j = 0; j < epochs; j++)
    {
        printf("\n");

        for (size_t b = 0; b < batches; ++b) {
            start = std::chrono::steady_clock::now();

    #ifdef __GPU_ACCELERATION_CUDA
            batch = copyBatch(data_set, b * batch_size, batch_memory, batch_size);
            
            for (size_t i = 0; i < batch.size(); i++)
            {              
                DMATRIX transposed_A0 = batch[i].image;
                
                DMatrixTranspose(transposed_A0, _A0);
                
                // Propagate to the first layer (Hidden layer)
                cudaMemset(_Z1.data, 0, _Z1.rows * _Z1.cols * sizeof(double));
                denseForwardProp(_Z1, _W0, _A0, _B0);
                DMatrixApply(_Z1, _A1, SIGMOID);
                
                // Propagate to the second layer (Output layer)
                cudaMemset(_Z2.data, 0, _Z2.rows * _Z2.cols * sizeof(double));
                denseForwardProp(_Z2, _W1, _A1, _B1);
                DMatrixApply(_Z2, _A2, SIGMOID);

                // Backpropagate the first layer
                DMatrixSubtract(_A2, batch[i].label, _A2);
                denseBackProp(_dW1, _dB1, _A2, _Z2, _A1, SIGMOID_DERIVATIVE);

                // Backpropagate the second layer
                DMatrixTranspose(_W1, _temp_W1_transposed);
                DMatrixDot(_temp_W1_transposed, _A2, _temp_A1);

                denseBackProp(_dW0, _dB0, _temp_A1, _Z1, _A0, SIGMOID_DERIVATIVE);
                          
                // Apply the gradients to the weights and biases
                DMatrixScale(_dW0, _dW0, alpha);
                DMatrixSubtract(_W0, _dW0, _W0);

                DMatrixScale(_dB0, _dB0, alpha);
                DMatrixSubtract(_B0, _dB0, _B0);

                DMatrixScale(_dW1, _dW1, alpha);
                DMatrixSubtract(_W1, _dW1, _W1);

                DMatrixScale(_dB1, _dB1, alpha);
                DMatrixSubtract(_B1, _dB1, _B1);
            }

            if (verbose)
            {
                auto stop = std::chrono::steady_clock::now();
                batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                printf("\repoch: %zd/%zd | batch: %zd/%zd | Error: %f (MSE) | speed: %zdms/batch (batch size: %zd)", j+1, epochs, b+1, batches, MSE(_A2), batch_time, batch_size);
            }

            

        #else

            for (size_t i = 0; i < batch_size; i++)
            {   
                // Flatten image data into a big one-dimensional vector
                Matrix A0 = data_set[b * batch_size + i].image_data.flatten(COLUMN, COLUMN_FLATTEN);

                // Propagate to the first layer (Hidden layer)
                Matrix Z1 = (W0 * A0) + B0;
                Matrix A1 = Z1.apply(sigmoid);

                // Propagate to the second layer (Output layer)
                Matrix Z2 = (W1 * A1) + B1;
                Matrix A2 = Z2.apply(sigmoid);

                // Calculate error
                Matrix yhat = one_hot(data_set[b * batch_size + i].label);

                E = (A2 - yhat);

                Matrix dW1 = hadamard(Z2.apply(sigmoid_d), (A2 - yhat)) * A1.transpose();
                Matrix dB1 = hadamard(Z2.apply(sigmoid_d), (A2 - yhat));
                Matrix dW0 = hadamard(Z1.apply(sigmoid_d), W1.transpose() * (A2 - yhat)) * A0.transpose();
                Matrix dB0 = hadamard(Z1.apply(sigmoid_d), W1.transpose() * (A2 - yhat));

                W0 = W0 - (alpha * dW0);
                W1 = W1 - (alpha * dW1);
                B0 = B0 - (alpha * dB0);
                B1 = B1 - (alpha * dB1);

            }

            if (verbose)
            {
                auto stop = std::chrono::steady_clock::now();
                batch_time = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start).count();
                printf("\repoch: %zd/%zd | batch: %zd/%zd | Error: %f (MSE) | speed: %zdms/batch (batch size: %zd)", j+1, epochs, b+1, batches, mean(E.apply([](double in) {return pow(in, 2);})), batch_time, batch_size);
            }
        #endif

            
        }
        
    }

#ifdef __GPU_ACCELERATION_CUDA
    freeBatch(batch_memory);

    W0.setdata(copyDeviceToHost(_W0), _W0.rows, _W0.cols);
    B0.setdata(copyDeviceToHost(_B0), _B0.rows, _B0.cols);
    W1.setdata(copyDeviceToHost(_W1), _W1.rows, _W1.cols);
    B1.setdata(copyDeviceToHost(_B1), _B1.rows, _B1.cols);
#endif

    auto train_stop = std::chrono::steady_clock::now();
    double epoch_time = (double) std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count() / epochs;
    if (verbose)
        printf("\n\nTime elapsed: %zds, ~%.2fs per epoch\n", std::chrono::duration_cast<std::chrono::seconds>(train_stop - train_start).count(), epoch_time);
}

#ifdef __GPU_ACCELERATION_CUDA
DPDOUBLE SimpleNeuralNetwork::allocBatch(size_t batch_size) {
    DPDOUBLE batch_data;

    if (cudaMalloc(&batch_data, (28 * 28 + 10) * sizeof(double) * batch_size) != cudaSuccess)
        ERR("[ERROR] (SimpleNeuralNetwork::allocBatch) Could not allocate GPU memory!");

    return batch_data;
}

std::vector<DIMAGE> SimpleNeuralNetwork::copyBatch(std::vector<Image>& data_set, size_t offset, DPDOUBLE batch_data, size_t batch_size)
{
    if (offset > data_set.size())
        ERR("[ERROR] (SimpleNeuralNetwork::copyBatch) Invalid batch numbers!");

    if (batch_data == NULL) {
        ERR("[ERROR] (SimpleNeuralNetwork::copyBatch) batch_data NULL!");
    }

    batch_size = std::min(batch_size, data_set.size() - offset);
    
    std::vector<DIMAGE> device_images;

    for (size_t i = 0; i < batch_size; ++i) { 
        DPDOUBLE image_data = batch_data + ((28 * 28 + 10) * i);
        DPDOUBLE label_data = batch_data + ((28 * 28 + 10) * i) + 28 * 28;

        __hostToDeviceData(data_set[offset + i].image_data.flatten(ROW, COLUMN_FLATTEN).getdata(), image_data, 1, 28*28);
        __hostToDeviceData(one_hot(data_set[offset + i].label).flatten(ROW, ROW_FLATTEN).getdata(), label_data, 1, 10);
        
        DMATRIX image{image_data, 1, 28*28};
        DMATRIX label{label_data, 10, 1};

        DIMAGE d_image{image, label};

        device_images.push_back(d_image);
    }

    return device_images;
}

void SimpleNeuralNetwork::freeBatch(DPDOUBLE batch_data)
{
    if (cudaFree(batch_data) != cudaSuccess)
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

            if (verbose) {
                printf("\r                                                                                                 ");
                printf("\rbatch: %.0f/%.0f | Accuracy: %0.2f %% | speed: %zdms/batch (batch size: %zd)", ceil(i / batch_size) + 1, ceil(data_set.size() / batch_size), (float)correct/i * 100.0f, batch_time_elapsed, batch_size);
            }

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
    Matrix A0 = i.image_data.flatten(COLUMN, COLUMN_FLATTEN);

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
double MSE(DMATRIX subtracted)
{
    Matrix hB(10, 1);

    hB.setdata(copyDeviceToHost(subtracted), subtracted.rows, subtracted.cols);

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
