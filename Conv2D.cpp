
#include "Conv2D.h"

Matrix pad(Matrix m, int padding)
{
    if (padding == 0)
        return m;

    Matrix padded(m.rows + 2 * padding, m.cols + 2 * padding);

    for (size_t i = 0; i < m.rows; i++)
        for (size_t j = 0; j < m.rows; j++)
            padded[i + padding][j + padding] = m[i][j];
    
    return padded;
}

MatrixList Conv2D(MatrixList input, MatrixList kernels, int options, uint32_t stride)
{
    if (options != VALID &&
        options != SAME  &&
        options != FULL  ||
        kernels.size() == 0 ||
        input.size() == 0   ||
        stride == 0)
        throw "INVALID OPTIONS";
    
    size_t padding = options;
    if (padding == FULL)
        padding = std::max(kernels[0].rows, kernels[0].cols) - 1;
    
    MatrixList outputs(kernels.size() * input.size());
    for (size_t i = 0; i < input.size(); i++)
    {
        for (size_t j = 0; j < kernels.size(); j++)
        {
            outputs[i * kernels.size() + j] = Matrix(((input[i].rows - kernels[j].rows + 2*padding) / stride) + 1, ((input[i].cols - kernels[j].cols + 2*padding) / stride) + 1);
            input[i] = pad(input[i], padding);
            for (size_t g = 0; g <= (input[i].rows - kernels[j].rows); g += stride)
                for (size_t k = 0; k <= (input[i].cols - kernels[j].cols); k += stride)
                {
                    double convSum = 0;
                    for (size_t r = 0; r < kernels[j].rows; r++)
                        for (size_t c = 0; c < kernels[j].cols; c++)
                            if (r + g < input[i].rows && c + k < input[i].cols)
                                convSum += input[i][r + g][c + k] * kernels[j][r][c];
                    outputs[i * kernels.size() + j][g / stride][k / stride] = convSum;
                }
            if (options == SAME)
                outputs[i * kernels.size() + j].resize(input[i].rows - 2, input[i].cols - 2);
        }
    }

    return outputs;
}

Matrix flatten(MatrixList& m, int options)
{
    if (m.size() == 0 ||
        (options != ROW_FLATTEN &&
         options != COLUMN_FLATTEN))
        throw "INVALID INPUT";

    Matrix flattened(0, 1);

    for (size_t i = 0; i < m.size(); i++)  
        flattened = flattened.merge(m[i].flatten(options));

    return flattened;
}

// NOTE: The flattened matrices must all have the same size
MatrixList dflatten(Matrix& m, TensorShape shape, int matrices, int options)
{
    if (m.rows == 0 ||
        m.cols == 0 ||
        m.rows != (shape.x * shape.y * matrices) ||
        shape == 0 ||
        matrices == 0 ||
        (options != ROW_FLATTEN &&
         options != COLUMN_FLATTEN))
        throw "INVALID INPUT";

    MatrixList deflattened(matrices, shape.x, shape.y);

    if (options == ROW_FLATTEN)
    {
        for (int k = 0; k < matrices; k++)
            for (size_t i = 0; i < shape.x; i++)
                for (size_t j = 0; j < shape.y; j++)
                    deflattened[k][i][j] = m[k * (shape.y * shape.x) + i * shape.y + j][0];
    }
    else 
    {
        for (int k = 0; k < matrices; k++)
            for (size_t i = 0; i < shape.y; i++)
                for (size_t j = 0; j < shape.x; j++)
                    deflattened[k][i][j] = m[k * (shape.y * shape.x) + j * shape.x + i][0];
    }

    return deflattened;
}