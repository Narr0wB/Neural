
#include "linalg.h"

int Matrix::resize(int newR, int newC)
{
    double** new_data = new double*[newR];

    if (new_data == nullptr)
        return -1;
    
    for (int i = 0; i < newR; i++)
    {
        new_data[i] = new double[newC];
    }
    
    for (int i = 0; i < newR; i++)
    {
        for (int j = 0; j < newC; j++)
        {
            if (i < this->rows && j < this->cols)
            {
                new_data[i][j] = this->data[i][j];
            }
            else
            {
                new_data[i][j] = 0.0;
            }
        }
    }
 
    for (int i = 0; i < this->rows; i++)
        delete[] this->data[i];
    delete[] this->data;
    
    this->rows = newR;
    this->cols = newC;

    this->data = new_data;
    return 1;
}

double* Matrix::operator[](size_t idx)
{
    if (idx > rows)
    {
        throw "ACCESS ERROR";
    }

    return data[idx];
}

Matrix Matrix::operator+(const Matrix& m) const
{
    if (this->rows != m.rows || 
        this->cols != m.cols) 
        throw "INVALID MATRICES";
    
    Matrix sum(this->rows, this->cols);
    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            sum.data[i][j] = m.data[i][j] + this->data[i][j];
        }
    }

    return sum;
}

Matrix Matrix::operator-(const Matrix& m) const
{
    if (this->rows != m.rows || 
        this->cols != m.cols) 
        throw "INVALID MATRICES";
    
    Matrix diff(this->rows, this->cols);
    
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            diff.data[i][j] = this->data[i][j] - m.data[i][j] ;
        }
    }

    return diff;
}

// matrix multiplication
Matrix Matrix::operator*(const Matrix& m) const
{   
    // Check if the first matrix has the same sumber of colums as the second's number of rows, if not return
    if ((this->cols != m.rows))
        throw "INVALID MATRICES";

    Matrix multp(this->rows, m.cols);

    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            double sum = 0;
            for (int k = 0; k < m.rows; k++)
            {
                sum += this->data[i][k] * m.data[k][j];
            }
            multp.data[i][j] = sum;
        }
    }
        
    return multp;
}

Matrix Matrix::operator*(double s) const
{
    Matrix multp = *this;

    for (int i = 0; i < this->rows; i++)
    {
        for (int j = 0; j < this->cols; j++)
        {
            multp.data[i][j] *= s;
        }
    }

    return multp;
}

Matrix operator*(double s, const Matrix& m)
{
    Matrix multp(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++)
    {
        for (int j = 0; j < m.cols; j++)
        {
            multp.data[i][j] = s * m.data[i][j];
        }
    }

    return multp;
}

Matrix& Matrix::operator=(const Matrix& m)
{    
    if (this == &m)
        return *this;

    this->resize(m.rows, m.cols);
    
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            this->data[i][j] = m.data[i][j];
    
    return *this;
}

bool Matrix::operator==(Matrix& m) const
{   
    if (this->rows != m.rows ||
        this->cols != m.cols)
        return false;
    
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
        {
            if (this->data[i][j] != m.data[i][j])
                return false;
        }

    return true;
}

Matrix Matrix::flatten(int options) const
{
    if (options != COLUMN_FLATTEN &&
        options != ROW_FLATTEN)
        throw "INVALID OPTIONS";
    

    Matrix flat(this->rows * this->cols, 1);

    size_t idx = 0;
    if (options == COLUMN_FLATTEN)
    { 
        for (int i = 0; i < this->rows; i++)
            for (int j = 0; j < this->cols; j++)
                flat.data[j * this->rows + i][0] = this->data[i][j];
    }
    else 
    {
        for (int i = 0; i < this->rows; i++) 
            for (int j = 0; j < this->cols; j++, idx++)
                flat.data[i * this->cols + j][0] = this->data[i][j];
    } 

    return flat;
}

void Matrix::fill(double f) 
{
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            this->data[i][j] = f;
}

double Matrix::maxarg(int options, size_t index) const
{   
    if (options != COLUMN &&
        options != ROW)
        return NAN;
    
    double maxarg = -INFINITY;

    if (options == COLUMN)
    {
        for (int i = 0; i < this->rows; i++)
            maxarg = std::max(maxarg, this->data[i][index]);
    }
    else 
    {
        for (int j = 0; j < this->cols; j++)
            maxarg = std::max(maxarg, this->data[index][j]);
    }

    return maxarg;
}

double Matrix::minarg(int options, size_t index) const
{   
    if (options != COLUMN &&
        options != ROW)
        return NAN;
    
    double minarg = INFINITY;

    if (options == COLUMN)
    {
        for (int i = 0; i < this->rows; i++)
            minarg = std::min(minarg, this->data[i][index]);
    }
    else 
    {
        for (int j = 0; j < this->cols; j++)
            minarg = std::min(minarg, this->data[index][j]);
    }

    return minarg;
}

size_t Matrix::argmax(int options, size_t index) const
{
    if (options != COLUMN &&
        options != ROW)
        return NAN;
    
    double maxarg = -INFINITY;
    size_t idx = 0;

    if (options == COLUMN)
    {
        for (int i = 0; i < this->rows; i++)
            if (this->data[i][index] > maxarg)
            {
                idx = i;
                maxarg = this->data[i][index];
            }
    }
    else 
    {
        for (int j = 0; j < this->cols; j++)
            if (this->data[index][j] > maxarg)
            {
                idx = j;
                maxarg = this->data[index][j];
            }
    }

    return idx;
}

void Matrix::randn(double mean, double stddev)
{
    rnd.seed(std::chrono::steady_clock::now().time_since_epoch().count());
    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            this->data[i][j] = std::normal_distribution<double>(mean, stddev)(rnd);
}

Matrix Matrix::apply(double (*func)(double)) const
{
    Matrix applied(this->rows, this->cols);

    for (int i = 0; i < this->rows; i++)
        for (int j = 0; j < this->cols; j++)
            applied.data[i][j] = func(this->data[i][j]);
    
    return applied;
}

Matrix Matrix::transpose() const
{
    Matrix transposed(this->cols, this->rows);

    for (int i = 0; i < this->cols; i++)
        for (int j = 0; j < this->rows; j++)
            transposed.data[i][j] = this->data[j][i];

    return transposed;
}

void Matrix::setdata(double** Ndata, size_t Nrows, size_t Ncols)
{
    for (int i = 0; i < rows; i++)
        delete[] data[i];
    delete[] data;

    this->data = Ndata;

    this->rows = Nrows;
    this->cols = Ncols;
}

double** Matrix::getdata()
{
    return this->data;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) 
{   
    os << std::setprecision(2) << std::fixed;
    os << std::endl;
    std::string tile("----");
    std::string t_tile = tile;
    std::string row("+");

    for (int i = 0; i < m.cols; i++)
    {   
        if (m.minarg(COLUMN, i) < 0)
            t_tile += "-";
        for (int j = 0; j < e_log10(mabs(m).maxarg(COLUMN, i)); j++)
            t_tile += "-";
        t_tile += "+";
        row += t_tile;
        t_tile = tile;
    }
    os << row << std::endl;

    std::string p("|");

    for (int i = 0; i < m.rows; i++)
    {
        os << p;
        for (int g = 0; g < m.cols; g++)
        {
            os << m.data[i][g];
            if (m.minarg(COLUMN, g) < 0 && m.data[i][g] >= 0)
            {   
                os << " ";
            }    
            if (e_log10(m.data[i][g]) < e_log10(mabs(m).maxarg(COLUMN, g)))
            {
                for (int f = 0; f < e_log10(mabs(m).maxarg(COLUMN, g))-e_log10(m.data[i][g]); f++)
                    os << " ";
            }
            os << p;
        }
        os << std::endl << row << std::endl;
    }


    return os;
}

Matrix mabs(const Matrix& m)
{
    Matrix g = m;

    for (int i = 0; i < g.rows; i++)
        for (int j = 0; j < g.cols; j++)
            g[i][j] = std::abs(g[i][j]);
    
    return g;
}

Matrix hadamard(Matrix m, Matrix d)
{
    if (m.cols != d.cols ||
        m.rows != d.rows)
        throw "INVALID MATRICES";

    Matrix product(m.rows, m.cols);

    for (int i = 0; i < m.rows; i++)
        for (int j = 0; j < m.cols; j++)
            product[i][j] = (m[i][j] * d[i][j]);
    
    return product;
}


Matrix softmax(Matrix m)
{
    Matrix T(m.rows, m.cols);

    double sum = 0;

    for (int i = 0; i < m.rows; i++)
    {
        sum += exp(m[i][0]);
    }

    for (int i = 0; i < m.rows; i++)
    {
        T[i][0] = exp(m[i][0]) / sum;
    }

    return T;
}

int e_log10(double arg)
{  
    if (std::log10(std::abs(arg)) < 0)
        return 0;
    
    return (int) std::log10(std::abs(arg));
}