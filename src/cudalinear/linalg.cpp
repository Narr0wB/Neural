
#include "linalg.h"

int Matrix::resize(size_t new_rows, size_t new_cols)
{
    double** new_data = new double*[new_rows];

    if (new_data == nullptr) 
        ERR("[ERROR] (Matrix::resize) Error allocating memory!");

    for (int i = 0; i < new_rows; ++i) {
        new_data[i] = new double[new_cols];

        if (new_data[i] == nullptr) 
            ERR("[ERROR] (Matrix::resize) Error allocating memory!");

        if (i < _rows) {
            memcpy(new_data[i], data[i], std::min(_cols, new_cols));
            delete data[i];
        }
    }

    if (new_rows < _rows) {
        for (int i = new_rows; i < _rows; ++i)
            delete data[i];
    }

    delete data;

    data = new_data;

    return 1;
}

double* Matrix::operator[](size_t idx)
{
    if (idx > _rows)
        ERR("[ERROR] Invalid matrix index!");

    return data[idx];
}

Matrix Matrix::operator+(const Matrix& m) const
{
    if (this->_rows != m._rows || 
        this->_cols != m._cols) 
        ERR("[ERROR] (Matrix::operator+) Invalid matrices for the \"+\" operator")
    
    Matrix sum(this->_rows, this->_cols);
    
    for (size_t i = 0; i < _rows; i++)
    {
        for (size_t j = 0; j < _cols; j++)
        {
            sum.data[i][j] = m.data[i][j] + this->data[i][j];
        }
    }

    return sum;
}

Matrix Matrix::operator-(const Matrix& m) const
{
    if (this->_rows != m._rows || 
        this->_cols != m._cols) 
        ERR("[ERROR] (Matrix::operator-) Invalid matrices for the \"-\" operator")
    
    Matrix diff(this->_rows, this->_cols);
    
    for (size_t i = 0; i < _rows; i++)
    {
        for (size_t j = 0; j < _cols; j++)
        {
            diff.data[i][j] = this->data[i][j] - m.data[i][j] ;
        }
    }

    return diff;
}

// Matrix Multiplication
Matrix Matrix::operator*(const Matrix& m) const
{   
    // Check if the first matrix has the same sumber of colums as the second's number of _rows, if not return
    if ((this->_cols != m._rows))
        ERR("[ERROR] (Matrix::operator*) Invalid matrices for the \"*\" operator")

    Matrix multp(this->_rows, m._cols);

    for (size_t i = 0; i < this->_rows; i++)
    {
        for (size_t j = 0; j < m._cols; j++)
        {
            double sum = 0;
            for (size_t k = 0; k < m._rows; k++)
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

    for (size_t i = 0; i < this->_rows; i++)
    {
        for (size_t j = 0; j < this->_cols; j++)
        {
            multp.data[i][j] *= s;
        }
    }

    return multp;
}

Matrix operator*(double s, const Matrix& m)
{
    Matrix multp(m._rows, m._cols);

    for (size_t i = 0; i < m._rows; i++)
    {
        for (size_t j = 0; j < m._cols; j++)
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

    for (int i = 0; i < _rows; ++i)
        delete data[i];

    delete data;

    _rows = m._rows;
    _cols = m._cols;

    double** new_data = new double*[_rows];
    for (int i = 0; i < _rows; ++i) {
        new_data[i] = new double[_cols];

        memcpy(new_data[i], m.data[i], _cols * sizeof(double));
    }

    data = new_data;
    return *this;
}

bool Matrix::operator==(Matrix& m) const
{   
    if (this->_rows != m._rows ||
        this->_cols != m._cols)
        return false;
    
    for (size_t i = 0; i < this->_rows; i++)
        for (size_t j = 0; j < this->_cols; j++)
        {
            if (this->data[i][j] != m.data[i][j])
                return false;
        }

    return true;
}

/*
Matrix Matrix::merge(const Matrix& m) const
{
    if ((_rows != 1 &&
        _cols != 1 &&
        m._rows != 1 &&
        m._cols != 1) ||
        (_rows != m._rows &&
        _cols != m._cols))
        ERR("[ERROR] (Matrix::merge) Invalid matrices sizes!")
    
    Matrix merged = *this;

    if (_rows == 1)
    {
        merged.resize(_rows, _cols + m._cols);
        for (size_t i = 0; i < m._cols; i++)
            merged.data[0][i + _cols] = m.data[0][i];
    }
    else
    {
        merged.resize(_rows + m._rows, _cols);
        for (size_t i = 0; i < m._rows; i++)
            merged.data[i + _rows][0] = m.data[i][0];
    }

    return merged;
}
*/

Matrix Matrix::flatten(int options) const
{
    if (options != COLUMN_FLATTEN &&
        options != ROW_FLATTEN)
        ERR("[ERROR] (Matrix::flatten) Invalid options!")
    

    Matrix flat(this->_rows * this->_cols, 1);

    if (options == COLUMN_FLATTEN)
    { 
        for (size_t j = 0; j < this->_cols; j++)
            for (size_t i = 0; i < this->_rows; i++)
                flat.data[j * this->_rows + i][0] = this->data[i][j];
    }
    else 
    {
        for (size_t i = 0; i < this->_rows; i++)
            for (size_t j = 0; j < this->_cols; j++)
                flat.data[i * this->_cols + j][0] = this->data[i][j];
    } 

    return flat;
}

void Matrix::fill(double f) 
{
    for (size_t i = 0; i < this->_rows; i++)
        for (size_t j = 0; j < this->_cols; j++)
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
        for (size_t i = 0; i < this->_rows; i++)
            maxarg = std::max(maxarg, this->data[i][index]);
    }
    else 
    {
        for (size_t j = 0; j < this->_cols; j++)
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
        for (size_t i = 0; i < this->_rows; i++)
            minarg = std::min(minarg, this->data[i][index]);
    }
    else 
    {
        for (size_t j = 0; j < this->_cols; j++)
            minarg = std::min(minarg, this->data[index][j]);
    }

    return minarg;
}

size_t Matrix::argmax(int options, size_t index) const
{
    if (options != COLUMN &&
        options != ROW)
        ERR("[ERROR] Invalid arguments!")
    
    double maxarg = -INFINITY;
    size_t idx = 0;

    if (options == COLUMN)
    {
        for (size_t i = 0; i < this->_rows; i++)
            if (this->data[i][index] > maxarg)
            {
                idx = i;
                maxarg = this->data[i][index];
            }
    }
    else 
    {
        for (size_t j = 0; j < this->_cols; j++)
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
    for (size_t i = 0; i < this->_rows; i++)
        for (size_t j = 0; j < this->_cols; j++)
            this->data[i][j] = std::normal_distribution<double>(mean, stddev)(rnd);
}

Matrix Matrix::apply(double (*func)(double)) const
{
    Matrix applied(this->_rows, this->_cols);

    for (size_t i = 0; i < this->_rows; i++)
        for (size_t j = 0; j < this->_cols; j++)
            applied.data[i][j] = func(this->data[i][j]);
    
    return applied;
}

Matrix Matrix::transpose() const
{
    Matrix transposed(this->_cols, this->_rows);

    for (size_t i = 0; i < this->_cols; i++)
        for (size_t j = 0; j < this->_rows; j++)
            transposed.data[i][j] = this->data[j][i];

    return transposed;
}

// WARNING: The matrix object on which this method is called will then own the data pointer!
void Matrix::setdata(double** n_data, size_t n_rows, size_t n_cols)
{
    for (int i = 0; i < _rows; ++i) 
        delete data[i];

    delete data;

    data = n_data;

    _rows = n_rows;
    _cols = n_cols;
}

double** Matrix::getdata()
{
    return data;
}

std::ostream& operator<<(std::ostream& os, const Matrix& m) 
{   
    os << std::setprecision(2) << std::fixed;
    os << std::endl;
    std::string tile("----");
    std::string t_tile = tile;
    std::string row("+");

    for (size_t i = 0; i < m._cols; i++)
    {   
        if (m.minarg(COLUMN, i) < 0)
            t_tile += "-";
        for (size_t j = 0; j < e_log10(mabs(m).maxarg(COLUMN, i)); j++)
            t_tile += "-";
        t_tile += "+";
        row += t_tile;
        t_tile = tile;
    }
    os << row << std::endl;

    std::string p("|");

    for (size_t i = 0; i < m._rows; i++)
    {
        os << p;
        for (size_t g = 0; g < m._cols; g++)
        {
            os << m.data[i][g];
            if (m.minarg(COLUMN, g) < 0 && m.data[i][g] >= 0)
            {   
                os << " ";
            }    
            if (e_log10(m.data[i][g]) < e_log10(mabs(m).maxarg(COLUMN, g)))
            {
                for (size_t f = 0; f < e_log10(mabs(m).maxarg(COLUMN, g))-e_log10(m.data[i][g]); f++)
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

    for (size_t i = 0; i < g._rows; i++)
        for (size_t j = 0; j < g._cols; j++)
            g[i][j] = std::abs(g[i][j]);
    
    return g;
}

Matrix hadamard(Matrix m, Matrix d)
{
    if (m._cols != d._cols ||
        m._rows != d._rows)
        ERR("[ERROR] (Matrix::hadamard) Invalid matrix sizes!")

    Matrix product(m._rows, m._cols);

    for (size_t i = 0; i < m._rows; i++)
        for (size_t j = 0; j < m._cols; j++)
            product[i][j] = (m[i][j] * d[i][j]);
    
    return product;
}


Matrix softmax(Matrix m)
{
    Matrix T(m._rows, m._cols);

    double sum = 0;

    for (size_t i = 0; i < m._rows; i++)
    {
        sum += exp(m[i][0]);
    }

    for (size_t i = 0; i < m._rows; i++)
    {
        T[i][0] = exp(m[i][0]) / sum;
    }

    return T;
}

double sum(Matrix m)
{
    double temp = 0;
    
    for (size_t i = 0; i < m._rows; i++)
        for (size_t j = 0; j < m._cols; j++)
            temp += m.data[i][j];

    return temp;
}

size_t e_log10(double arg)
{  
    if (std::log10(std::abs(arg)) < 0)
        return 0;
    
    return (size_t) std::log10(std::abs(arg));
}