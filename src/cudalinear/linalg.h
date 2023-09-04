
#ifndef LINALG_H
#define LINALG_H

#include <math.h>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

#define COLUMN_FLATTEN 104
#define ROW_FLATTEN 103

#define COLUMN 1
#define ROW 0

#define ERR(msg) {std::cerr << msg << std::endl; exit(EXIT_FAILURE);}

class Matrix 
{
    private:
        double** data;
        std::mt19937 rnd;

        size_t _rows = 0;
        size_t _cols = 0;    
          
    public:
        Matrix(size_t r, size_t c, double f) : _rows(r), _cols(c) 
        {
            if (r < 0 || c < 0)
            {
                return;
            }
            
            data = new double*[r];
            for (size_t i = 0; i < r; ++i) data[i] = new double[c];
            
            fill(f);
        };

        Matrix(size_t r, size_t c) : _rows(r), _cols(c) 
        {
            if (r < 0 || c < 0)
            {
                return;
            }
            
            data = new double*[r];
            for (size_t i = 0; i < r; ++i) data[i] = new double[c];
            
            fill(0);
        };

        Matrix(const Matrix& m) 
        {
            _rows = m._rows;
            _cols = m._cols;

            data = new double*[_rows];
            for (size_t i = 0; i < _rows; ++i) {
                data[i] = new double[_cols];
                memcpy(data[i], m.data[i], _cols * sizeof(double));
            }
        };

        Matrix(double** _data, size_t rows, size_t cols) : data(_data), _rows(rows), _cols(cols) {};
        Matrix() {};

        ~Matrix() 
        {
            if (data == nullptr) return;

            
            for (size_t i = 0; i < _rows; ++i) {
                delete data[i];
            }
                

            delete data;
            
        };

        inline size_t rows() const { return _rows; }
        inline size_t cols() const { return _cols; }

        // Matrix Operations --------------------------------------------------------------------------------
        double* operator[](size_t idx);
        Matrix operator+(const Matrix& m) const;
        Matrix operator-(const Matrix& m) const;
        Matrix operator*(const Matrix& m) const;
        Matrix operator*(double s) const;

        friend Matrix operator*(double s, const Matrix& m);
        Matrix& operator=(const Matrix& m);
        bool operator==(Matrix& m) const;

        // Misc Matrix Operations ---------------------------------------------------------------------------
        Matrix flatten(int options) const;
        // ------
        Matrix merge(const Matrix& m) const;
        // ------
        Matrix apply(double (*func)(double)) const;
        Matrix transpose() const;

        void fill(double f);
        void randn(double mean, double stddev);
        void setdata(double** n_data, size_t n_rows, size_t n_cols);

        int resize(size_t new_rows, size_t new_cols);
        double maxarg(int options, size_t index) const;
        double minarg(int options, size_t index) const;
        double** getdata();
        size_t argmax(int options, size_t index) const;
        
        // Matrix Friend Operations -------------------------------------------------------------------------
        friend std::ostream& operator<<(std::ostream &os, const Matrix& m);
        friend Matrix mabs(const Matrix& m);
        friend Matrix hadamard(Matrix m, Matrix d);
        friend Matrix softmax(Matrix m);
        friend double sum(Matrix m);
};

struct TensorShape
{
    size_t x;
    size_t y;

    TensorShape(size_t u_x, size_t u_y) : x(u_x), y(u_y) {}
    TensorShape() : x(0), y(0) {} 

    bool operator==(const TensorShape& t) const 
    {
        if (t.x == x && t.y == y)
            return true;
        
        return false;
    }

    bool operator==(uint32_t i)
    {
        if (x == i && y == i)
            return true;

        return false;
    }
};

size_t e_log10(double arg);

#endif