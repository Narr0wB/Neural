
#ifndef LINALG_INCLUDED

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

struct Matrix 
{
    private:
        mutable double** data;
        std::mt19937 rnd;
        
    public:
        mutable int rows = 0;
        mutable int cols = 0;

        Matrix(int r, int c) : rows(r), cols(c) 
        {
            if (r < 0 || c < 0)
            {
                return;
            }
            
            data = new double*[r];
            for (int i = 0; i < r; i++)
                data[i] = new double[c];
            
            fill(0);
        };

        Matrix(const Matrix& m) 
        {
            rows = m.rows;
            cols = m.cols;

            data = new double*[rows];
            for (int i = 0; i < rows; i++)
                data[i] = new double[cols];

            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    data[r][c] = m.data[r][c];
        };

        Matrix() 
        {
            data = new double*[rows];
            for (int i = 0; i < rows; i++)
                data[i] = new double[cols];
        };

        ~Matrix() 
        {
            for (int i = 0; i < rows; i++)
                delete[] data[i];
            delete[] data;
        };

        // matrix operations
        double* operator[](size_t idx);
        Matrix operator+(const Matrix& m) const;
        Matrix operator-(const Matrix& m) const;
        Matrix operator*(const Matrix& m) const;
        Matrix operator*(double s) const;
        Matrix& operator=(const Matrix& m);
        friend Matrix operator*(double s, const Matrix& m);
        bool operator==(Matrix& m) const;

        // Print the matrix (std::cout)
        friend std::ostream& operator<<(std::ostream &os, const Matrix& m);

        Matrix flatten(int options) const;
        // Fill the matrix with f
        void fill(double f);
        // Resize the matrix to a new dimension given by newR x newC
        int resize(int newR, int newC);
        double maxarg(int options, size_t index) const;
        double minarg(int options, size_t index) const;
        size_t argmax(int options, size_t index) const;
        void randn(double mean, double stddev);
        Matrix apply(double (*func)(double)) const;
        Matrix transpose() const;
};

int e_log10(double arg);

Matrix mabs(const Matrix& m);
Matrix hadamard(Matrix m, Matrix d);
Matrix softmax(Matrix m);

double uniform_distribution(double low, double high);

#define LINALG_INCLUDED

#endif