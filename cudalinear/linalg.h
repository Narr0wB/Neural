
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

        Matrix flatten(int options) const;
        void fill(double f);
        int resize(int newR, int newC);
        double maxarg(int options, size_t index) const;
        double minarg(int options, size_t index) const;
        size_t argmax(int options, size_t index) const;
        void randn(double mean, double stddev);
        Matrix apply(double (*func)(double)) const;
        Matrix transpose() const;
        void setdata(double** Ndata, size_t Nrows, size_t Ncols);
        double** getdata();

        // Print the matrix (std::cout)
        friend std::ostream& operator<<(std::ostream &os, const Matrix& m);
        friend Matrix mabs(const Matrix& m);
        friend Matrix hadamard(Matrix m, Matrix d);
        friend Matrix softmax(Matrix m);
};

int e_log10(double arg);

#endif