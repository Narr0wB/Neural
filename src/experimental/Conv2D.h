
#ifndef CONV2D_H
#define CONV2D_H

#include "cudalinear/linalg.h"

#define VALID 0
#define SAME 1
#define FULL 3

template <typename T>
T* nRealloc(T* _ptr, size_t _oSize, size_t _nSize)
{
    if (_oSize == _nSize)
        return _ptr;

    T* _newPtr = new T[_nSize];

    for (size_t i = 0; i < _oSize; i++)
        _newPtr[i] = _ptr[i];
    
    delete[] _ptr;

    return _newPtr;
}

struct MatrixList
{
    private:
        Matrix* matrices;
        size_t allocated;
        size_t _size;

    public:
        MatrixList(size_t length, size_t x, size_t y, double fill = 0) : allocated(length), _size(length)
        {
            matrices = new Matrix[allocated];
            for (size_t i = 0; i < allocated; i++)
                matrices[i] = Matrix(x, y, fill);
        }

        MatrixList(size_t length) : allocated(length), _size(0)
        {
            matrices = new Matrix[allocated];
        }

        MatrixList(Matrix m) : _size(1), allocated(1)
        {      
            matrices = new Matrix[allocated];
            matrices[0] = m;
        }

        MatrixList(const MatrixList& m) : _size(m._size), allocated(m.allocated)
        {
            matrices = new Matrix[allocated];

            for (size_t i = 0; i < allocated; i++)
                matrices[i] = m.matrices[i];
        }

        MatrixList(MatrixList&& m) : _size(m._size), allocated(m.allocated), matrices(m.matrices)
        {
            m.matrices = nullptr;
        }

        MatrixList() {};

        ~MatrixList()
        {
            delete[] matrices;
        }

        inline MatrixList& operator=(const MatrixList& m)
        {
            delete[] matrices;
            
            allocated = m.allocated;
            _size = m._size;

            matrices = new Matrix[allocated];

            memcpy(matrices, m.matrices, allocated * sizeof(Matrix));
        }

        inline void randn(double mean, double stddev)
        {
            for (size_t i = 0; i < _size; i++)
                matrices[i].randn(mean, stddev);
        }

        inline Matrix& operator[](size_t idx)
        {   
            if (idx > allocated)
                throw "INVALID INDEX";
              
            return matrices[idx];
        }

        inline void append(const Matrix& m)
        {   
            if (_size + 1 > allocated)
            {
                matrices = nRealloc<Matrix>(matrices, allocated, allocated + 1);
                allocated = _size + 1;
            }

            matrices[_size] = m;
            _size++;
        }

        inline void append(const MatrixList& m)
        {
            if (_size + m._size > allocated)
            {
                matrices = nRealloc<Matrix>(matrices, allocated, _size + m._size);
                allocated = _size + m._size;
            }

            for (size_t i = 0; i < m._size; i++)
                matrices[i + _size] = m.matrices[i];

            _size += m._size;   
        }

        inline size_t size() const { return allocated; }
};

Matrix pad(Matrix m, int padding);

MatrixList Conv2D(MatrixList input, MatrixList kernels, int padding = VALID, uint32_t stride = 1);

Matrix flatten(MatrixList& m, int options);

MatrixList dflatten(Matrix& m, TensorShape shape, int matrices, int options);

#endif