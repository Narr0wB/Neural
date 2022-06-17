
#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <fstream>
#include <ios>
#include <string.h>

#include "cudalinear/linalg.h"

#define MAX_BUFFER 10000

struct Image 
{
    Matrix* image_data;

    public:
        mutable int label;

        Image() {};

        friend std::ostream& operator<<(std::ostream& os, const Image& img);  
};

struct ImageList
{
    Image** list;
    size_t length;

    Image* operator[](size_t idx);
    friend std::ostream& operator<<(std::ostream& os, const ImageList& imagelist);
};

ImageList csv_to_image(const char* path, int n_of_imgs);

#endif