
#ifndef IMAGE_H
#define IMAGE_H

#include <iostream>
#include <fstream>
#include <ios>
#include <string.h>
#include <sstream>

#include "linalg/linalg.h"

#define MAX_BUFFER 10000

class Image 
{
    public:
        int label;

        Matrix image_data;
        size_t m_width;
        size_t m_height;

        Image(size_t width, size_t height) : image_data(width, height), m_width(width), m_height(height)  {};

        Image(const Image& other) : label(other.label), image_data(other.image_data), m_width(other.m_width), m_height(other.m_height)  {
            label = other.label;
        }

        friend std::ostream& operator<<(std::ostream& os, Image& img);  
};

// struct ImageList
// {
//     Image** list;
//     size_t length;

//     Image* operator[](size_t idx);
//     friend std::ostream& operator<<(std::ostream& os, const ImageList& imagelist);
// };

std::vector<Image> csv_to_image(const char* path, int n_of_imgs);

#endif