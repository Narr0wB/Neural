
#include "image.h"

std::ostream& operator<<(std::ostream& os, const Image& img)
{
    char ascii[] = ".,-~:;=!*#$@";
    for (int i = 0; i < img.image_data->rows; i++)
    {
        for (int c = 0; c < img.image_data->rows; c++)
        {
            os << ascii[(int) std::round((*(img.image_data))[i][c]*11)] << " ";
        }
        os << std::endl;
    }
    os << "Label: " << img.label;
    return os;
}

Image* ImageList::operator[](size_t idx)
{
    return list[idx];
}

std::ostream& operator<<(std::ostream& os, const ImageList& imagelist)
{
    for (size_t i = 0; i < imagelist.length; i++)
    {
        Image img = *(imagelist.list[i]);
        char ascii[] = ".,-~:;=!*#$@";
        for (int i = 0; i < img.image_data->rows; i++)
        {
            for (int c = 0; c < img.image_data->rows; c++)
            {
                os << ascii[(int) std::round((*(img.image_data))[i][c]*11)] << " ";
            }
            os << std::endl;
        }
        os << "Label: " << img.label << std::endl;
    }
    return os;
}

ImageList csv_to_image(const char* path, int n_of_imgs)
{
    FILE* fp = fopen(path, "r");
    Image** images = new Image*[n_of_imgs];
    ImageList img_l;
    char data[MAX_BUFFER];

    if (fp == NULL)
        throw "Invalid file path";

    fgets(data, MAX_BUFFER, fp);

    int i = 0;
    while (feof(fp) != 1 && i < n_of_imgs)
    {
        images[i] = new Image;

        int j = 0;
        fgets(data, MAX_BUFFER, fp);
        char* token = strtok(data, ",");
        images[i]->image_data = new Matrix(28, 28);
        while (token != NULL)
        {
            if (j==0)
            {
                images[i]->label = std::atoi(token);
            }
            else
            {
                (*(images[i])->image_data)[(j-1) / 28][(j-1) % 28] = std::atoi(token) / 256.0;
            }
            token = strtok(NULL, ",");
            j++;
        }
        i++;
    }
    img_l.list = images;
    img_l.length = i; 
    fclose(fp);
    return img_l;
}