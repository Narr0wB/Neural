
#include "image.h"

std::ostream& operator<<(std::ostream& os, Image& img)
{
    char ascii[] = ".,-~:;=!*#$@";
    for (size_t i = 0; i < img.image_data.rows(); i++)
    {
        for (size_t c = 0; c < img.image_data.cols(); c++)
        {
            os << ascii[(int) std::round(img.image_data[i][c]*11)] << " ";
        }
        os << std::endl;
    }
    os << "Label: " << img.label << std::endl;
    return os;
}

// Image* ImageList::operator[](size_t idx)
// {
//     return list[idx];
// }

// std::ostream& operator<<(std::ostream& os, const std::vector<Image> imagelist)
// {
//     for (size_t i = 0; i < imagelist.size(); i++)
//     {
//         Image img = imagelist[i];
//         char ascii[] = ".,-~:;=!*#$@";
//         for (int i = 0; i < img.image_data.rows(); i++)
//         {
//             for (int c = 0; c < img.image_data.rows(); c++)
//             {
//                 os << ascii[(int) std::round(img.image_data[i][c]*11)] << " ";
//             }
//             os << std::endl;
//         }
//         os << "Label: " << img.label << std::endl;
//     }
//     return os;
// }

std::vector<Image> csv_to_image(const char* path, int n_of_imgs)
{
    std::ifstream csv_file(path, std::ios::binary);

    std::vector<Image> csv_images;

    if (!csv_file.is_open()) {
        std::cerr << "[ERROR] Invalid file path!";
        exit(EXIT_FAILURE);
    }

    csv_file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    
    
    std::string line;
    for (int n = 0; std::getline(csv_file, line) && n < n_of_imgs; ++n) {
        Image img(28, 28);

        std::stringstream ss(line);
        std::string result;
        

        for (int i = 0; std::getline(ss, result, ','); ++i) {
            if (i == 0) {
                img.label = std::stoi(result);  
                continue;
            }

            img.image_data[((i-1) / 28)][((i-1) % 28)] = std::stod(result) / 255.0;
        }

        

        csv_images.push_back(img);
    }

    csv_file.close();
    return csv_images;

    // FILE* fp = fopen(path, "r");
    // Image** images = new Image*[n_of_imgs];
    // ImageList img_l;
    // char data[MAX_BUFFER];

    // if (fp == NULL)
    //     throw "Invalid file path";

    // fgets(data, MAX_BUFFER, fp);

    // int i = 0;
    // while (feof(fp) != 1 && i < n_of_imgs)
    // {
    //     images[i] = new Image;

    //     int j = 0;
    //     fgets(data, MAX_BUFFER, fp);
    //     char* token = strtok(data, ",");
    //     images[i]->image_data = new Matrix(28, 28);
    //     while (token != NULL)
    //     {
    //         if (j==0)
    //         {
    //             images[i]->label = std::atoi(token);
    //         }
    //         else
    //         {
    //             (*(images[i])->image_data)[(j-1) / 28][(j-1) % 28] = std::atoi(token) / 256.0;
    //         }
    //         token = strtok(NULL, ",");
    //         j++;
    //     }
    //     i++;
    // }
    // img_l.list = images;
    // img_l.length = i; 
    // fclose(fp);
    // return img_l;
}