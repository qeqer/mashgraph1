#include "align.h"
#include <string>
#include <cmath>
#include <vector>

using std::string;
using std::cout;
using std::endl;
using std::get;

template<typename ValueT>
ValueT min (ValueT &a, ValueT &b) {
    if (a < b) return a;
    else return b;
}
template<typename ValueT>
ValueT abs (ValueT &a) {
    if (a < 0) return -a;
    else return a;
}




void MSE(Image img, const int h, const int w, double &mse1, double &mse2) { //computing MSE //change: for 3 channels at the time, faster and withoud tie
    uint n_rows = img.n_rows;
    uint n_cols = img.n_cols; ////
    //experiment
    const uint border_x = 15; 
    const uint border_y = 15;

    Image imgcor = img.submatrix(border_x, border_y, n_rows - 2 * border_x, n_cols - 2 * border_y); //bordering white spaces (????????????)

    n_rows = imgcor.n_rows - abs(h); //new size of cropped matrixes
    n_cols = imgcor.n_cols - abs(w);

    Image chan1 = imgcor.submatrix(h > 0 ? 0 : -h, w > 0 ? 0 : -w, n_rows, n_cols);
    Image chan2 = imgcor.submatrix(h < 0 ? 0 : h, w < 0 ? 0 : w, n_rows, n_cols); //OOOOO SLICES!! WHERE IS MY MIND 

    int sum1 = 0, sum2 = 0;
    for (uint i = 0; i < n_rows; i++) {
        for (uint j = 0; j < n_cols; j++) {
            int f_pix = get<0>(chan2(i, j)) - get<1>(chan1(i, j)); //suitable
            int s_pix = get<0>(chan2(i, j)) - get<2>(chan1(i, j));
            sum1 += f_pix * f_pix;
            sum2 += s_pix * s_pix;
        }
    }
    mse1 = sum1 / (n_rows * n_cols);
    mse2 = sum2 / (n_rows * n_cols);
    return;
}

Image aligning2 (Image img, int h1, int h2, int w1, int w2) { //faster, better and without "second image smaller" bug
    Image chan1 = img.submatrix(h1 > 0 ? 0 : -h1, w1 > 0 ? 0 : -w1, img.n_rows - abs(h1), img.n_cols - abs(w1));
    Image chan12 = img.submatrix(h1 < 0 ? 0 : h1, w1 < 0 ? 0 : w1, img.n_rows - abs(h1), img.n_cols - abs(w1));

    Image chan2 = img.submatrix(h2 > 0 ? 0 : -h2, w2 > 0 ? 0 : -w2, img.n_rows - abs(h2), img.n_cols - abs(w2));
    Image chan22 = img.submatrix(h2 < 0 ? 0 : h2, w2 < 0 ? 0 : w2, img.n_rows - abs(h2), img.n_cols - abs(w2)); //this is the zero fixed chanel

    int h = h1 - h2;
    int w = w1 - w2;

    Image res0 = chan12.submatrix(h > 0 ? 0 : -h, w > 0 ? 0 : -w, chan1.n_rows - abs(h), chan1.n_cols - abs(w));
    Image res1 = chan1.submatrix(h > 0 ? 0 : -h, w > 0 ? 0 : -w, chan1.n_rows - abs(h), chan1.n_cols - abs(w));
    Image res2 = chan2.submatrix(h < 0 ? 0 : h, w < 0 ? 0 : w, chan2.n_rows - abs(h), chan2.n_cols - abs(w));

    int res_rows = min(res1.n_rows, res2.n_rows), res_cols = min(res1.n_cols, res2.n_cols);
    Image res(res_rows, res_cols);
    for (int i = 0; i < res_rows; i++) {
        for (int j = 0; j < res_cols; j++) {
            res(i, j) = std::make_tuple(get<0>(res0(i, j)), get<1>(res1(i, j)), get<2>(res2(i,j)));
        }
    }
    return res;
}

Image aligning(Image img, int h1, int h2, int w1, int w2) { //aligning for 3 chanels, first is not moving


    Image chan1 = img.submatrix(h1 > 0 ? 0 : -h1, w1 > 0 ? 0 : -w1, img.n_rows - abs(h1), img.n_cols - abs(w1));
    Image chan12 = img.submatrix(h1 < 0 ? 0 : h1, w1 < 0 ? 0 : w1, img.n_rows - abs(h1), img.n_cols - abs(w1));

    int real_rows = img.n_rows - abs(h1), real_cols = img.n_cols - abs(w1);
    Image res1(real_rows, real_cols);

    for (int i = 0; i < real_rows; i++) {
        for (int j = 0; j < real_cols; j++) {
            res1(i,j) = std::make_tuple(std::get<0>(chan12(i,j)), std::get<1>(chan1(i,j)), 0);
        }
    }

    real_rows = img.n_rows - abs(h2), real_cols = img.n_cols - abs(w2);
    Image chan3 = img.submatrix(h2 > 0 ? 0 : -h2, w2 > 0 ? 0 : -w2, real_rows, real_cols);
    Image chan32 = img.submatrix(h2 < 0 ? 0 : h2, w2 < 0 ? 0 : w2, real_rows, real_cols);

    Image res2(real_rows, real_cols);

    for (int i = 0; i < real_rows; i++) {
        for (int j = 0; j < real_cols; j++) {
            res2(i,j) = std::make_tuple(std::get<0>(chan32(i,j)), 0, std::get<2>(chan3(i,j)));
        }
    }
    //save_image(res1, "/home/keker/Desktop/res1.bmp");
    //save_image(res2, "/home/keker/Desktop/res2.bmp");

    int h = h1 - h2;
    int w = w1 - w2;

    real_rows = real_rows - abs(h);
    real_cols = real_cols - abs(w);

    cout << res1.n_rows << " " << res2.n_rows << endl;
    res1 = res1.submatrix(h > 0 ? 0 : -h, w > 0 ? 0 : -w, real_rows, real_cols);
    res2 = res2.submatrix(h < 0 ? 0 : h, w < 0 ? 0 : w, real_rows, real_cols);


    Image res(real_rows, real_cols);

    for (int i = 0; i < real_rows; i++) {
        for (int j = 0; j < real_cols; j++) {
            res(i,j) = std::make_tuple(std::get<0>(res1(i,j)), std::get<1>(res1(i,j)), std::get<2>(res2(i,j)));
        }
    }
    //save_image(res, "/home/keker/Desktop/res.bmp");    
    return res;
}

Image align(Image srcImage, bool isPostprocessing, std::string postprocessingType, double fraction, bool isMirror, 
            bool isInterp, bool isSubpixel, double subScale)
{
    const int rad = 15; //radius of searching
    uint row = srcImage.n_rows;
    uint col = srcImage.n_cols;
    row = row / 3;
    Image res = srcImage.submatrix(0, 0, row, col).deep_copy();

    for (uint i = 0; i < row; ++i) {
        for (uint j = 0; j < col; ++j) {
            std::get<1>(res(i, j)) = std::get<0>(srcImage(row * 1 + i, j));
            std::get<2>(res(i, j)) = std::get<0>(srcImage(row * 2 + i, j));
        }
    }

    double MSE1, MSE2, temp1, temp2;
    MSE(res, 0, 0, MSE1, MSE2);
    int h1 = 0, w1 = 0, h2 = 0, w2 = 0;

    for (int h = -rad; h <=rad; ++h) { //metrics for 3 channels
        for (int w = -rad; w <= rad; ++w) {
            MSE(res, h, w, temp1, temp2);
            if (temp1 < MSE1) {
                MSE1 = temp1, h1 = h, w1 = w;
            }
            if (temp2 < MSE2) {
                MSE2 = temp2, h2 = h, w2 = w;
            }
        }
    }
    
    res = aligning2(res, h1, h2, w1, w2);

    if (isPostprocessing) {
        if (postprocessingType == "--gray-world") {
            res = gray_world(res);
        }
        if (postprocessingType == "--unsharp") {
            res = unsharp(res);
        }
        if (postprocessingType == "--autocontrast") {
            res = autocontrast(res, fraction);
        }
    }

    return res;
}

Image sobel_x(Image src_image) {
    Matrix<double> kernel = {{-1, 0, 1},
                             {-2, 0, 2},
                             {-1, 0, 1}};
    return custom(src_image, kernel);
}

Image sobel_y(Image src_image) {
    Matrix<double> kernel = {{ 1,  2,  1},
                             { 0,  0,  0},
                             {-1, -2, -1}};
    return custom(src_image, kernel);
}


template<typename T>
T check255(T check) {
    T res = check;
    if (check < 0) 
        res = 0;
    if (check > 255)
        res = 255;
    return res;

} 

Image convolution(Image img, double kernel[3][3]) {
    int edge = 3 / 2;
    double temp_sum_b, temp_sum_g, temp_sum_r;
    uint b, g, r;
    Image res(img.n_rows, img.n_cols);
    for (uint i = edge; i < img.n_rows - edge; i++) {
        for (uint j = edge; j < img.n_cols - edge; j++) {
            temp_sum_g = 0, temp_sum_r = 0, temp_sum_b = 0;
            for (int k = -edge; k <= edge; k++) {
                for (int l = -edge; l <= edge; l++) {
                    std::tie(b, g, r) = img(i + k, j + l);
                    // cout << b << " " << kernel[k + edge][l + edge] * b << endl;
                    temp_sum_b += kernel[k + edge][l + edge] * b;
                    temp_sum_g += kernel[k + edge][l + edge] * g;
                    temp_sum_r += kernel[k + edge][l + edge] * r;
                }
            }
            //cout << temp_sum_b << " " << temp_sum_g << " " << temp_sum_r << endl;
            res(i, j) = std::make_tuple(check255(temp_sum_b) , check255(temp_sum_g), check255(temp_sum_r));
        }
    }
    return res;
}

Image unsharp(Image src_image) {
    double unsharp_kernel[3][3] = { 
        {-1.0/6, -2.0/3, -1.0/6},
        {-2.0/3, 13.0/3, -2.0/3},
        {-1.0/6, -2.0/3, -1.0/6}
    };

    return convolution(src_image, unsharp_kernel);
}

Image gray_world(Image src_image) {
    double ave_r = 0, ave_g = 0, ave_b = 0, ave = 0; //average of channels
    int b, g, r;
    int n_rows = src_image.n_rows, n_cols = src_image.n_cols;
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            std::tie(b, g, r) = src_image(i, j);
            ave_b += b;
            ave_g += g;
            ave_r += r;
        }
    }
    
    ave_b = ave_b / (n_rows * n_cols);
    ave_g = ave_g / (n_rows * n_cols);
    ave_r = ave_r / (n_rows * n_cols);
    ave = (ave_r + ave_g + ave_b) / 3;

    ave_b = ave_b < 0.01 ? 0 : ave / ave_b;
    ave_g = ave_g < 0.01 ? 0 : ave / ave_g;
    ave_r = ave_r < 0.01 ? 0 : ave / ave_r;
    int tempb, tempg, tempr;
    for (int i = 0; i < n_rows; i++) {
        for (int j = 0; j < n_cols; j++) {
            std::tie(b, g, r) = src_image(i, j);
            tempb = b * ave_b;
            tempg = g * ave_g;
            tempr = r * ave_r;
            src_image(i, j) = std::make_tuple(tempb > 255 ? 255 : tempb, tempg > 255 ? 255 : tempg, tempr > 255 ? 255 : tempr);
        }
    }

    return src_image;
}

Image resize(Image src_image, double scale) {
    return src_image;
}

Image custom(Image src_image, Matrix<double> kernel) { //Matrix.. nonono
    // Function custom is useful for making concrete linear filtrations
    // like gaussian or sobel. So, we assume that you implement custom
    // and then implement other filtrations using this function.
    // sobel_x and sobel_y are given as an example.
    return src_image;
}

Image autocontrast(Image src_image, double fraction) {
    std::vector<uint> hist(256, 0);
    uint n_rows = src_image.n_rows, n_cols = src_image.n_cols;
    uint b, g, r;
    for (uint i = 0; i < n_rows; i++) {
        for (uint j = 0; j < n_cols; j++) {
            std::tie(b, g, r) = src_image(i, j);
            uint y = 0.2125 * r + 0.7154 * g + 0.0721 * b;
            y = check255(y);
            hist[y]++;
        }
    }
    
    uint pixel_ex = n_rows * n_cols * fraction;
    uint pix_sum = 0;
    uint left = 0, right = 255;
    while (pix_sum < pixel_ex) {
        pix_sum += hist[left];
        left++;
    }
    pix_sum = 0;
    while (pix_sum < pixel_ex) {
        pix_sum += hist[right];
        right--;
    }
    double pix_change = 255.0 / (right - left);

    for (uint i = 0; i < n_rows; i++) {
        for (uint j = 0; j < n_cols; j++) {
            std::tie(b, g, r) = src_image(i, j);
            b = check255(int((b - left) * pix_change));
            g = check255(int((g - left) * pix_change));
            r = check255(int((r - left) * pix_change));
            //cout << pix_change << " " << left  <<  " " << b  <<  " " << g <<  " " << r << endl;
            src_image(i, j) = std::make_tuple(b, g, r);
        } 
    }

    return src_image;
}

Image gaussian(Image src_image, double sigma, int radius)  {
    return src_image;
}
Image gaussian_separable(Image src_image, double sigma, int radius) {
    return src_image;
}

Image median(Image src_image, int radius) {
    return src_image;
}

Image median_linear(Image src_image, int radius) {
    return src_image;
}

Image median_const(Image src_image, int radius) {
    return src_image;
}

Image canny(Image src_image, int threshold1, int threshold2) {
    return src_image;
}
