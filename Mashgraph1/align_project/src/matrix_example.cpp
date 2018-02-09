#include <iostream>
#include "matrix.h"
#include "io.h"

using std::cout;
using std::endl;

using std::tuple;
using std::get;
using std::tie;
using std::make_tuple;

// Matrix usage example
// Also see: matrix.h, matrix.hpp for comments on how filtering works

class BoxFilterOp
{
public:
    tuple<uint, uint, uint> operator () (const Image &m) const
    {
        uint size = 2 * radius + 1;
        uint r, g, b, sum_r = 0, sum_g = 0, sum_b = 0;
        for (uint i = 0; i < size; ++i) {
            for (uint j = 0; j < size; ++j) {
                // Tie is useful for taking elements from tuple
                tie(r, g, b) = m(i, j);
                sum_r += r;
                sum_g += g;
                sum_b += b;
            }
        }
        auto norm = size * size;
        sum_r /= norm;
        sum_g /= norm;
        sum_b /= norm;
        return make_tuple(sum_r, sum_g, sum_b);
    }
    // Radius of neighbourhoud, which is passed to that operator
    static const int radius = 1;
};


double MSE(Image img, const uint i2_chan, const int h, const int w) {
    double res = 0;
    uint n_rows = img.n_rows;
    uint n_cols = img.n_cols;
    uint a[3];
    uint b[3];
    

    Image chan1 = img.submatrix(h > 0 ? 0 : -h, w > 0 ? 0 : -w, n_rows - abs(h), n_cols - abs(w));
    Image chan2 = img.submatrix(h < 0 ? 0 : h, w < 0 ? 0 : w, n_rows - abs(h), n_cols - abs(w)); //OOOOO SLICES!! WHERE IS MY MIND
    save_image(chan1, "/home/keker/Desktop/res1.bmp");
    save_image(chan2, "/home/keker/Desktop/res2.bmp");
    
    int real_rows = chan1.n_rows;
    int real_cols = chan1.n_cols;
    //cout << real_rows << " " << real_cols << " " << h <<  " " << w << endl;

    for (int i = 0; i < real_rows; i++) {
        for (int j = 0; j < real_cols; j++) {
            std::tie(a[0], a[1], a[2]) = chan1(i,j); //very suitable...
            std::tie(b[0], b[1], b[2]) = chan2(i,j);
            res += (b[i2_chan] - a[0])*(b[i2_chan] - a[0]);
        }
    }
    res = res / real_cols / real_rows;
    //cout << res << " " << h << " " << w << endl;
    return res;
}

int main(int argc, char **argv)
{
    // Image = Matrix<tuple<uint, uint, uint>>
    // tuple is from c++ 11 standard
    Image res = load_image(argv[1]);
    cout << res.n_rows << endl;
    cout << res.n_cols << endl;
    uint row = res.n_rows / 3;
    Image img = res.submatrix(0, 0, row, res.n_cols);
    for (uint i = 0; i < img.n_rows; ++i) {
        for (uint j = 0; j < img.n_cols; ++j) {
            std::get<1>(img(i, j)) = std::get<0>(res(row * 1 + i, j));
            std::get<2>(img(i, j)) = std::get<0>(res(row * 2 + i, j));
        }
    }
    save_image(img, "/home/keker/Desktop/try.bmp");
    
        //MSE(img, 0, i, j);
    
    //Image img2 = img3.unary_map(BoxFilterOp());
    //save_image(img3, argv[2]);
}
