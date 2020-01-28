#ifndef __VLAD_H__
#define __VLAD_H__

#include <bits/stdc++.h>

using namespace std;

struct Keypoint {
    float x, y, s;
    float a11,a12,a21,a22;
    float response;
    int type;
    unsigned char desc[128];
};


struct VLAD {
public:
    VLAD(const string &centers_path, const size_t &rows, const size_t &cols);
    vector<double> encode(const vector<Keypoint> &keys, bool use_norm);

private:
    vector<vector<double>> centers;
    size_t rows, cols;

};

struct ProductQuantization {
public:
    ProductQuantization(const string &centers_path, const size_t &rows, const size_t &cols, const size_t &dim);
    vector<unsigned char> predict(const vector<double> &vlad);

private:
    vector<vector<double>> centers;
    size_t rows, cols;
    size_t dim;
};
#endif // __VLAD_H__
