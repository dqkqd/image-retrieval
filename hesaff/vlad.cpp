
#include "vlad.h"
#include <sys/times.h>
#include <time.h>
#include <sys/time.h>

using namespace std;

VLAD::VLAD(const string &centers_path, const size_t &rows, const size_t &cols) {
    this->rows = rows;
    this->cols = cols;

    ifstream centers_file;
    centers_file.open(centers_path);

    if (!centers_file) {
        cerr << "Cannot open files: " << centers_path << endl;
        exit(1);
    }

    centers.assign(rows, vector<double>(cols));
    double x;
    size_t i = 0;
    while (centers_file >> x) {
        centers[i/cols][i%cols] = x;
        i++;
    }
    centers_file.close();
}

vector<double> VLAD::encode(const vector<Keypoint> &keys, bool use_norm) {
    vector<vector<double>> v(rows, vector<double>(cols));
    size_t N = keys.size();
    
    for (size_t i=0; i<N; ++i) {
        // predict labels
        size_t label = 0;
        double best;
        for (size_t j=0; j<rows; ++j) {
            double score = 0.0f;
            for (size_t k=0; k<cols; ++k) {
                double s = double(keys[i].desc[k]) - centers[j][k];
                score += s*s;
            }

            if (j == 0) {
                best = score;
            }
            else if (best > score){
                best = score;
                label = j;
            }
        }
        // vlad
        for (size_t j=0; j<cols; j++) {
            v[label][j] += keys[i].desc[j] - centers[label][j];
        }
    }
    vector<double> norms(rows, 1.0f);
    if (use_norm) {
        for (size_t i=0; i<rows; ++i) {
            norms[i] = 0.0f;
            for (size_t j=0; j<cols; ++j) {
                norms[i] += v[i][j] * v[i][j];
            }
            norms[i] = max(sqrt(norms[i]), 1e-12);
        }
    }
    vector<double> flatten(rows*cols, 0.0f);
    for (size_t i=0; i<rows; ++i) {
        for (size_t j=0; j<cols; ++j) {
            flatten[i*cols+ j] = v[i][j] / norms[i];
        }
    }
    return flatten;
}

ProductQuantization::ProductQuantization (const string &centers_path, const size_t &rows, const size_t &cols, const size_t &dim) {
    this->rows = rows;
    this->cols = cols;
    this->dim = dim;

    assert(!(cols % dim));

    ifstream centers_file;
    centers_file.open(centers_path);

    if (!centers_file) {
        cerr << "Cannot open files: " << centers_path << endl;
        exit(1);
    }

    centers.assign(rows, vector<double>(cols));
    double x;
    size_t i = 0;
    while (centers_file >> x) {
        centers[i/cols][i%cols] = x;
        i++;
    }
    centers_file.close();
}

vector<unsigned char> ProductQuantization::predict(const vector<double> &vlad) {
    vector<unsigned char> labels(cols/dim, 0);
    // predict on each batch
    for (size_t i=0; i<labels.size(); ++i) {
        size_t start = i * dim;
        size_t end = (i+1) * dim;
        double best;
        for (size_t j=0; j<rows; ++j) {
            double score = 0;
            for (size_t k=start; k<end; ++k) {
                double s = vlad[k] - centers[j][k];
                score += s*s;
            }
            if (j == 0) {
                best = score;
            }
            else if (best > score) {
                best = score;
                labels[i] = j;
            }
        }
    }
    return labels;
}


