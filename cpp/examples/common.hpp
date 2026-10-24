// Shared helpers for the examples: iris loading, a seeded shuffle split and standardisation.
#pragma once

#include <algorithm>
#include <iostream>
#include <map>
#include <numeric>
#include <random>

#include "mlcpp/matrix.hpp"

namespace ex {

using mlcpp::Matrix;
using mlcpp::Vector;

struct Dataset {
    Matrix X;
    std::vector<int> y;
    std::vector<std::string> classes;
};

inline Dataset load_iris() {
    auto csv = mlcpp::read_csv(std::string(MLCPP_DATA_DIR) + "/iris.csv");
    Dataset d{csv.X, {}, {}};
    std::map<std::string, int> code;
    for (auto& s : csv.labels) {
        if (!code.count(s)) {
            code[s] = static_cast<int>(d.classes.size());
            d.classes.push_back(s);
        }
        d.y.push_back(code[s]);
    }
    return d;
}

inline Matrix take(const Matrix& X, const std::vector<std::size_t>& idx) {
    Matrix out(idx.size(), X.cols());
    for (std::size_t i = 0; i < idx.size(); ++i)
        for (std::size_t j = 0; j < X.cols(); ++j) out(i, j) = X(idx[i], j);
    return out;
}

template <class T>
std::vector<T> take(const std::vector<T>& v, const std::vector<std::size_t>& idx) {
    std::vector<T> out;
    for (auto i : idx) out.push_back(v[i]);
    return out;
}

inline std::pair<std::vector<std::size_t>, std::vector<std::size_t>> split(std::size_t n, double test = 0.3, unsigned seed = 0) {
    std::vector<std::size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(idx.begin(), idx.end(), rng);
    const std::size_t nt = static_cast<std::size_t>(n * test);
    return {{idx.begin() + nt, idx.end()}, {idx.begin(), idx.begin() + nt}};
}

// Fit the scaler on train only, apply to both - the same leakage rule as in the Python lessons.
inline void standardize(Matrix& train, Matrix& test) {
    for (std::size_t j = 0; j < train.cols(); ++j) {
        double m = 0, s = 0;
        for (std::size_t i = 0; i < train.rows(); ++i) m += train(i, j) / train.rows();
        for (std::size_t i = 0; i < train.rows(); ++i) s += (train(i, j) - m) * (train(i, j) - m) / train.rows();
        s = std::sqrt(s) + 1e-12;
        for (std::size_t i = 0; i < train.rows(); ++i) train(i, j) = (train(i, j) - m) / s;
        for (std::size_t i = 0; i < test.rows(); ++i) test(i, j) = (test(i, j) - m) / s;
    }
}

}  // namespace ex
