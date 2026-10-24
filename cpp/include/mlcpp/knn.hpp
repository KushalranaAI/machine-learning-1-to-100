// k-nearest neighbours classifier: store the training data, vote among the k closest points.
#pragma once

#include <algorithm>
#include <map>
#include <numeric>
#include <utility>

#include "matrix.hpp"

namespace mlcpp {

class KNNClassifier {
public:
    explicit KNNClassifier(std::size_t k = 5, bool distance_weighted = false) : k_(k), weighted_(distance_weighted) {}

    void fit(const Matrix& X, const std::vector<int>& y) {
        X_ = X;
        y_ = y;
    }

    int predict_one(const Vector& x) const {
        std::vector<std::pair<double, int>> d(X_.rows());
        for (std::size_t i = 0; i < X_.rows(); ++i) d[i] = {squared_distance(x, X_.row(i)), y_[i]};
        const std::size_t k = std::min(k_, d.size());
        // partial sort: only the k smallest distances need to be ordered - O(n log k)
        std::partial_sort(d.begin(), d.begin() + k, d.end());
        std::map<int, double> votes;
        for (std::size_t i = 0; i < k; ++i) votes[d[i].second] += weighted_ ? 1.0 / (std::sqrt(d[i].first) + 1e-12) : 1.0;
        return std::max_element(votes.begin(), votes.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
    }

    double accuracy(const Matrix& X, const std::vector<int>& y) const {
        int ok = 0;
        for (std::size_t i = 0; i < X.rows(); ++i) ok += predict_one(X.row(i)) == y[i];
        return static_cast<double>(ok) / X.rows();
    }

private:
    std::size_t k_;
    bool weighted_;
    Matrix X_;
    std::vector<int> y_;
};

}  // namespace mlcpp
