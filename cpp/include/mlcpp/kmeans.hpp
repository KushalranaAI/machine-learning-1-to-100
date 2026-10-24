// k-means with k-means++ initialisation (Lloyd's algorithm).
//   assign: c(i) = argmin_k ||xᵢ - μ_k||²     update: μ_k = mean of assigned points
#pragma once

#include <limits>
#include <random>

#include "matrix.hpp"

namespace mlcpp {

class KMeans {
public:
    KMeans(std::size_t k, int max_iter = 300, unsigned seed = 42) : k_(k), max_iter_(max_iter), rng_(seed) {}

    void fit(const Matrix& X) {
        const std::size_t n = X.rows();
        init_plus_plus(X);
        labels_.assign(n, -1);
        for (iterations_ = 1; iterations_ <= max_iter_; ++iterations_) {
            bool changed = false;
            for (std::size_t i = 0; i < n; ++i) {  // assignment step
                const int c = nearest(X.row(i));
                if (c != labels_[i]) {
                    labels_[i] = c;
                    changed = true;
                }
            }
            if (!changed) break;
            Matrix sums(k_, X.cols());  // update step
            std::vector<std::size_t> counts(k_, 0);
            for (std::size_t i = 0; i < n; ++i) {
                counts[labels_[i]]++;
                for (std::size_t j = 0; j < X.cols(); ++j) sums(labels_[i], j) += X(i, j);
            }
            for (std::size_t c = 0; c < k_; ++c)
                if (counts[c])
                    for (std::size_t j = 0; j < X.cols(); ++j) centers_[c][j] = sums(c, j) / counts[c];
        }
        inertia_ = 0;
        for (std::size_t i = 0; i < n; ++i) inertia_ += squared_distance(X.row(i), centers_[labels_[i]]);
    }

    int nearest(const Vector& x) const {
        int best = 0;
        double bd = std::numeric_limits<double>::max();
        for (std::size_t c = 0; c < centers_.size(); ++c) {
            const double d = squared_distance(x, centers_[c]);
            if (d < bd) {
                bd = d;
                best = static_cast<int>(c);
            }
        }
        return best;
    }

    const std::vector<int>& labels() const { return labels_; }
    const std::vector<Vector>& centers() const { return centers_; }
    double inertia() const { return inertia_; }
    int iterations() const { return iterations_; }

private:
    // k-means++: next centre drawn with probability proportional to D(x)², the squared distance to the closest centre.
    void init_plus_plus(const Matrix& X) {
        std::uniform_int_distribution<std::size_t> pick(0, X.rows() - 1);
        centers_ = {X.row(pick(rng_))};
        while (centers_.size() < k_) {
            Vector d2(X.rows());
            for (std::size_t i = 0; i < X.rows(); ++i) d2[i] = squared_distance(X.row(i), centers_[nearest(X.row(i))]);
            std::discrete_distribution<std::size_t> weighted(d2.begin(), d2.end());
            centers_.push_back(X.row(weighted(rng_)));
        }
    }

    std::size_t k_;
    int max_iter_, iterations_ = 0;
    std::mt19937 rng_;
    std::vector<Vector> centers_;
    std::vector<int> labels_;
    double inertia_ = 0;
};

}  // namespace mlcpp
