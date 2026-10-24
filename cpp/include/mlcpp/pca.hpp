// PCA by power iteration with deflation on the covariance matrix.
//   repeat v ← Σv / ||Σv||  → top eigenvector;  then Σ ← Σ - λ v vᵀ and repeat for the next component.
#pragma once

#include <random>

#include "matrix.hpp"

namespace mlcpp {

class PCA {
public:
    explicit PCA(std::size_t n_components) : k_(n_components) {}

    void fit(const Matrix& X, int iters = 500) {
        const std::size_t n = X.rows(), d = X.cols();
        mean_.assign(d, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < d; ++j) mean_[j] += X(i, j) / n;
        Matrix C(d, d);  // covariance Σ = XcᵀXc / (n-1)
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t a = 0; a < d; ++a)
                for (std::size_t b = 0; b < d; ++b) C(a, b) += (X(i, a) - mean_[a]) * (X(i, b) - mean_[b]) / (n - 1);
        std::mt19937 rng(0);
        std::normal_distribution<double> N(0, 1);
        components_.clear();
        variances_.clear();
        for (std::size_t c = 0; c < k_; ++c) {
            Vector v(d);
            for (auto& x : v) x = N(rng);
            double lambda = 0;
            for (int it = 0; it < iters; ++it) {
                Vector w = C * v;
                const double norm = std::sqrt(dot(w, w));
                if (norm == 0) break;
                for (std::size_t j = 0; j < d; ++j) v[j] = w[j] / norm;
                lambda = dot(v, C * v);  // Rayleigh quotient = eigenvalue estimate
            }
            // sign convention: largest-magnitude loading positive
            std::size_t big = 0;
            for (std::size_t j = 1; j < d; ++j)
                if (std::abs(v[j]) > std::abs(v[big])) big = j;
            if (v[big] < 0)
                for (auto& x : v) x = -x;
            components_.push_back(v);
            variances_.push_back(lambda);
            for (std::size_t a = 0; a < d; ++a)  // deflation: remove this direction
                for (std::size_t b = 0; b < d; ++b) C(a, b) -= lambda * v[a] * v[b];
        }
    }

    Vector transform_one(const Vector& x) const {
        Vector z(components_.size());
        for (std::size_t c = 0; c < components_.size(); ++c) {
            double s = 0;
            for (std::size_t j = 0; j < x.size(); ++j) s += (x[j] - mean_[j]) * components_[c][j];
            z[c] = s;
        }
        return z;
    }

    const std::vector<Vector>& components() const { return components_; }
    const Vector& explained_variance() const { return variances_; }

private:
    std::size_t k_;
    Vector mean_, variances_;
    std::vector<Vector> components_;
};

}  // namespace mlcpp
