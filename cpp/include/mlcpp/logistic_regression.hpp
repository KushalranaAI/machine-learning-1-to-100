// Binary logistic regression trained by batch gradient descent on the cross-entropy.
//   p = σ(wᵀx + b),  ∇w = (1/n) Xᵀ(p - y) + (λ/n) w
#pragma once

#include "matrix.hpp"

namespace mlcpp {

inline double sigmoid(double z) {  // stable for large |z|
    if (z >= 0) return 1.0 / (1.0 + std::exp(-z));
    const double e = std::exp(z);
    return e / (1.0 + e);
}

class LogisticRegression {
public:
    LogisticRegression(double lr = 0.1, int epochs = 1000, double l2 = 0.0) : lr_(lr), epochs_(epochs), l2_(l2) {}

    Vector fit(const Matrix& X, const std::vector<int>& y) {
        const std::size_t n = X.rows(), d = X.cols();
        w_.assign(d, 0.0);
        b_ = 0.0;
        Vector losses;
        for (int e = 0; e < epochs_; ++e) {
            Vector gw(d, 0.0);
            double gb = 0, loss = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double p = predict_proba_one(X.row(i));
                const double err = p - y[i];                       // the famous "prediction minus target"
                for (std::size_t j = 0; j < d; ++j) gw[j] += err * X(i, j) / n;
                gb += err / n;
                loss -= (y[i] ? std::log(p + 1e-12) : std::log(1 - p + 1e-12)) / n;
            }
            for (std::size_t j = 0; j < d; ++j) w_[j] -= lr_ * (gw[j] + l2_ * w_[j] / n);
            b_ -= lr_ * gb;
            losses.push_back(loss);
        }
        return losses;
    }

    double predict_proba_one(const Vector& x) const { return sigmoid(dot(w_, x) + b_); }
    int predict_one(const Vector& x, double threshold = 0.5) const { return predict_proba_one(x) >= threshold; }

    double accuracy(const Matrix& X, const std::vector<int>& y) const {
        int ok = 0;
        for (std::size_t i = 0; i < X.rows(); ++i) ok += predict_one(X.row(i)) == y[i];
        return static_cast<double>(ok) / X.rows();
    }

    const Vector& coef() const { return w_; }
    double intercept() const { return b_; }

private:
    double lr_;
    int epochs_;
    double l2_;
    Vector w_;
    double b_ = 0.0;
};

}  // namespace mlcpp
