// A one-hidden-layer neural network (tanh hidden units, softmax output) trained with mini-batch SGD.
// Forward:  h = tanh(x W1 + b1),  p = softmax(h W2 + b2)
// Backward: δ2 = p - y;  ∂W2 = hᵀδ2;  δ1 = (δ2 W2ᵀ) ⊙ (1 - h²);  ∂W1 = xᵀδ1
#pragma once

#include <algorithm>
#include <numeric>
#include <random>

#include "matrix.hpp"

namespace mlcpp {

class MLP {
public:
    MLP(std::size_t n_in, std::size_t n_hidden, std::size_t n_out, unsigned seed = 0)
        : W1_(n_in, n_hidden), W2_(n_hidden, n_out), b1_(n_hidden, 0.0), b2_(n_out, 0.0), rng_(seed) {
        std::normal_distribution<double> N(0, 1);  // Xavier initialisation
        const double s1 = std::sqrt(2.0 / (n_in + n_hidden)), s2 = std::sqrt(2.0 / (n_hidden + n_out));
        for (std::size_t i = 0; i < n_in; ++i)
            for (std::size_t j = 0; j < n_hidden; ++j) W1_(i, j) = N(rng_) * s1;
        for (std::size_t i = 0; i < n_hidden; ++i)
            for (std::size_t j = 0; j < n_out; ++j) W2_(i, j) = N(rng_) * s2;
    }

    void forward(const Vector& x, Vector& h, Vector& p) const {
        h.assign(b1_.begin(), b1_.end());
        for (std::size_t i = 0; i < x.size(); ++i)
            for (std::size_t j = 0; j < h.size(); ++j) h[j] += x[i] * W1_(i, j);
        for (auto& v : h) v = std::tanh(v);
        p.assign(b2_.begin(), b2_.end());
        for (std::size_t i = 0; i < h.size(); ++i)
            for (std::size_t k = 0; k < p.size(); ++k) p[k] += h[i] * W2_(i, k);
        const double mx = *std::max_element(p.begin(), p.end());  // stable softmax
        double s = 0;
        for (auto& v : p) s += (v = std::exp(v - mx));
        for (auto& v : p) v /= s;
    }

    // Returns the average cross-entropy per epoch.
    Vector fit(const Matrix& X, const std::vector<int>& y, int epochs = 50, double lr = 0.1, std::size_t batch = 16) {
        const std::size_t n = X.rows(), H = b1_.size(), K = b2_.size(), D = X.cols();
        std::vector<std::size_t> order(n);
        std::iota(order.begin(), order.end(), 0);
        Vector history;
        for (int e = 0; e < epochs; ++e) {
            std::shuffle(order.begin(), order.end(), rng_);
            double loss = 0;
            for (std::size_t start = 0; start < n; start += batch) {
                const std::size_t end = std::min(n, start + batch), m = end - start;
                Matrix gW1(D, H), gW2(H, K);
                Vector gb1(H, 0.0), gb2(K, 0.0), h, p;
                for (std::size_t t = start; t < end; ++t) {
                    const Vector x = X.row(order[t]);
                    forward(x, h, p);
                    loss -= std::log(p[y[order[t]]] + 1e-12) / n;
                    Vector d2 = p;
                    d2[y[order[t]]] -= 1.0;  // softmax + cross-entropy gradient
                    Vector d1(H, 0.0);
                    for (std::size_t j = 0; j < H; ++j) {
                        for (std::size_t k = 0; k < K; ++k) {
                            gW2(j, k) += h[j] * d2[k];
                            d1[j] += d2[k] * W2_(j, k);
                        }
                        d1[j] *= 1 - h[j] * h[j];  // tanh'
                    }
                    for (std::size_t k = 0; k < K; ++k) gb2[k] += d2[k];
                    for (std::size_t i = 0; i < D; ++i)
                        for (std::size_t j = 0; j < H; ++j) gW1(i, j) += x[i] * d1[j];
                    for (std::size_t j = 0; j < H; ++j) gb1[j] += d1[j];
                }
                for (std::size_t i = 0; i < D; ++i)
                    for (std::size_t j = 0; j < H; ++j) W1_(i, j) -= lr * gW1(i, j) / m;
                for (std::size_t j = 0; j < H; ++j) {
                    b1_[j] -= lr * gb1[j] / m;
                    for (std::size_t k = 0; k < K; ++k) W2_(j, k) -= lr * gW2(j, k) / m;
                }
                for (std::size_t k = 0; k < K; ++k) b2_[k] -= lr * gb2[k] / m;
            }
            history.push_back(loss);
        }
        return history;
    }

    int predict_one(const Vector& x) const {
        Vector h, p;
        forward(x, h, p);
        return static_cast<int>(std::max_element(p.begin(), p.end()) - p.begin());
    }

    double accuracy(const Matrix& X, const std::vector<int>& y) const {
        int ok = 0;
        for (std::size_t i = 0; i < X.rows(); ++i) ok += predict_one(X.row(i)) == y[i];
        return static_cast<double>(ok) / X.rows();
    }

private:
    Matrix W1_, W2_;
    Vector b1_, b2_;
    mutable std::mt19937 rng_;
};

}  // namespace mlcpp
