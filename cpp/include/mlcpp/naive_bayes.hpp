// Gaussian naive Bayes: log P(c) + Σⱼ log N(xⱼ; μ_cj, σ²_cj), computed in log space.
#pragma once

#include <map>

#include "matrix.hpp"

namespace mlcpp {

class GaussianNB {
public:
    void fit(const Matrix& X, const std::vector<int>& y) {
        const std::size_t n = X.rows(), d = X.cols();
        std::map<int, std::vector<std::size_t>> idx;
        for (std::size_t i = 0; i < n; ++i) idx[y[i]].push_back(i);
        double max_var = 0;
        for (std::size_t j = 0; j < d; ++j) {  // var_smoothing like scikit-learn: 1e-9 × largest feature variance
            double m = 0, v = 0;
            for (std::size_t i = 0; i < n; ++i) m += X(i, j) / n;
            for (std::size_t i = 0; i < n; ++i) v += (X(i, j) - m) * (X(i, j) - m) / n;
            max_var = std::max(max_var, v);
        }
        classes_.clear();
        for (auto& [c, rows] : idx) {
            ClassStats s;
            s.log_prior = std::log(static_cast<double>(rows.size()) / n);
            s.mean.assign(d, 0.0);
            s.var.assign(d, 0.0);
            for (auto i : rows)
                for (std::size_t j = 0; j < d; ++j) s.mean[j] += X(i, j) / rows.size();
            for (auto i : rows)
                for (std::size_t j = 0; j < d; ++j) s.var[j] += (X(i, j) - s.mean[j]) * (X(i, j) - s.mean[j]) / rows.size();
            for (auto& v : s.var) v += 1e-9 * max_var;
            classes_[c] = s;
        }
    }

    // Unnormalised log posterior for each class.
    std::map<int, double> log_joint(const Vector& x) const {
        std::map<int, double> out;
        for (auto& [c, s] : classes_) {
            double lp = s.log_prior;
            for (std::size_t j = 0; j < x.size(); ++j)
                lp += -0.5 * std::log(2 * M_PI * s.var[j]) - (x[j] - s.mean[j]) * (x[j] - s.mean[j]) / (2 * s.var[j]);
            out[c] = lp;
        }
        return out;
    }

    int predict_one(const Vector& x) const {
        auto lj = log_joint(x);
        return std::max_element(lj.begin(), lj.end(), [](auto& a, auto& b) { return a.second < b.second; })->first;
    }

    double accuracy(const Matrix& X, const std::vector<int>& y) const {
        int ok = 0;
        for (std::size_t i = 0; i < X.rows(); ++i) ok += predict_one(X.row(i)) == y[i];
        return static_cast<double>(ok) / X.rows();
    }

private:
    struct ClassStats {
        double log_prior;
        Vector mean, var;
    };
    std::map<int, ClassStats> classes_;
};

}  // namespace mlcpp
