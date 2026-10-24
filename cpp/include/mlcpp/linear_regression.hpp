// Linear regression: ŷ = Xw + b
//   fit_normal_equation: solve (XᵀX + αI) w = Xᵀy with Cholesky (α > 0 gives Ridge; intercept not penalised)
//   fit_gradient_descent: w ← w - η (2/n) Xᵀ(Xw - y)
#pragma once

#include "matrix.hpp"

namespace mlcpp {

class LinearRegression {
public:
    explicit LinearRegression(double ridge_alpha = 0.0) : alpha_(ridge_alpha) {}

    void fit_normal_equation(const Matrix& X, const Vector& y) {
        const std::size_t n = X.rows(), d = X.cols();
        // Centre X and y so the intercept is not penalised: b = ȳ - x̄ᵀw.
        Vector xm(d, 0.0);
        double ym = 0;
        for (std::size_t i = 0; i < n; ++i) {
            for (std::size_t j = 0; j < d; ++j) xm[j] += X(i, j) / n;
            ym += y[i] / n;
        }
        Matrix A(d, d);
        Vector rhs(d, 0.0);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < d; ++j) {
                const double xij = X(i, j) - xm[j];
                rhs[j] += xij * (y[i] - ym);
                for (std::size_t k = 0; k < d; ++k) A(j, k) += xij * (X(i, k) - xm[k]);
            }
        for (std::size_t j = 0; j < d; ++j) A(j, j) += alpha_;
        w_ = cholesky_solve(A, rhs);
        b_ = ym - dot(xm, w_);
    }

    // Returns the MSE after every epoch so the caller can plot the loss curve.
    Vector fit_gradient_descent(const Matrix& X, const Vector& y, double lr = 0.01, int epochs = 1000) {
        const std::size_t n = X.rows(), d = X.cols();
        w_.assign(d, 0.0);
        b_ = 0.0;
        Vector history;
        for (int e = 0; e < epochs; ++e) {
            Vector gw(d, 0.0);
            double gb = 0, mse = 0;
            for (std::size_t i = 0; i < n; ++i) {
                const double r = predict_one(X.row(i)) - y[i];   // residual
                for (std::size_t j = 0; j < d; ++j) gw[j] += 2.0 * r * X(i, j) / n;
                gb += 2.0 * r / n;
                mse += r * r / n;
            }
            for (std::size_t j = 0; j < d; ++j) w_[j] -= lr * (gw[j] + 2.0 * alpha_ * w_[j] / n);
            b_ -= lr * gb;
            history.push_back(mse);
        }
        return history;
    }

    double predict_one(const Vector& x) const { return dot(w_, x) + b_; }

    Vector predict(const Matrix& X) const {
        Vector out(X.rows());
        for (std::size_t i = 0; i < X.rows(); ++i) out[i] = predict_one(X.row(i));
        return out;
    }

    const Vector& coef() const { return w_; }
    double intercept() const { return b_; }

private:
    double alpha_;
    Vector w_;
    double b_ = 0.0;
};

}  // namespace mlcpp
