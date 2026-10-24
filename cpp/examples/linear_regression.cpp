// Predict petal width from the other three iris measurements, two ways: normal equation and gradient descent.
#include "common.hpp"
#include "mlcpp/linear_regression.hpp"

int main() {
    auto d = ex::load_iris();
    ex::Matrix X(d.X.rows(), 3);
    ex::Vector y(d.X.rows());
    for (std::size_t i = 0; i < d.X.rows(); ++i) {
        for (std::size_t j = 0; j < 3; ++j) X(i, j) = d.X(i, j);
        y[i] = d.X(i, 3);
    }
    auto [tr, te] = ex::split(X.rows());
    auto Xtr = ex::take(X, tr), Xte = ex::take(X, te);
    auto ytr = ex::take(y, tr), yte = ex::take(y, te);
    ex::standardize(Xtr, Xte);

    auto r2 = [&](const ex::Vector& p) {
        double m = 0, ss_res = 0, ss_tot = 0;
        for (double v : yte) m += v / yte.size();
        for (std::size_t i = 0; i < yte.size(); ++i) {
            ss_res += (yte[i] - p[i]) * (yte[i] - p[i]);
            ss_tot += (yte[i] - m) * (yte[i] - m);
        }
        return 1 - ss_res / ss_tot;
    };

    mlcpp::LinearRegression ne;
    ne.fit_normal_equation(Xtr, ytr);
    std::cout << "Normal equation   w = [";
    for (double w : ne.coef()) std::cout << ' ' << w;
    std::cout << " ]  b = " << ne.intercept() << "  test R^2 = " << r2(ne.predict(Xte)) << '\n';

    mlcpp::LinearRegression gd;
    auto hist = gd.fit_gradient_descent(Xtr, ytr, 0.05, 2000);
    std::cout << "Gradient descent  w = [";
    for (double w : gd.coef()) std::cout << ' ' << w;
    std::cout << " ]  b = " << gd.intercept() << "  test R^2 = " << r2(gd.predict(Xte)) << '\n';
    std::cout << "MSE after 1, 10, 100, 2000 epochs: " << hist[0] << ", " << hist[9] << ", " << hist[99] << ", " << hist.back() << '\n';
}
