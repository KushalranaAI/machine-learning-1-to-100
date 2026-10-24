// Self-contained tests (no framework): each check prints PASS/FAIL, the exit code is the number of failures.
#include <cmath>
#include <iostream>
#include <random>

#include "mlcpp/decision_tree.hpp"
#include "mlcpp/kmeans.hpp"
#include "mlcpp/knn.hpp"
#include "mlcpp/linear_regression.hpp"
#include "mlcpp/logistic_regression.hpp"
#include "mlcpp/naive_bayes.hpp"
#include "mlcpp/neural_network.hpp"
#include "mlcpp/pca.hpp"

using namespace mlcpp;

static int failures = 0;
static void check(bool ok, const std::string& name) {
    std::cout << (ok ? "PASS  " : "FAIL  ") << name << '\n';
    failures += !ok;
}

// Two Gaussian blobs in 2-D, centres (-2,-2) and (2,2).
static void blobs(Matrix& X, std::vector<int>& y, std::size_t n, unsigned seed, double spread = 0.7) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> N(0, spread);
    X = Matrix(n, 2);
    y.assign(n, 0);
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = i % 2;
        const double c = y[i] ? 2.0 : -2.0;
        X(i, 0) = c + N(rng);
        X(i, 1) = c + N(rng);
    }
}

int main() {
    {  // Cholesky solves a known SPD system exactly
        Matrix A{{4, 2, 0}, {2, 5, 1}, {0, 1, 3}};
        Vector x_true{1, -2, 3}, b = A * x_true, x = cholesky_solve(A, b);
        double err = 0;
        for (int i = 0; i < 3; ++i) err += std::abs(x[i] - x_true[i]);
        check(err < 1e-10, "cholesky_solve recovers x");
    }
    {  // Linear regression recovers the generating weights; GD agrees with the normal equation
        std::mt19937 rng(1);
        std::normal_distribution<double> N(0, 1);
        Matrix X(200, 3);
        Vector y(200);
        for (std::size_t i = 0; i < 200; ++i) {
            for (std::size_t j = 0; j < 3; ++j) X(i, j) = N(rng);
            y[i] = 3 * X(i, 0) - 2 * X(i, 1) + 0.5 * X(i, 2) + 4 + 0.01 * N(rng);
        }
        LinearRegression ne;
        ne.fit_normal_equation(X, y);
        check(std::abs(ne.coef()[0] - 3) < 0.01 && std::abs(ne.coef()[1] + 2) < 0.01 && std::abs(ne.intercept() - 4) < 0.01,
              "normal equation recovers w = (3, -2, 0.5), b = 4");
        LinearRegression gd;
        auto hist = gd.fit_gradient_descent(X, y, 0.1, 500);
        double diff = std::abs(gd.intercept() - ne.intercept());
        for (int j = 0; j < 3; ++j) diff += std::abs(gd.coef()[j] - ne.coef()[j]);
        check(diff < 1e-4, "gradient descent converges to the normal-equation solution");
        bool monotone = true;
        for (std::size_t e = 1; e < hist.size(); ++e) monotone &= hist[e] <= hist[e - 1] + 1e-12;
        check(monotone, "GD loss never increases with a safe learning rate");
        LinearRegression ridge(1000.0);
        ridge.fit_normal_equation(X, y);
        check(std::abs(ridge.coef()[0]) < std::abs(ne.coef()[0]), "ridge shrinks the coefficients");
    }
    Matrix X;
    std::vector<int> y;
    blobs(X, y, 400, 2);
    Matrix Xt;
    std::vector<int> yt;
    blobs(Xt, yt, 200, 3);
    {
        LogisticRegression lr(0.5, 300);
        auto loss = lr.fit(X, y);
        check(loss.back() < loss.front() && lr.accuracy(Xt, yt) > 0.97, "logistic regression separates two blobs");
        check(lr.coef()[0] > 0 && lr.coef()[1] > 0, "logistic weights point from blob 0 to blob 1");
    }
    {
        KNNClassifier knn(5);
        knn.fit(X, y);
        check(knn.accuracy(Xt, yt) > 0.97, "kNN (k=5) on two blobs");
        KNNClassifier one(1);
        one.fit(X, y);
        check(one.accuracy(X, y) == 1.0, "1-NN has 100% training accuracy");
    }
    {
        GaussianNB nb;
        nb.fit(X, y);
        check(nb.accuracy(Xt, yt) > 0.97, "Gaussian naive Bayes on two blobs");
    }
    {
        DecisionTreeClassifier stump(1);
        stump.fit(X, y);
        check(stump.depth() == 1 && stump.accuracy(Xt, yt) > 0.95, "a depth-1 stump already separates the blobs");
        Matrix Xx(4, 2);  // XOR needs depth 2
        std::vector<int> yx{0, 1, 1, 0};
        Xx(1, 1) = Xx(2, 0) = Xx(3, 0) = Xx(3, 1) = 1;
        DecisionTreeClassifier deep(5);
        deep.fit(Xx, yx);
        check(deep.accuracy(Xx, yx) == 1.0 && deep.depth() == 2, "tree learns XOR with depth 2");
    }
    {
        KMeans km(2);
        km.fit(X);
        int agree = 0;
        for (std::size_t i = 0; i < y.size(); ++i) agree += km.labels()[i] == y[i];
        const double purity = std::max(agree, static_cast<int>(y.size()) - agree) / static_cast<double>(y.size());
        check(purity > 0.97, "k-means recovers the two blobs");
        KMeans k1(1);
        k1.fit(X);
        check(km.inertia() < k1.inertia(), "inertia decreases from k=1 to k=2");
    }
    {  // PCA on data stretched along (1,1): the first component must be ±(1,1)/√2
        std::mt19937 rng(4);
        std::normal_distribution<double> N(0, 1);
        Matrix P(500, 2);
        for (std::size_t i = 0; i < 500; ++i) {
            const double a = 3 * N(rng), b = 0.3 * N(rng);
            P(i, 0) = (a + b) / std::sqrt(2);
            P(i, 1) = (a - b) / std::sqrt(2);
        }
        PCA pca(2);
        pca.fit(P);
        const auto& v = pca.components()[0];
        check(std::abs(std::abs(v[0]) - 1 / std::sqrt(2)) < 0.01 && std::abs(dot(v, pca.components()[1])) < 1e-6,
              "PCA finds the (1,1) direction and orthogonal components");
        check(pca.explained_variance()[0] > 50 * pca.explained_variance()[1], "PC1 variance >> PC2 variance (9 vs 0.09)");
    }
    {  // MLP learns XOR, which no linear model can
        Matrix Xx(200, 2);
        std::vector<int> yx(200);
        std::mt19937 rng(5);
        std::uniform_real_distribution<double> U(-1, 1);
        for (std::size_t i = 0; i < 200; ++i) {
            Xx(i, 0) = U(rng);
            Xx(i, 1) = U(rng);
            yx[i] = (Xx(i, 0) > 0) != (Xx(i, 1) > 0);
        }
        MLP net(2, 16, 2, 0);
        auto loss = net.fit(Xx, yx, 300, 0.5, 16);
        check(loss.back() < 0.5 * loss.front() && net.accuracy(Xx, yx) > 0.93, "MLP learns XOR");
    }
    std::cout << (failures ? "\nSOME TESTS FAILED\n" : "\nALL TESTS PASSED\n");
    return failures;
}
