// Binary logistic regression: is this iris a virginica?
#include "common.hpp"
#include "mlcpp/logistic_regression.hpp"

int main() {
    auto d = ex::load_iris();
    std::vector<int> y;
    for (int c : d.y) y.push_back(c == 2);
    auto [tr, te] = ex::split(d.X.rows());
    auto Xtr = ex::take(d.X, tr), Xte = ex::take(d.X, te);
    ex::standardize(Xtr, Xte);
    mlcpp::LogisticRegression clf(0.5, 1000);
    auto loss = clf.fit(Xtr, ex::take(y, tr));
    std::cout << "cross-entropy: epoch 1 = " << loss[0] << ", epoch 1000 = " << loss.back() << '\n';
    std::cout << "test accuracy (virginica vs rest) = " << clf.accuracy(Xte, ex::take(y, te)) << '\n';
}
