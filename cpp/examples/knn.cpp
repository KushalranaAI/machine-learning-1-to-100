#include "common.hpp"
#include "mlcpp/knn.hpp"

int main() {
    auto d = ex::load_iris();
    auto [tr, te] = ex::split(d.X.rows());
    auto Xtr = ex::take(d.X, tr), Xte = ex::take(d.X, te);
    ex::standardize(Xtr, Xte);
    for (std::size_t k : {1, 3, 5, 15, 51}) {
        mlcpp::KNNClassifier knn(k);
        knn.fit(Xtr, ex::take(d.y, tr));
        std::cout << "k = " << k << "  test accuracy = " << knn.accuracy(Xte, ex::take(d.y, te)) << '\n';
    }
}
