#include "common.hpp"
#include "mlcpp/neural_network.hpp"

int main() {
    auto d = ex::load_iris();
    auto [tr, te] = ex::split(d.X.rows());
    auto Xtr = ex::take(d.X, tr), Xte = ex::take(d.X, te);
    ex::standardize(Xtr, Xte);
    mlcpp::MLP net(4, 16, 3, 1);
    auto loss = net.fit(Xtr, ex::take(d.y, tr), 200, 0.1, 16);
    std::cout << "loss: epoch 1 = " << loss[0] << ", 50 = " << loss[49] << ", 200 = " << loss.back() << '\n';
    std::cout << "MLP 4-16-3 test accuracy = " << net.accuracy(Xte, ex::take(d.y, te)) << '\n';
}
