#include "common.hpp"
#include "mlcpp/naive_bayes.hpp"

int main() {
    auto d = ex::load_iris();
    auto [tr, te] = ex::split(d.X.rows());
    mlcpp::GaussianNB nb;
    nb.fit(ex::take(d.X, tr), ex::take(d.y, tr));
    std::cout << "Gaussian naive Bayes test accuracy = " << nb.accuracy(ex::take(d.X, te), ex::take(d.y, te)) << '\n';
    auto x = d.X.row(te[0]);
    std::cout << "log P(c) + log P(x|c) for one test flower:";
    for (auto& [c, lp] : nb.log_joint(x)) std::cout << "  " << d.classes[c] << '=' << lp;
    std::cout << "\n(true class: " << d.classes[d.y[te[0]]] << ")\n";
}
