#include "common.hpp"
#include "mlcpp/pca.hpp"

int main() {
    auto d = ex::load_iris();
    mlcpp::PCA pca(4);
    pca.fit(d.X);
    double total = 0;
    for (double v : pca.explained_variance()) total += v;
    for (std::size_t c = 0; c < 4; ++c) {
        std::cout << "PC" << c + 1 << "  variance = " << pca.explained_variance()[c] << "  ratio = " << pca.explained_variance()[c] / total << "  loadings:";
        for (double v : pca.components()[c]) std::cout << ' ' << v;
        std::cout << '\n';
    }
}
