#include "common.hpp"
#include "mlcpp/kmeans.hpp"

int main() {
    auto d = ex::load_iris();
    std::cout << "elbow curve (k, inertia):\n";
    for (std::size_t k = 1; k <= 6; ++k) {
        mlcpp::KMeans km(k);
        km.fit(d.X);
        std::cout << "  " << k << "  " << km.inertia() << "  (" << km.iterations() << " iterations)\n";
    }
    mlcpp::KMeans km(3);
    km.fit(d.X);
    // contingency table: how the unsupervised clusters line up with the real species
    int table[3][3] = {};
    for (std::size_t i = 0; i < d.y.size(); ++i) table[km.labels()[i]][d.y[i]]++;
    std::cout << "cluster vs species:\n";
    for (int c = 0; c < 3; ++c) std::cout << "  cluster " << c << ": " << table[c][0] << ' ' << table[c][1] << ' ' << table[c][2] << '\n';
}
