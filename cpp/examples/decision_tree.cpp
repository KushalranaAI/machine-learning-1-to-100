#include "common.hpp"
#include "mlcpp/decision_tree.hpp"

int main() {
    auto d = ex::load_iris();
    auto [tr, te] = ex::split(d.X.rows());
    for (int depth : {1, 2, 3, 5, 10}) {
        mlcpp::DecisionTreeClassifier tree(depth);
        tree.fit(ex::take(d.X, tr), ex::take(d.y, tr));
        std::cout << "max_depth = " << depth << "  depth used = " << tree.depth()
                  << "  train = " << tree.accuracy(ex::take(d.X, tr), ex::take(d.y, tr))
                  << "  test = " << tree.accuracy(ex::take(d.X, te), ex::take(d.y, te)) << '\n';
    }
    mlcpp::DecisionTreeClassifier tree(3);
    tree.fit(d.X, d.y);
    std::cout << "feature importances (sepal_l, sepal_w, petal_l, petal_w):";
    for (double v : tree.feature_importances()) std::cout << ' ' << v;
    std::cout << '\n';
}
