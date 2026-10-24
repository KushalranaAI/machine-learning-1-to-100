// CART classification tree with Gini impurity.
// Best split per feature in O(n log n): sort once, sweep left→right updating running class counts.
#pragma once

#include <algorithm>
#include <memory>
#include <numeric>

#include "matrix.hpp"

namespace mlcpp {

class DecisionTreeClassifier {
public:
    explicit DecisionTreeClassifier(int max_depth = 5, std::size_t min_samples_leaf = 1)
        : max_depth_(max_depth), min_leaf_(min_samples_leaf) {}

    void fit(const Matrix& X, const std::vector<int>& y) {
        n_classes_ = *std::max_element(y.begin(), y.end()) + 1;
        importances_.assign(X.cols(), 0.0);
        std::vector<std::size_t> idx(X.rows());
        std::iota(idx.begin(), idx.end(), 0);
        root_ = grow(X, y, idx, 0);
        const double total = std::accumulate(importances_.begin(), importances_.end(), 0.0);
        if (total > 0)
            for (auto& v : importances_) v /= total;
    }

    int predict_one(const Vector& x) const {
        const Node* node = root_.get();
        while (node->feature >= 0) node = x[node->feature] <= node->threshold ? node->left.get() : node->right.get();
        return node->label;
    }

    double accuracy(const Matrix& X, const std::vector<int>& y) const {
        int ok = 0;
        for (std::size_t i = 0; i < X.rows(); ++i) ok += predict_one(X.row(i)) == y[i];
        return static_cast<double>(ok) / X.rows();
    }

    const Vector& feature_importances() const { return importances_; }
    int depth() const { return depth_of(root_.get()); }

private:
    struct Node {
        int feature = -1;  // -1 = leaf
        double threshold = 0;
        int label = 0;
        std::unique_ptr<Node> left, right;
    };

    static double gini(const std::vector<double>& counts, double total) {
        if (total <= 0) return 0;
        double s = 1.0;
        for (double c : counts) s -= (c / total) * (c / total);
        return s;
    }

    std::unique_ptr<Node> grow(const Matrix& X, const std::vector<int>& y, std::vector<std::size_t>& idx, int depth) {
        auto node = std::make_unique<Node>();
        std::vector<double> counts(n_classes_, 0.0);
        for (auto i : idx) counts[y[i]]++;
        node->label = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
        const double n = idx.size(), parent = gini(counts, n);
        if (depth >= max_depth_ || parent == 0.0 || idx.size() < 2 * min_leaf_) return node;

        // Like scikit-learn, accept a zero-gain split: on XOR the root split gains nothing, but its children do.
        double best_gain = -1.0, best_t = 0;
        int best_f = -1;
        for (std::size_t f = 0; f < X.cols(); ++f) {
            std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) { return X(a, f) < X(b, f); });
            std::vector<double> left(n_classes_, 0.0);
            for (std::size_t k = 0; k + 1 < idx.size(); ++k) {
                left[y[idx[k]]]++;
                const std::size_t nl = k + 1, nr = idx.size() - nl;
                if (nl < min_leaf_ || nr < min_leaf_ || X(idx[k], f) == X(idx[k + 1], f)) continue;
                std::vector<double> right(n_classes_);
                for (int c = 0; c < n_classes_; ++c) right[c] = counts[c] - left[c];
                const double gain = parent - (nl * gini(left, nl) + nr * gini(right, nr)) / n;
                if (gain > best_gain) {
                    best_gain = gain;
                    best_f = static_cast<int>(f);
                    best_t = (X(idx[k], f) + X(idx[k + 1], f)) / 2;
                }
            }
        }
        if (best_f < 0) return node;
        importances_[best_f] += best_gain * n;
        std::vector<std::size_t> L, R;
        for (auto i : idx) (X(i, best_f) <= best_t ? L : R).push_back(i);
        node->feature = best_f;
        node->threshold = best_t;
        node->left = grow(X, y, L, depth + 1);
        node->right = grow(X, y, R, depth + 1);
        return node;
    }

    static int depth_of(const Node* n) { return n->feature < 0 ? 0 : 1 + std::max(depth_of(n->left.get()), depth_of(n->right.get())); }

    int max_depth_, n_classes_ = 0;
    std::size_t min_leaf_;
    std::unique_ptr<Node> root_;
    Vector importances_;
};

}  // namespace mlcpp
