# mlcpp: the core algorithms in C++17

A header-only re-implementation of the main algorithms from `mlscratch/`. It has no dependencies beyond
the standard library, and every function is a plain loop you can map line-by-line onto the maths in the lessons.

```bash
cd cpp
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure      # 17 checks
./build/kmeans                                   # any example: linear_regression, logistic_regression, knn,
                                                 # naive_bayes, decision_tree, kmeans, pca, neural_network
```
(Windows/MSVC: the same commands work. Executables land in `build/Release/`.)

| Header | Algorithm | The maths, as coded | Lesson |
|---|---|---|---|
| `matrix.hpp` | dense matrix, `cholesky_solve`, CSV reader | A = LLᵀ, then forward and back substitution | 00-math/01 |
| `linear_regression.hpp` | OLS / Ridge (normal equation) + gradient descent | (XᵀX + αI)w = Xᵀy on centred data; w ← w − η·(2/n)Xᵀ(Xw − y) | 07/01, 07/03 |
| `logistic_regression.hpp` | binary logistic regression | p = σ(wᵀx + b); ∇ = (1/n)Xᵀ(p − y) | 08/01 |
| `knn.hpp` | k-nearest neighbours | `partial_sort` keeps only the k smallest distances, O(n log k) | 08/03 |
| `naive_bayes.hpp` | Gaussian naive Bayes | log P(c) + Σ log N(xⱼ; μ, σ²) | 08/04 |
| `decision_tree.hpp` | CART with Gini | sort each feature once, sweep running class counts | 08/06 |
| `kmeans.hpp` | k-means++ and Lloyd's algorithm | D(x)²-weighted seeding; assign → update until no label changes | 10/01 |
| `pca.hpp` | PCA by power iteration + deflation | v ← Σv/‖Σv‖, λ = vᵀΣv, Σ ← Σ − λvvᵀ | 05/03 |
| `neural_network.hpp` | 1-hidden-layer MLP, softmax, SGD | δ₂ = p − y, δ₁ = (δ₂W₂ᵀ) ⊙ (1 − h²) | 11/02 |

## Results on iris (`data/iris.csv`, seeded 70/30 split)

| Example | Output |
|---|---|
| `pca` | explained-variance ratio 0.925, 0.053, 0.017, 0.005 (identical to `mlscratch.PCA` and scikit-learn) |
| `kmeans` | inertia at k = 3 is 78.85; setosa forms its own cluster, versicolor and virginica overlap |
| `linear_regression` | the normal equation and GD give the same weights, test R² = 0.936 (predicting petal width) |
| `logistic_regression` | virginica vs rest: test accuracy 0.978 |
| `knn` | k = 3 or 5: 0.956; k = 51 underfits: 0.867 |
| `decision_tree` | depth 3: test 0.978; all importance falls on the petal features |
| `neural_network` | 4-16-3 MLP: test 0.978 |

## Why C++ at all?
Python with NumPy is how you'll prototype. C++ is what many production inference engines, game
AIs, embedded devices and libraries like XGBoost, LightGBM and scikit-learn's Cython cores are written in.
Seeing the same algorithm without vectorised NumPy calls makes every loop, and so every
cost, explicit. For example, kNN prediction is O(n·d) per query, and the tree's split search is
O(d · n log n) per node.

## Exercises
1. Add `predict_proba` to `KNNClassifier` (the vote shares) and plot a reliability curve.
2. Give `DecisionTreeClassifier` an `entropy` criterion and compare the trees.
3. Turn `MLP` into a regressor: identity output, MSE loss. Only `d2` changes. Why?
4. Time `KNNClassifier::predict_one` against `mlscratch.neighbors` for n = 10⁵. Where does each spend its time?
