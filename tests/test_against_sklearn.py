"""Every from-scratch algorithm is checked against scikit-learn (or a known mathematical property)."""

import numpy as np
import pytest
from sklearn import cluster as skc
from sklearn import datasets as skd
from sklearn import decomposition as skdec
from sklearn import discriminant_analysis as skda
from sklearn import ensemble as ske
from sklearn import feature_extraction as skfe
from sklearn import linear_model as sklm
from sklearn import metrics as skm
from sklearn import mixture as skmix
from sklearn import naive_bayes as sknb
from sklearn import neighbors as skn
from sklearn import preprocessing as skp
from sklearn import svm as sksvm
from sklearn import tree as skt
from sklearn.model_selection import train_test_split as sk_split

import mlscratch as ml


@pytest.fixture(scope="module")
def reg_data():
    X, y = skd.make_regression(300, 5, noise=5, random_state=0)
    return X, y


@pytest.fixture(scope="module")
def clf_data():
    X, y = skd.make_classification(400, 6, n_informative=4, random_state=0)
    return X, y


@pytest.fixture(scope="module")
def iris():
    X, y, _ = ml.datasets.load_iris()
    return X, y


# ------------------------------------------------------------------- preprocessing & metrics

def test_scalers(reg_data):
    X, _ = reg_data
    np.testing.assert_allclose(ml.preprocessing.StandardScaler().fit_transform(X), skp.StandardScaler().fit_transform(X))
    np.testing.assert_allclose(ml.preprocessing.MinMaxScaler().fit_transform(X), skp.MinMaxScaler().fit_transform(X))
    np.testing.assert_allclose(ml.preprocessing.RobustScaler().fit_transform(X), skp.RobustScaler().fit_transform(X))


def test_power_transformer_makes_skewed_data_normal():
    x = np.random.default_rng(0).lognormal(size=(2000, 1))
    z = ml.preprocessing.PowerTransformer().fit_transform(x).ravel()
    skew = lambda v: np.mean((v - v.mean()) ** 3) / v.std() ** 3  # noqa: E731
    assert abs(skew(z)) < 0.2 < skew(x.ravel())
    ref = skp.PowerTransformer().fit(x).lambdas_[0]
    assert abs(ml.preprocessing.PowerTransformer().fit(x).lambdas_[0] - ref) < 0.05


def test_encoders_and_polynomial():
    X = np.array([["red", "S"], ["blue", "M"], ["red", "L"], ["green", "M"]], dtype=object)
    np.testing.assert_array_equal(ml.preprocessing.OneHotEncoder().fit_transform(X), skp.OneHotEncoder().fit_transform(X).toarray())
    Xn = np.random.default_rng(0).normal(size=(10, 3))
    np.testing.assert_allclose(ml.preprocessing.PolynomialFeatures(3).fit_transform(Xn),
                               skp.PolynomialFeatures(3, include_bias=False).fit_transform(Xn))
    np.testing.assert_allclose(ml.preprocessing.PolynomialFeatures(2, interaction_only=True).fit_transform(Xn),
                               skp.PolynomialFeatures(2, interaction_only=True, include_bias=False).fit_transform(Xn))


def test_imputers():
    X = np.array([[1, 2], [np.nan, 3], [7, np.nan], [4, 5.0]])
    from sklearn.impute import KNNImputer, SimpleImputer
    for s in ("mean", "median"):
        np.testing.assert_allclose(ml.preprocessing.SimpleImputer(s).fit_transform(X), SimpleImputer(strategy=s).fit_transform(X))
    np.testing.assert_allclose(ml.preprocessing.KNNImputer(2).fit_transform(X), KNNImputer(n_neighbors=2).fit_transform(X))


def test_classification_metrics():
    rng = np.random.default_rng(0)
    y, s = rng.integers(0, 2, 500), rng.random(500)
    p = (s > 0.5).astype(int)
    assert ml.metrics.accuracy_score(y, p) == pytest.approx(skm.accuracy_score(y, p))
    assert ml.metrics.precision_score(y, p) == pytest.approx(skm.precision_score(y, p))
    assert ml.metrics.recall_score(y, p) == pytest.approx(skm.recall_score(y, p))
    assert ml.metrics.f1_score(y, p) == pytest.approx(skm.f1_score(y, p))
    assert ml.metrics.roc_auc_score(y, s) == pytest.approx(skm.roc_auc_score(y, s))
    assert ml.metrics.log_loss(y, s) == pytest.approx(skm.log_loss(y, s))
    np.testing.assert_array_equal(ml.metrics.confusion_matrix(y, p), skm.confusion_matrix(y, p))
    y3, p3 = rng.integers(0, 3, 300), rng.integers(0, 3, 300)
    assert ml.metrics.f1_score(y3, p3, average="macro") == pytest.approx(skm.f1_score(y3, p3, average="macro"))
    assert ml.metrics.f1_score(y3, p3, average="weighted") == pytest.approx(skm.f1_score(y3, p3, average="weighted"))


def test_regression_metrics(reg_data):
    y = reg_data[1]
    yp = y + np.random.default_rng(0).normal(size=len(y))
    assert ml.metrics.r2_score(y, yp) == pytest.approx(skm.r2_score(y, yp))
    assert ml.metrics.mean_absolute_error(y, yp) == pytest.approx(skm.mean_absolute_error(y, yp))


def test_silhouette():
    X, y = skd.make_blobs(150, centers=3, random_state=0)
    assert ml.metrics.silhouette_score(X, y) == pytest.approx(skm.silhouette_score(X, y))


def test_stratified_split_keeps_ratio():
    y = np.r_[np.zeros(90), np.ones(10)]
    _, _, ytr, yte = ml.model_selection.train_test_split(np.arange(100), y, test_size=0.2, stratify=y, random_state=0)
    assert yte.mean() == pytest.approx(0.1) and ytr.mean() == pytest.approx(0.1)


# --------------------------------------------------------------------------- linear models

def test_linear_regression(reg_data):
    X, y = reg_data
    ref = sklm.LinearRegression().fit(X, y)
    for solver in ("normal", "gd"):
        m = ml.linear_model.LinearRegression(solver=solver, lr=0.1).fit(X, y)
        np.testing.assert_allclose(m.coef_, ref.coef_, rtol=1e-4)
        assert m.intercept_ == pytest.approx(ref.intercept_, abs=1e-3)


def test_ridge_lasso_elasticnet(reg_data):
    X, y = reg_data
    np.testing.assert_allclose(ml.linear_model.Ridge(10).fit(X, y).coef_, sklm.Ridge(10).fit(X, y).coef_, rtol=1e-6)
    np.testing.assert_allclose(ml.linear_model.Lasso(1.0, tol=1e-10).fit(X, y).coef_, sklm.Lasso(1.0, tol=1e-10).fit(X, y).coef_, rtol=1e-4)
    np.testing.assert_allclose(ml.linear_model.ElasticNet(0.5, 0.3, tol=1e-10).fit(X, y).coef_,
                               sklm.ElasticNet(alpha=0.5, l1_ratio=0.3, tol=1e-10).fit(X, y).coef_, rtol=1e-4)


def test_lasso_gives_exact_zeros():
    X, y = skd.make_regression(200, 20, n_informative=3, noise=1, random_state=0)
    coef = ml.linear_model.Lasso(5.0).fit(X, y).coef_
    assert np.sum(coef == 0) >= 15


def test_logistic_regression_matches_sklearn(clf_data):
    X, y = clf_data
    m = ml.linear_model.LogisticRegression(lr=0.5, n_iter=20000, C=1e6).fit(X, y)
    ref = sklm.LogisticRegression(C=1e6, max_iter=10000).fit(X, y)
    np.testing.assert_allclose(m.coef_.ravel(), ref.coef_.ravel(), rtol=0.02, atol=0.02)
    assert m.score(X, y) == pytest.approx(ref.score(X, y), abs=0.01)


def test_softmax_and_perceptron(iris):
    X, y = iris
    Xs = ml.preprocessing.StandardScaler().fit_transform(X)
    assert ml.linear_model.SoftmaxRegression(n_iter=3000).fit(Xs, y).score(Xs, y) > 0.95
    Xb, yb = skd.make_blobs(100, centers=[[-3, -3], [3, 3]], random_state=0)
    assert ml.linear_model.Perceptron(random_state=0).fit(Xb, yb).score(Xb, yb) == 1.0


# ------------------------------------------------------------------------ kNN, NB, SVM, LDA

def test_knn(iris):
    X, y = iris
    for w in ("uniform", "distance"):
        a = ml.neighbors.KNeighborsClassifier(5, weights=w).fit(X, y).predict(X)
        b = skn.KNeighborsClassifier(5, weights=w).fit(X, y).predict(X)
        assert np.mean(a == b) > 0.99
    Xr, yr = skd.make_regression(100, 3, random_state=0)
    np.testing.assert_allclose(ml.neighbors.KNeighborsRegressor(4).fit(Xr, yr).predict(Xr), skn.KNeighborsRegressor(4).fit(Xr, yr).predict(Xr))


def test_naive_bayes(iris):
    X, y = iris
    np.testing.assert_allclose(ml.naive_bayes.GaussianNB().fit(X, y).predict_proba(X), sknb.GaussianNB().fit(X, y).predict_proba(X), atol=1e-6)
    counts = np.random.default_rng(0).poisson(2, size=(200, 15))
    yc = np.random.default_rng(1).integers(0, 3, 200)
    np.testing.assert_allclose(ml.naive_bayes.MultinomialNB().fit(counts, yc).predict_proba(counts),
                               sknb.MultinomialNB().fit(counts, yc).predict_proba(counts), atol=1e-8)
    np.testing.assert_allclose(ml.naive_bayes.BernoulliNB().fit(counts, yc).predict_proba(counts),
                               sknb.BernoulliNB().fit(counts, yc).predict_proba(counts), atol=1e-8)


def test_svm():
    X, y = skd.make_moons(200, noise=0.15, random_state=0)
    ours = ml.svm.SVC(C=1.0, kernel="rbf", gamma=1.0).fit(X, y)
    ref = sksvm.SVC(C=1.0, kernel="rbf", gamma=1.0).fit(X, y)
    assert np.mean(ours.predict(X) == ref.predict(X)) > 0.97
    Xl, yl = skd.make_blobs(200, centers=[[-2, -2], [2, 2]], random_state=0)
    assert ml.svm.LinearSVC().fit(Xl, yl).score(Xl, yl) > 0.98


def test_lda(iris):
    X, y = iris
    assert ml.decomposition.LinearDiscriminantAnalysis().fit(X, y).score(X, y) == pytest.approx(
        skda.LinearDiscriminantAnalysis().fit(X, y).score(X, y), abs=0.01)


# ------------------------------------------------------------------------- trees & ensembles

def test_decision_trees_match_sklearn(iris, clf_data, reg_data):
    X, y = iris
    for crit in ("gini", "entropy"):
        ours = ml.tree.DecisionTreeClassifier(criterion=crit, max_depth=3).fit(X, y)
        ref = skt.DecisionTreeClassifier(criterion=crit, max_depth=3, random_state=0).fit(X, y)
        assert ours.score(X, y) == pytest.approx(ref.score(X, y))
    # Shallow trees on continuous data have no tied splits, so the trees must be identical.
    Xc, yc = clf_data
    for crit in ("gini", "entropy"):
        ours = ml.tree.DecisionTreeClassifier(criterion=crit, max_depth=2).fit(Xc, yc)
        ref = skt.DecisionTreeClassifier(criterion=crit, max_depth=2, random_state=0).fit(Xc, yc)
        np.testing.assert_allclose(ours.feature_importances_, ref.feature_importances_, atol=1e-10)
    # Deeper trees can tie (several splits equally good); both must reach the same optimum.
    ours = ml.tree.DecisionTreeClassifier(max_depth=5).fit(Xc, yc)
    ref = skt.DecisionTreeClassifier(max_depth=5, random_state=0).fit(Xc, yc).tree_
    leaf = ref.children_left == -1
    assert sum(l.n_samples * l.impurity for l in ours.leaves()) == pytest.approx(
        (ref.weighted_n_node_samples[leaf] * ref.impurity[leaf]).sum())
    Xr, yr = reg_data
    np.testing.assert_allclose(ml.tree.DecisionTreeRegressor(max_depth=4).fit(Xr, yr).predict(Xr),
                               skt.DecisionTreeRegressor(max_depth=4, random_state=0).fit(Xr, yr).predict(Xr))


def test_tree_sample_weight_and_min_leaf():
    X, y = skd.make_classification(300, 4, random_state=1)
    w = np.random.default_rng(0).random(300)
    ours = ml.tree.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5).fit(X, y, sample_weight=w)
    ref = skt.DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=0).fit(X, y, sample_weight=w)
    assert np.mean(ours.predict(X) == ref.predict(X)) > 0.98


def test_ensembles(clf_data):
    X, y = clf_data
    Xtr, Xte, ytr, yte = sk_split(X, y, random_state=0)
    rf = ml.ensemble.RandomForestClassifier(50, random_state=0).fit(Xtr, ytr)
    assert rf.score(Xte, yte) > ske.RandomForestClassifier(50, random_state=0).fit(Xtr, ytr).score(Xte, yte) - 0.05
    assert 0.7 < rf.oob_score_ <= 1
    ada = ml.ensemble.AdaBoostClassifier(50).fit(Xtr, ytr)
    ada_ref = ske.AdaBoostClassifier(skt.DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=0).fit(Xtr, ytr)
    np.testing.assert_allclose(ada.alphas_, ada_ref.estimator_weights_[: len(ada.alphas_)], rtol=1e-6)
    assert np.mean(ada.predict(Xte) == ada_ref.predict(Xte)) == 1.0
    # Early rounds must match exactly (no ties yet); later rounds may pick different tied splits.
    gb = ml.ensemble.GradientBoostingClassifier(5, max_depth=3).fit(Xtr, ytr)
    ref = ske.GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=0).fit(Xtr, ytr)
    np.testing.assert_allclose(gb.predict_proba(Xte)[:, 1], ref.predict_proba(Xte)[:, 1], atol=1e-10)
    gb100 = ml.ensemble.GradientBoostingClassifier(100, max_depth=3).fit(Xtr, ytr)
    ref100 = ske.GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=0).fit(Xtr, ytr)
    assert abs(gb100.score(Xte, yte) - ref100.score(Xte, yte)) <= 0.03


def test_gradient_boosting_regressor_matches_sklearn(reg_data):
    X, y = reg_data
    ours = ml.ensemble.GradientBoostingRegressor(50, 0.1, 3).fit(X, y)
    ref = ske.GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.predict(X), ref.predict(X), rtol=1e-6)


def test_multiclass_gradient_boosting(iris):
    X, y = iris
    ours = ml.ensemble.GradientBoostingClassifier(3, max_depth=2).fit(X, y)
    ref = ske.GradientBoostingClassifier(n_estimators=3, max_depth=2, random_state=0).fit(X, y)
    np.testing.assert_allclose(ours.predict_proba(X), ref.predict_proba(X), atol=1e-6)


def test_xgboost_like(clf_data, reg_data):
    X, y = clf_data
    m = ml.ensemble.XGBoostLike("binary:logistic", n_estimators=50, max_depth=3, learning_rate=0.3).fit(X, y)
    assert np.mean(m.predict(X) == y) > 0.95
    Xr, yr = reg_data
    r = ml.ensemble.XGBoostLike(n_estimators=100, max_depth=3).fit(Xr, yr)
    assert ml.metrics.r2_score(yr, r.predict(Xr)) > 0.95


# ------------------------------------------------------------------------------ unsupervised

def test_kmeans_matches_sklearn_inertia():
    X, _ = skd.make_blobs(300, centers=4, random_state=0)
    ours = ml.cluster.KMeans(4, random_state=0).fit(X)
    ref = skc.KMeans(4, n_init=10, random_state=0).fit(X)
    assert ours.inertia_ == pytest.approx(ref.inertia_, rel=1e-6)


def test_agglomerative_matches_sklearn():
    X, _ = skd.make_blobs(60, centers=3, random_state=2)
    for link in ("ward", "single", "complete", "average"):
        ours = ml.cluster.AgglomerativeClustering(3, link).fit_predict(X)
        ref = skc.AgglomerativeClustering(3, linkage=link).fit_predict(X)
        assert skm.adjusted_rand_score(ours, ref) == pytest.approx(1.0)


def test_dbscan_matches_sklearn():
    X, _ = skd.make_moons(200, noise=0.05, random_state=0)
    ours = ml.cluster.DBSCAN(0.2, 5).fit_predict(X)
    ref = skc.DBSCAN(eps=0.2, min_samples=5).fit_predict(X)
    assert skm.adjusted_rand_score(ours, ref) == pytest.approx(1.0)


def test_gmm():
    X, _ = skd.make_blobs(400, centers=3, random_state=0)
    ours = ml.cluster.GaussianMixture(3, random_state=0).fit(X)
    ref = skmix.GaussianMixture(3, random_state=0).fit(X)
    assert ours.score(X) == pytest.approx(ref.score(X), rel=1e-3)
    assert np.all(np.diff(ours.lower_bound_history_) > -1e-9)        # EM never decreases the likelihood


def test_pca_and_kernel_pca(iris):
    X, _ = iris
    ours, ref = ml.decomposition.PCA(2).fit(X), skdec.PCA(2).fit(X)
    np.testing.assert_allclose(ours.explained_variance_, ref.explained_variance_)
    np.testing.assert_allclose(np.abs(ours.transform(X)), np.abs(ref.transform(X)), atol=1e-8)
    assert ml.decomposition.PCA(0.95).fit(X).n_components_ == skdec.PCA(0.95).fit(X).n_components_
    a = ml.decomposition.KernelPCA(2, gamma=0.5).fit_transform(X)
    b = skdec.KernelPCA(2, kernel="rbf", gamma=0.5).fit_transform(X)
    np.testing.assert_allclose(np.abs(a), np.abs(b), atol=1e-6)


def test_tsne_separates_clusters():
    X, y = skd.make_blobs(90, centers=3, n_features=10, random_state=0)
    Y = ml.decomposition.TSNE(perplexity=15, n_iter=400, random_state=0).fit_transform(X)
    assert ml.metrics.silhouette_score(Y, y) > 0.6


def test_isolation_forest_finds_outliers():
    rng = np.random.default_rng(0)
    angles = rng.uniform(0, 2 * np.pi, 15)
    outliers = np.c_[np.cos(angles), np.sin(angles)] * rng.uniform(5, 8, (15, 1))   # scattered far from the bulk
    X = np.vstack([rng.normal(0, 1, (400, 2)), outliers])
    ours = ml.anomaly.IsolationForest(random_state=0, contamination=15 / 415).fit(X).predict(X)
    ref = ske.IsolationForest(random_state=0, contamination=15 / 415).fit(X).predict(X)
    assert np.mean(ours[-15:] == -1) >= 0.9
    assert np.mean(ours == ref) > 0.97


def test_feature_selection():
    X, y = skd.make_classification(300, 10, n_informative=3, n_redundant=0, shuffle=False, random_state=0)
    from sklearn.feature_selection import chi2 as skchi2
    from sklearn.feature_selection import f_classif as skf
    np.testing.assert_allclose(ml.feature_selection.f_classif(X, y), skf(X, y)[0], rtol=1e-8)
    Xp = np.abs(X)
    np.testing.assert_allclose(ml.feature_selection.chi2(Xp, y), skchi2(Xp, y)[0], rtol=1e-8)
    sel = ml.feature_selection.RFE(ml.linear_model.LogisticRegression(n_iter=500), 3).fit(X, y)
    assert set(np.flatnonzero(sel.support_)) == {0, 1, 2}


def test_text_vectorizers():
    docs = ["free prize call now", "call me later", "win a free prize now now", "see you later"]
    np.testing.assert_allclose(ml.text.CountVectorizer().fit_transform(docs), skfe.text.CountVectorizer().fit_transform(docs).toarray())
    np.testing.assert_allclose(ml.text.TfidfVectorizer().fit_transform(docs), skfe.text.TfidfVectorizer().fit_transform(docs).toarray())
    np.testing.assert_allclose(ml.text.CountVectorizer(ngram_range=(1, 2)).fit_transform(docs),
                               skfe.text.CountVectorizer(ngram_range=(1, 2)).fit_transform(docs).toarray())


def test_apriori():
    tx = [{"bread", "milk"}, {"bread", "diaper", "beer", "eggs"}, {"milk", "diaper", "beer", "cola"},
          {"bread", "milk", "diaper", "beer"}, {"bread", "milk", "diaper", "cola"}]
    freq = ml.association.apriori(tx, 0.6)
    assert freq[frozenset({"diaper", "beer"})] == pytest.approx(0.6)
    rules = ml.association.association_rules(freq, 0.7)
    assert any(r["antecedent"] == {"beer"} and r["consequent"] == {"diaper"} and r["confidence"] == 1.0 for r in rules)


# ---------------------------------------------------------------------------- neural networks

def test_backprop_gradients_are_correct():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(8, 5))
    Y = np.eye(3)[rng.integers(0, 3, 8)]
    nn = ml.neural_network
    model = nn.Sequential([nn.Dense(5, 7, rng=0), nn.Tanh(), nn.Dense(7, 6, rng=1), nn.Sigmoid(), nn.Dense(6, 3, rng=2)])
    assert nn.gradient_check(model, X, Y) < 1e-6


def test_conv_and_rnn_gradients():
    nn = ml.neural_network
    rng = np.random.default_rng(0)
    for layer, x in [(nn.Conv2D(2, 3, k=3, rng=0), rng.normal(size=(2, 2, 5, 5))),
                     (nn.SimpleRNN(3, 4, rng=0), rng.normal(size=(2, 6, 3)))]:
        out = layer.forward(x)
        g = rng.normal(size=out.shape)
        layer.backward(g)
        name = "W" if "W" in layer.params else "Wh"
        p = layer.params[name]
        idx = (0,) * p.ndim
        old = p[idx]
        p[idx] = old + 1e-6
        lp = (layer.forward(x) * g).sum()
        p[idx] = old - 1e-6
        lm = (layer.forward(x) * g).sum()
        p[idx] = old
        assert (lp - lm) / 2e-6 == pytest.approx(layer.grads[name][idx], rel=1e-5)


def test_mlp_learns_nonlinear_boundary():
    X, y = skd.make_moons(300, noise=0.1, random_state=0)
    m = ml.neural_network.MLPClassifier((32, 32), lr=0.01, epochs=200, random_state=0).fit(X, y)
    assert m.score(X, y) > 0.97
    Xr = np.linspace(-3, 3, 200)[:, None]
    yr = np.sin(Xr).ravel()
    r = ml.neural_network.MLPRegressor((32,), activation="tanh", lr=0.01, epochs=300, random_state=0).fit(Xr, yr)
    assert r.score(Xr, yr) > 0.97


def test_optimizers_minimise_rosenbrock_like_bowl():
    grad = lambda t: np.array([2 * t[0], 20 * t[1]])  # noqa: E731  f = x² + 10y²
    for opt in (ml.optim.SGD(0.04), ml.optim.SGD(0.02, momentum=0.9), ml.optim.Adam(0.3), ml.optim.RMSProp(0.05)):
        path = ml.optim.minimize(grad, [3.0, 2.0], opt, 300)
        assert np.linalg.norm(path[-1]) < 0.05


# ------------------------------------------------------------------------------------------ RL

def test_value_and_policy_iteration_agree():
    env = ml.rl.GridWorld(slip=0.2)
    V1, p1, _ = ml.rl.value_iteration(env, 0.95)
    V2, p2, _ = ml.rl.policy_iteration(env, 0.95)
    np.testing.assert_allclose(V1, V2, atol=1e-6)


def test_q_learning_finds_goal():
    env = ml.rl.GridWorld(random_state=0)
    _, policy, returns = ml.rl.q_learning(env, episodes=1500, random_state=0)
    _, optimal, _ = ml.rl.value_iteration(env)
    s, steps = env.reset(), 0
    done = False
    while not done and steps < 50:
        s, _, done = env.step(policy[s])
        steps += 1
    assert env.grid[env.s[0]][env.s[1]] == "G"
    assert np.mean(returns[-100:]) > 0.5


def test_bandit_ucb_prefers_best_arm():
    _, counts, _ = ml.rl.MultiArmedBandit([0.1, 0.5, 0.9]).run("ucb", 2000)
    assert np.argmax(counts) == 2


def test_grid_search_picks_reasonable_k(iris):
    X, y = iris
    gs = ml.model_selection.GridSearchCV(ml.neighbors.KNeighborsClassifier(), {"n_neighbors": [1, 5, 15]}, cv=5).fit(X, y)
    assert gs.best_score_ > 0.9 and gs.best_params_["n_neighbors"] in (5, 15)
