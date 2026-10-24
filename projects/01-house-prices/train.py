"""Step 3 - compare models on a validation set, tune the best ones, then report the test score once.

Split: 60% train / 20% validation / 20% test. Ensembles are too slow to cross-validate quickly in pure NumPy,
so model selection uses the validation set; the linear models also get 5-fold CV to show the fold-to-fold spread.
Target: log(price). Errors are multiplicative (10% off is equally bad at $100k and $400k), and the log makes them additive.
"""

import time

import numpy as np

from features import HousingFeatures, load
from mlscratch.ensemble import GradientBoostingRegressor, RandomForestRegressor
from mlscratch.linear_model import LinearRegression, Ridge
from mlscratch.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from mlscratch.model_selection import KFold, cross_val_score, train_test_split
from mlscratch.neighbors import KNeighborsRegressor
from mlscratch.tree import DecisionTreeRegressor
from mlscratch.viz import plt, savefig

df = load()
y_all = np.log(df.median_house_value.to_numpy())
idx_trval, idx_test = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
idx_train, idx_val = train_test_split(idx_trval, test_size=0.25, random_state=42)
feats = HousingFeatures().fit(df.iloc[idx_train])          # fitted on train rows only
X_train, X_val, X_test = (feats.transform(df.iloc[i]) for i in (idx_train, idx_val, idx_test))
y_train, y_val, y_test = y_all[idx_train], y_all[idx_val], y_all[idx_test]
print(f"train {len(idx_train):,} · validation {len(idx_val):,} · test {len(idx_test):,} · {X_train.shape[1]} features\n")


def dollars_rmse(y_log, p_log):
    return root_mean_squared_error(np.exp(y_log), np.exp(p_log))


print("5-fold CV on the training set (R² on log price):")
for name, model in [("LinearRegression", LinearRegression()), ("Ridge(α=1)", Ridge(1.0))]:
    s = cross_val_score(model, X_train, y_train, cv=KFold(5, random_state=0), scoring=r2_score)
    print(f"  {name:<18} {s.mean():.4f} ± {s.std():.4f}")

candidates = {
    "Baseline (mean)": None,
    "LinearRegression": LinearRegression(),
    "Ridge α=10": Ridge(10.0),
    "kNN k=10 (distance)": KNeighborsRegressor(10, weights="distance"),
    "DecisionTree depth=10": DecisionTreeRegressor(max_depth=10, min_samples_leaf=5),
    "RandomForest 40×depth14": RandomForestRegressor(n_estimators=40, max_depth=14, max_features=0.5, min_samples_leaf=3, random_state=0),
    "GradientBoosting 400×d5": GradientBoostingRegressor(n_estimators=400, learning_rate=0.1, max_depth=5, subsample=0.8, min_samples_leaf=5, random_state=0),
}
print(f"\n{'model':<26}{'val RMSE (log)':>15}{'val RMSE ($)':>14}{'val R²':>9}{'fit s':>8}")
results, fitted = {}, {}
for name, model in candidates.items():
    t = time.time()
    if model is None:
        p = np.full(len(y_val), y_train.mean())
    else:
        model.fit(X_train, y_train)
        p = model.predict(X_val)
        fitted[name] = model
    results[name] = root_mean_squared_error(y_val, p)
    print(f"{name:<26}{results[name]:>15.4f}{dollars_rmse(y_val, p):>14,.0f}{r2_score(y_val, p):>9.3f}{time.time() - t:>8.1f}")

# Early stopping by hand: staged_predict gives the validation error after every boosting round.
gbm = fitted["GradientBoosting 400×d5"]
val_curve = [root_mean_squared_error(y_val, p) for p in gbm.staged_predict(X_val)]
best_rounds = int(np.argmin(val_curve)) + 1
if best_rounds == len(val_curve):
    print(f"\nGradient boosting: validation RMSE still falling at round {best_rounds} ({val_curve[99]:.4f} at 100 → "
          f"{val_curve[-1]:.4f}); more rounds would help a little. Train RMSE {np.sqrt(gbm.train_loss_[-1]):.4f} shows the gap.")
else:
    print(f"\nGradient boosting: best validation RMSE {min(val_curve):.4f} after {best_rounds} rounds; after that it overfits")

best_name = min(results, key=results.get)
print(f"\nBest on validation: {best_name}. Refit on train+validation and score the untouched test set ONCE:")
feats_final = HousingFeatures().fit(df.iloc[idx_trval])
X_trval, X_test = feats_final.transform(df.iloc[idx_trval]), feats_final.transform(df.iloc[idx_test])
final = fitted[best_name].clone()
if isinstance(final, GradientBoostingRegressor):
    final.set_params(n_estimators=best_rounds)
final.fit(X_trval, y_all[idx_trval])
p_test = final.predict(X_test)
print(f"  test RMSE (log) = {root_mean_squared_error(y_test, p_test):.4f}")
print(f"  test RMSE ($)   = {dollars_rmse(y_test, p_test):,.0f}")
print(f"  test MAE ($)    = {mean_absolute_error(np.exp(y_test), np.exp(p_test)):,.0f}")
print(f"  test R² (log)   = {r2_score(y_test, p_test):.3f}")
print(f"  median |error|  = {np.median(np.abs(np.exp(p_test) / np.exp(y_test) - 1)):.1%} of the true price")

# Multicollinearity: total_rooms, total_bedrooms, population and households are all "district size" (r > 0.9).
# OLS splits that shared signal into large opposite-signed weights that change from fold to fold (see the CV spread
# above); Ridge's penalty picks a stable compromise.
lin, ridge = fitted["LinearRegression"], fitted["Ridge α=10"]
print("\nStandardized coefficients - OLS vs Ridge on the collinear 'size' features:")
for f in ["total_rooms", "total_bedrooms", "population", "households", "median_income"]:
    i = feats.names_.index(f)
    print(f"  {f:<16} OLS {lin.coef_[i]:+.3f}   Ridge {ridge.coef_[i]:+.3f}")

P = plt()
fig, axes = P.subplots(1, 3, figsize=(17, 4.6))
names = list(results)
axes[0].barh(names, [results[n] for n in names], color="#4c78a8")
axes[0].set(title="Validation RMSE (log price) - lower is better")
axes[1].plot(range(1, len(val_curve) + 1), val_curve, label="validation")
axes[1].plot(range(1, len(val_curve) + 1), np.sqrt(gbm.train_loss_), label="train")
axes[1].axvline(best_rounds, color="red", ls="--", label=f"best = {best_rounds}")
axes[1].set(title="Gradient boosting: error vs rounds", xlabel="rounds", ylabel="RMSE (log)", ylim=(0, 0.6))
axes[1].legend()
axes[2].scatter(np.exp(y_test) / 1000, np.exp(p_test) / 1000, s=3, alpha=0.3)
axes[2].plot([0, 500], [0, 500], "r--")
axes[2].set(title=f"Test set: predicted vs actual ({best_name.split()[0]})", xlabel="actual k$", ylabel="predicted k$")
savefig(fig, "04_model_comparison")

# Impurity importance of a boosted model = average over its trees (each tree's importances sum to 1).
trees = getattr(final, "estimators_", [])
imp = np.mean([t.feature_importances_ for t in trees], axis=0) if trees else getattr(final, "feature_importances_", None)
if imp is not None:
    order = np.argsort(imp)
    print("\nTop 5 features by importance:", ", ".join(f"{feats_final.names_[i]} ({imp[i]:.2f})" for i in order[::-1][:5]))
    fig, ax = P.subplots(figsize=(7, 6))
    ax.barh(np.array(feats_final.names_)[order], imp[order])
    ax.set(title="Impurity-based feature importance (final model)")
    savefig(fig, "05_feature_importance")
