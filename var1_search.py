"""
var1 - full grid search over (degree, alpha), ridge.

Covers EVERY degree the assignment permits (1 to 10), so each one is either
selected or rejected with a number rather than left untested.  Every degree
gets the same fine alpha grid, so no degree can be unfairly rated because of
a coarse alpha spacing.  Scored by 5-fold CV on both required metrics.

Note on the high degrees: with only 1000 training rows, degree 9 needs 5005
terms and degree 10 needs 8007 -- far more unknowns than equations.  They are
still fitted here (ridge can solve an underdetermined system), and their
scores are reported so the rejection is evidence-based.
"""
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV

HERE = Path(__file__).resolve().parent     # relative to this script, not to cwd
TRAIN = HERE / "IMT2024063_train_var1.csv"
COLS = [f"x{i}" for i in range(1, 7)]

df = pd.read_csv(TRAIN)
X, y = df[COLS].to_numpy(), df["y"].to_numpy()

pipe = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                 ("scale", StandardScaler()),
                 ("ridge", Ridge())])

grid = {"poly__degree": list(range(1, 11)),   # the assignment allows up to 10
        "ridge__alpha": list(np.logspace(-2, 3, 21))}

gs = GridSearchCV(pipe, grid,
                  scoring={"mse": "neg_mean_squared_error", "r2": "r2"},
                  refit="mse",
                  cv=KFold(5, shuffle=True, random_state=0),
                  n_jobs=-1)
gs.fit(X, y)

res = pd.DataFrame(gs.cv_results_)
res["mse"] = -res["mean_test_mse"]
res["r2"] = res["mean_test_r2"]

print(f"searched {len(res)} combinations "
      f"({len(grid['poly__degree'])} degrees x {len(grid['ridge__alpha'])} alphas)\n")
print("best alpha found for each degree:")
print(f"{'deg':>4} {'best alpha':>11} {'CV MSE':>9} {'CV R2':>8}")
for d, g in res.groupby("param_poly__degree"):
    b = g.loc[g["mse"].idxmin()]
    print(f"{d:>4} {b['param_ridge__alpha']:>11.3f} {b['mse']:>9.4f} {b['r2']:>8.4f}")

b = gs.best_params_
print(f"\nWINNER: degree {b['poly__degree']}, alpha {b['ridge__alpha']:.3f}")
print(f"        CV MSE {-gs.best_score_:.4f}")

res[["param_poly__degree", "param_ridge__alpha", "mse", "r2"]] \
   .to_csv(HERE / "var1_grid_results.csv", index=False)
print("        full grid saved to var1_grid_results.csv")
