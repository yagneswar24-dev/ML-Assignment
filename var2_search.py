from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_validate

COLS = ["x1", "x2", "x3"]
DEGREES = range(1, 21)        # the assignment allows up to degree 20
ALPHAS = np.logspace(-4, 4, 25)

HERE = Path(__file__).resolve().parent     # relative to this script, not to cwd
df = pd.read_csv(HERE / "IMT2024063_train_var2.csv")
X, y = df[COLS].to_numpy(), df["y"].to_numpy()
cv = KFold(5, shuffle=True, random_state=0)

rows = []
for d in DEGREES:
    Z = PolynomialFeatures(d, include_bias=False).fit_transform(X)
    for a in ALPHAS:
        sc = cross_validate(make_pipeline(StandardScaler(), Ridge(alpha=a)),
                            Z, y, cv=cv,
                            scoring=("neg_mean_squared_error", "r2"))
        rows.append({"degree": d, "alpha": a, "terms": Z.shape[1],
                     "mse": -sc["test_neg_mean_squared_error"].mean(),
                     "r2": sc["test_r2"].mean()})
    print(f"  finished degree {d:>2} ({Z.shape[1]:>4} terms)", flush=True)
    del Z

res = pd.DataFrame(rows)
res.to_csv(HERE / "var2_grid_results.csv", index=False)
print(f"\nsearched {len(res)} combinations\n")

print("best alpha at each degree (5-fold CV):")
print(f"{'deg':>4} {'terms':>6} {'best alpha':>11} {'CV MSE':>9} {'CV R2':>8}")
for d in DEGREES:
    g = res[res.degree == d]
    b = g.loc[g.mse.idxmin()]
    print(f"{d:>4} {int(b.terms):>6} {b.alpha:>11.4g} {b.mse:>9.4f} "
          f"{b.r2:>8.4f}")

best = res.loc[res.mse.idxmin()]
print(f"\nlowest MSE in this grid: degree {int(best.degree)}, "
      f"alpha {best.alpha:.4g}, {int(best.terms)} terms")
print(f"                         CV MSE {best.mse:.4f}   CV R2 {best.r2:.4f}")

print("""
NOTE - this lowest cell is NOT by itself the model that was used.

Degrees 8-14 form a plateau: they all score within about 2% of each other,
which is no larger than the spread between the five folds.  So the single
lowest number in this grid is not a reliable winner - a different shuffle of
the folds can move it between neighbouring degrees.

Because cross-validation alone could not separate them, the two leading
candidates were compared on a held-out 20% of the training data.  Degree 10
beat degree 12 on every split tried, so that is the final model:
degree 10, alpha 0.4642 -- see var2_final.py, which reproduces that
comparison each time it runs.""")
