import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)

RUN_SEARCH = False            

DEGREE = 5              
ALPHA = 0.0056234            
RIDGE_ALPHA = 17.783         

HERE = Path(__file__).resolve().parent    
COLS = [f"x{i}" for i in range(1, 7)]      
TRAIN = HERE / "IMT2024063_train_var1.csv"
TEST = HERE / "IMT2024063_test_var1.csv"
PRED = HERE / "IMT2024063_pred_var1.csv"
PLOT = HERE / "var1_degree_selection.png"

cv = KFold(n_splits=5, shuffle=True, random_state=0)


def model(degree=DEGREE, alpha=ALPHA, kind="lasso"):
    fit = {"lasso": Lasso(alpha=alpha, max_iter=50000),
           "ridge": Ridge(alpha=alpha),
           "elasticnet": ElasticNet(alpha=alpha, l1_ratio=0.5, max_iter=50000),
           "ols": Ridge(alpha=1e-10)}[kind]
    return make_pipeline(PolynomialFeatures(degree, include_bias=False),
                         StandardScaler(), fit)


df = pd.read_csv(TRAIN)
X, y = df[COLS].to_numpy(), df["y"].to_numpy()


# ================================================================= 1. search
def search():

    grids = {"ridge": np.logspace(-2, 3, 21),
             "lasso": np.logspace(-4, 0.5, 19),
             "elasticnet": np.logspace(-4, 0.5, 19)}
    print("full sweep: degrees 1-10 x alpha, 5-fold CV\n")
    best = {}
    for kind, alphas in grids.items():
        print(f"{'deg':>4} {'terms':>6} {'best alpha':>11} {'CV MSE':>9} "
              f"{'CV R2':>8}   [{kind}]")
        for d in range(1, 11):
            cell = None
            for a in alphas:
                m = -cross_val_score(model(d, a, kind), X, y, cv=cv,
                                     scoring="neg_mean_squared_error").mean()
                if cell is None or m < cell[0]:
                    cell = (m, a)
            n = PolynomialFeatures(d, include_bias=False).fit(X[:2]).n_output_features_
            r2 = cross_val_score(model(d, cell[1], kind), X, y, cv=cv,
                                 scoring="r2").mean()
            print(f"{d:>4} {n:>6} {cell[1]:>11.5g} {cell[0]:>9.4f} {r2:>8.4f}")
            if kind not in best or cell[0] < best[kind][0]:
                best[kind] = (cell[0], d, cell[1])
        print()
    for kind, (m, d, a) in best.items():
        print(f"best {kind:<11}: degree {d}, alpha {a:.5g}, CV MSE {m:.4f}")
    return best


if RUN_SEARCH:
    search()


# ============================================== 2. scores for the chosen model
Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, random_state=42)

print("Held-out 20% (200 rows), lasso against the ridge baseline")
print(f"{'model':<26} {'train MSE':>10} {'holdout MSE':>12} {'holdout R2':>11}")
for label, kind, alpha in [("Ridge  alpha 17.783", "ridge", RIDGE_ALPHA),
                           ("Lasso  alpha 0.0056234", "lasso", ALPHA)]:
    m = model(DEGREE, alpha, kind).fit(Xtr, ytr)
    print(f"{label:<26} {mean_squared_error(ytr, m.predict(Xtr)):>10.4f} "
          f"{mean_squared_error(yho, m.predict(Xho)):>12.4f} "
          f"{r2_score(yho, m.predict(Xho)):>11.4f}")

cv_mse = -cross_val_score(model(), X, y, cv=cv,
                          scoring="neg_mean_squared_error").mean()
cv_r2 = cross_val_score(model(), X, y, cv=cv, scoring="r2").mean()
print(f"\n5-fold CV over all {len(y)} rows:  MSE {cv_mse:.4f}   R2 {cv_r2:.4f}")


# ========================================================== 3. final model
# Refit on all 1000 rows.  The degree is already settled, so there is no
# reason to hold any data back now.
final = model().fit(X, y)
coef = final[-1].coef_
kept = int((coef != 0).sum())
print(f"\nFinal model: degree {DEGREE}, Lasso alpha {ALPHA}, "
      f"{kept} of {len(coef)} terms kept ({kept/len(coef):.0%})")

names = final[0].get_feature_names_out(COLS)
print("\n  ten largest surviving terms:")
for i in np.argsort(-np.abs(coef))[:10]:
    if coef[i]:
        print(f"    {names[i]:<22} {coef[i]:+9.3f}")


# ====================================================== 4. test predictions
test = pd.read_csv(TEST)
pred = final.predict(test[COLS].to_numpy())      # original row order preserved
pd.DataFrame({"y": pred}).to_csv(PRED, index=False)

print("\nSanity check")
print(f"  training y : mean {y.mean():7.3f}  std {y.std():6.3f}  "
      f"min {y.min():7.3f}  max {y.max():7.3f}")
print(f"  predicted y: mean {pred.mean():7.3f}  std {pred.std():6.3f}  "
      f"min {pred.min():7.3f}  max {pred.max():7.3f}")
print("  a wider spread is expected: the test set is edge-heavy (54% of its"
      "\n  inputs exceed |0.9| against 36% in training) and y varies more near"
      "\n  the edges, so the predictions should spread more than training y.")
print(f"  wrote {len(pred)} predictions to {PRED.name}")


# =============================================================== 5. plot
degs = range(1, 11)
curves = {f"Lasso (alpha={ALPHA})": ("lasso", ALPHA),
          f"Ridge (alpha={RIDGE_ALPHA})": ("ridge", RIDGE_ALPHA),
          "Plain least squares": ("ols", 0.0)}
values = {k: [] for k in curves}
for d in degs:
    for label, (kind, alpha) in curves.items():
        values[label].append(
            -cross_val_score(model(d, alpha, kind), X, y, cv=cv,
                             scoring="neg_mean_squared_error").mean())

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for (label, vals), style in zip(values.items(), ["o-", "^-", "s--"]):
    ax.plot(list(degs), vals, style, label=label)
ax.axvline(DEGREE, color="grey", ls=":", lw=1)
ax.annotate(f"chosen: degree {DEGREE}, Lasso",
            (DEGREE, values[f"Lasso (alpha={ALPHA})"][DEGREE - 1]),
            textcoords="offset points", xytext=(15, -32),
            arrowprops=dict(arrowstyle="->", color="grey"))
ax.set_yscale("log")
ax.set_xticks(list(degs))
ax.set_xlabel("polynomial degree (total degree)")
ax.set_ylabel("5-fold CV MSE (log scale)")
ax.set_title("var1: choosing the polynomial degree and the penalty")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT, dpi=150)
print(f"  wrote {PLOT.name}")
