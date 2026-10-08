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
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# the plain-least-squares curve in the plot is deliberately unregularised and
# so is ill-conditioned at high degree; that is the point of showing it
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)

# ---------------------------------------------------------------- settings
RUN_SEARCH = False            # True re-runs the full sweep (slow)

DEGREE = 10                   # chosen by the sweep plus the held-out tiebreak
ALPHA = 0.4642                # ridge penalty strength
RIVAL_DEGREE, RIVAL_ALPHA = 12, 1.7783      # the grid minimum, for comparison

HERE = Path(__file__).resolve().parent        # paths relative to this file,
COLS = ["x1", "x2", "x3"]                     # not to the working directory
TRAIN = HERE / "IMT2024063_train_var2.csv"
TEST = HERE / "IMT2024063_test_var2.csv"
PRED = HERE / "IMT2024063_pred_var2.csv"
PLOT = HERE / "var2_degree_selection.png"

cv = KFold(n_splits=5, shuffle=True, random_state=0)


def model(degree=DEGREE, alpha=ALPHA, kind="ridge"):
    fit = {"ridge": Ridge(alpha=alpha),
           "lasso": Lasso(alpha=alpha, max_iter=50000),
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
    print("full sweep: degrees 1-20 x alpha, 5-fold CV\n")
    best = {}
    for kind, alphas in grids.items():
        print(f"{'deg':>4} {'terms':>6} {'best alpha':>11} {'CV MSE':>9} "
              f"{'CV R2':>8}   [{kind}]")
        for d in range(1, 21):
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
    print("\nNote: the lowest ridge cell is degree 12, but degrees 8-14 are a"
          "\nplateau, so the held-out comparison below decides between them.")
    return best


if RUN_SEARCH:
    search()


# ======================================= 2. held-out tiebreak: degree 10 vs 12
print("Held-out 20%: degree 10 against degree 12 (the grid minimum)")
print(f"{'split':<10} {'degree 10 MSE':>14} {'degree 12 MSE':>14} {'winner':>9}")
for seed in (42, 1, 7):
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.2, random_state=seed)
    m10 = mean_squared_error(yb, model(DEGREE, ALPHA).fit(Xa, ya).predict(Xb))
    m12 = mean_squared_error(
        yb, model(RIVAL_DEGREE, RIVAL_ALPHA).fit(Xa, ya).predict(Xb))
    print(f"seed {seed:<5} {m10:>14.4f} {m12:>14.4f} "
          f"{('deg 10' if m10 < m12 else 'deg 12'):>9}")


# ============================================== 3. scores for the chosen model
Xtr, Xho, ytr, yho = train_test_split(X, y, test_size=0.2, random_state=42)
m = model().fit(Xtr, ytr)
print(f"\nChosen model on the held-out 20%:")
print(f"  train MSE {mean_squared_error(ytr, m.predict(Xtr)):.4f}   "
      f"holdout MSE {mean_squared_error(yho, m.predict(Xho)):.4f}   "
      f"R2 {r2_score(yho, m.predict(Xho)):.4f}")

cv_mse = -cross_val_score(model(), X, y, cv=cv,
                          scoring="neg_mean_squared_error").mean()
cv_r2 = cross_val_score(model(), X, y, cv=cv, scoring="r2").mean()
print(f"5-fold CV over all {len(y)} rows:  MSE {cv_mse:.4f}   R2 {cv_r2:.4f}")


# ========================================================== 4. final model
final = model().fit(X, y)
print(f"\nFinal model: degree {DEGREE}, Ridge alpha {ALPHA}, "
      f"{final[0].n_output_features_} terms, trained on {len(y)} rows")


# ====================================================== 5. test predictions
test = pd.read_csv(TEST)
pred = final.predict(test[COLS].to_numpy())      # original row order preserved
pd.DataFrame({"y": pred}).to_csv(PRED, index=False)

print("\nSanity check")
print(f"  training y : mean {y.mean():7.3f}  std {y.std():6.3f}  "
      f"min {y.min():8.3f}  max {y.max():7.3f}")
print(f"  predicted y: mean {pred.mean():7.3f}  std {pred.std():6.3f}  "
      f"min {pred.min():8.3f}  max {pred.max():7.3f}")
print("  nine predictions sit above the training maximum.  They are all at"
      "\n  corners of the input cube, where no training sample is close and y"
      "\n  varies greatly.  They were left unclipped: the true surface can"
      "\n  exceed the largest sampled value, and clipping would add bias.")
print(f"  wrote {len(pred)} predictions to {PRED.name}")


# =============================================================== 6. plot
degs = list(range(1, 21))
ridge_mse, ols_mse, ses = [], [], []
for d in degs:
    s = -cross_val_score(model(d, ALPHA, "ridge"), X, y, cv=cv,
                         scoring="neg_mean_squared_error")
    ridge_mse.append(s.mean())
    ses.append(s.std(ddof=1) / np.sqrt(len(s)))
    ols_mse.append(-cross_val_score(model(d, 0.0, "ols"), X, y, cv=cv,
                                    scoring="neg_mean_squared_error").mean())

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.errorbar(degs, ridge_mse, yerr=ses, fmt="o-", capsize=3,
            label=f"Ridge (alpha={ALPHA})")
ax.plot(degs, ols_mse, "s--", alpha=.7, label="Plain least squares")
ax.axvspan(8, 14, color="green", alpha=.10,
           label="flat region: degrees 8-14 within ~2%")
ax.axvline(DEGREE, color="grey", ls=":", lw=1)
ax.annotate(f"chosen: degree {DEGREE}", (DEGREE, ridge_mse[DEGREE - 1]),
            textcoords="offset points", xytext=(18, 38),
            arrowprops=dict(arrowstyle="->", color="grey"))
ax.set_yscale("log")
ax.set_xticks(degs)
ax.tick_params(axis="x", labelsize=8)
ax.set_xlabel("polynomial degree (total degree)")
ax.set_ylabel("5-fold CV MSE (log scale)")
ax.set_title("var2: choosing the polynomial degree")
ax.legend()
fig.tight_layout()
fig.savefig(PLOT, dpi=150)
print(f"  wrote {PLOT.name}")
