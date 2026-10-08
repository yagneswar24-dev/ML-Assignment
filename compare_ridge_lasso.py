import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.exceptions import ConvergenceWarning
from scipy.linalg import LinAlgWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent     # relative to this script, not to cwd
PROBLEMS = {
    "var1": dict(cols=[f"x{i}" for i in range(1, 7)], degrees=range(1, 11)),
    "var2": dict(cols=["x1", "x2", "x3"], degrees=range(1, 21)),
}
METHODS = {
    "Ridge":      (lambda a: Ridge(alpha=a),                np.logspace(-2, 3, 21)),
    "Lasso":      (lambda a: Lasso(alpha=a, max_iter=20000),
                   np.logspace(-4, 0.5, 19)),
    "ElasticNet": (lambda a: ElasticNet(alpha=a, l1_ratio=0.5, max_iter=20000),
                   np.logspace(-4, 0.5, 19)),
}

# one 5-fold split, shared by every method so the comparison is like-for-like
cv = KFold(5, shuffle=True, random_state=0)

for name, cfg in PROBLEMS.items():
    df = pd.read_csv(HERE / f"IMT2024063_train_{name}.csv")
    X, y = df[cfg["cols"]].to_numpy(), df["y"].to_numpy()
    X_pool, X_lock, y_pool, y_lock = train_test_split(
        X, y, test_size=0.2, random_state=42)

    print(f"\n{'='*84}\n{name}  ({X.shape[1]} inputs, {len(y)} rows)  "
          f"degrees {cfg['degrees'].start}-{cfg['degrees'].stop - 1}\n{'='*84}",
          flush=True)

    # per-degree best for every method -> this is the rejection evidence
    table = {m: {} for m in METHODS}
    for d in cfg["degrees"]:
        Z = PolynomialFeatures(d, include_bias=False).fit_transform(X)
        for mname, (make, alphas) in METHODS.items():
            best = None
            for a in alphas:
                s = -cross_val_score(make_pipeline(StandardScaler(), make(a)),
                                     Z, y, cv=cv,
                                     scoring="neg_mean_squared_error").mean()
                if best is None or s < best[0]:
                    best = (s, a)
            table[mname][d] = (best[0], best[1], Z.shape[1])
        row = "  ".join(f"{m} {table[m][d][0]:8.4f}" for m in METHODS)
        print(f"  degree {d:>2} ({Z.shape[1]:>5} terms):  {row}", flush=True)
        del Z

    print(f"\n{'deg':>4} {'terms':>6}" + "".join(f"{m:>22}" for m in METHODS))
    print(f"{'':>11}" + "".join(f"{'alpha':>11}{'CV MSE':>11}" for _ in METHODS))
    for d in cfg["degrees"]:
        line = f"{d:>4} {table['Ridge'][d][2]:>6}"
        for m in METHODS:
            mse, a, _ = table[m][d]
            line += f"{a:>11.4g}{mse:>11.4f}"
        print(line)

    # ---- the winner of each method, with the extra detail --------------
    print(f"\n{'method':<11} {'deg':>4} {'alpha':>10} {'CV MSE':>9} {'CV R2':>8} "
          f"{'kept':>6} {'fit ms':>8} {'holdout MSE':>12} {'holdout R2':>11}")
    summary = {}
    for mname, (make, _) in METHODS.items():
        d = min(table[mname], key=lambda k: table[mname][k][0])
        mse, a, nterms = table[mname][d]
        Z = PolynomialFeatures(d, include_bias=False).fit_transform(X)
        r2 = cross_val_score(make_pipeline(StandardScaler(), make(a)), Z, y,
                             cv=cv, scoring="r2").mean()
        t = time.perf_counter()
        fitted = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(),
                               make(a)).fit(X, y)
        ms = (time.perf_counter() - t) * 1000
        kept = int((fitted[-1].coef_ != 0).sum())
        hm = make_pipeline(PolynomialFeatures(d, include_bias=False), StandardScaler(),
                           make(a)).fit(X_pool, y_pool)
        hp = hm.predict(X_lock)
        summary[mname] = dict(deg=d, terms=nterms, kept=kept, model=fitted)
        print(f"{mname:<11} {d:>4} {a:>10.4g} {mse:>9.4f} {r2:>8.4f} {kept:>6} "
              f"{ms:>8.1f} {mean_squared_error(y_lock, hp):>12.4f} "
              f"{r2_score(y_lock, hp):>11.4f}")

    ls = summary["Lasso"]
    names = ls["model"][0].get_feature_names_out(cfg["cols"])
    coef = ls["model"][-1].coef_
    print(f"\nLasso kept {ls['kept']} of {ls['terms']} terms "
          f"({ls['kept']/ls['terms']:.0%}).  Ten largest:")
    for i in np.argsort(-np.abs(coef))[:10]:
        if coef[i] != 0:
            print(f"    {names[i]:<24} {coef[i]:+9.3f}")

