from pathlib import Path
import numpy as np, pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

COLS = [f"x{i}" for i in range(1, 7)]
HERE = Path(__file__).resolve().parent     # relative to this script, not to cwd
df = pd.read_csv(HERE / "IMT2024063_train_var1.csv")
X, y = df[COLS].to_numpy(), df["y"].to_numpy()

# ---- 1. lock away 200 rows BEFORE anything else -----------------------
X_pool, X_lock, y_pool, y_lock = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(f"search pool {len(y_pool)} rows | locked away {len(y_lock)} rows\n")

# ---- 2. the whole search, on the pool only ----------------------------
pipe = Pipeline([("poly", PolynomialFeatures(include_bias=False)),
                 ("scale", StandardScaler()), ("ridge", Ridge())])
grid = {"poly__degree": list(range(1, 9)),
        "ridge__alpha": list(np.logspace(-2, 3, 21))}
gs = GridSearchCV(pipe, grid, scoring="neg_mean_squared_error",
                  cv=KFold(5, shuffle=True, random_state=0), n_jobs=-1)
gs.fit(X_pool, y_pool)

b = gs.best_params_
print(f"search on 800 rows picked: degree {b['poly__degree']}, "
      f"alpha {b['ridge__alpha']:.4g}")
print(f"  its CV MSE on the pool   : {-gs.best_score_:.4f}")

# ---- 3. score on the locked rows --------------------------------------
pred = gs.best_estimator_.predict(X_lock)
print(f"\nTRULY INDEPENDENT estimate (200 rows nothing above ever touched):")
print(f"  MSE {mean_squared_error(y_lock, pred):.4f}   "
      f"R2 {r2_score(y_lock, pred):.4f}")

# ---- for comparison: the model actually shipped -----------------------
ship = Pipeline([("poly", PolynomialFeatures(5, include_bias=False)),
                 ("scale", StandardScaler()),
                 ("ridge", Ridge(alpha=17.783))]).fit(X_pool, y_pool)
print(f"\nshipped config (degree 5, alpha 17.783) on the same locked rows:")
print(f"  MSE {mean_squared_error(y_lock, ship.predict(X_lock)):.4f}   "
      f"R2 {r2_score(y_lock, ship.predict(X_lock)):.4f}")
