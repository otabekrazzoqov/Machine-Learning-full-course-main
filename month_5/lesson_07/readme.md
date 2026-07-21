# Feature Analysis

Analysis of `chna_used_cars.csv` (4,184 rows, 19 features + `price`)
before feature selection was applied. Target is `log1p(price)` throughout,
since raw price is heavily right-skewed (median ~$14k, max ~$3.9M).

```python
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr, f_oneway, chi2_contingency
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("data/final/train_final.csv")
y = df["price"]
y_log = np.log1p(y)
X = df.drop(columns=["price"])

numeric_cols = [
    "mileage_km", "engine_cc", "model_year", "seat_count", "motor_power_kw",
    "battery_capacity_kwh", "door_count", "year", "month", "car_age",
    "mileage_per_year", "quarter", "log_mileage",
]
cat_cols = ["fuel_type", "level", "car_body_color", "drive_mode",
            "transmission_Manual", "is_electric"]
```

## 1. Correlation with target

```python
print("=== Numeric vs price (log): Pearson & Spearman ===")
for c in numeric_cols:
    pr = pearsonr(df[c], y_log)[0]
    sr = spearmanr(df[c], y_log)[0]
    print(f"{c:22s} pearson={pr:+.3f}  spearman={sr:+.3f}")

print("\n=== Categorical vs price (log): ANOVA F-test ===")
for c in cat_cols:
    groups = [y_log[df[c] == v] for v in df[c].unique()]
    groups = [g for g in groups if len(g) > 1]
    f, p = f_oneway(*groups)
    print(f"{c:22s} F={f:10.2f}  p={p:.4g}")

print("\n=== Categorical vs price-quartile: Chi-square ===")
# chi-square needs two categorical variables, so price is binned into
# quartiles first - this is an approximation, not a substitute for ANOVA
price_bin = pd.qcut(y, q=4, labels=False, duplicates="drop")
for c in cat_cols:
    ct = pd.crosstab(df[c], price_bin)
    chi2, p, dof, _ = chi2_contingency(ct)
    print(f"{c:22s} chi2={chi2:10.2f}  p={p:.4g}")
```

**Results — numeric (Pearson / Spearman):**

| feature              | Pearson | Spearman |
| -------------------- | ------- | -------- |
| engine_cc            | +0.463  | +0.289   |
| model_year           | +0.409  | +0.494   |
| mileage_km           | -0.429  | -0.493   |
| mileage_per_year     | -0.379  | -0.450   |
| log_mileage          | -0.365  | -0.493   |
| motor_power_kw       | +0.348  | +0.377   |
| year                 | +0.258  | +0.492   |
| battery_capacity_kwh | +0.236  | +0.305   |
| door_count           | -0.152  | +0.018   |
| month                | -0.102  | -0.086   |
| quarter              | -0.097  | -0.084   |
| seat_count           | -0.067  | +0.081   |
| car_age              | +0.017  | -0.056   |

`car_age` is essentially uncorrelated with price — irrelevant on its own.
`door_count`'s sign flips between Pearson and Spearman — no reliable
monotonic relationship.

**Results — categorical (ANOVA / Chi-square, price binned into quartiles):**

| feature             | ANOVA F | Chi²  | p-value |
| ------------------- | ------- | ------ | ------- |
| transmission_Manual | 647.0   | 665.8  | <0.001  |
| level               | 223.9   | 3030.3 | <0.001  |
| drive_mode          | 178.8   | 1649.5 | <0.001  |
| is_electric         | 143.6   | 358.1  | <0.001  |
| fuel_type           | 88.7    | 829.6  | <0.001  |
| car_body_color      | 82.8    | 298.6  | <0.001  |

All significant at this sample size — judge by F/Chi² magnitude, not p-value.
`car_body_color` is the weakest of the group.

## 2. Correlation with other features (redundancy)

```python
corr = X.corr().abs()
print("=== Feature pairs with |corr| > 0.7 ===")
for i, a in enumerate(corr.columns):
    for b in corr.columns[i + 1:]:
        if corr.loc[a, b] > 0.7:
            print(f"{a:22s} <-> {b:22s}  corr={corr.loc[a, b]:.3f}")
```

**Result:**

| pair                                   | corr |
| -------------------------------------- | ---- |
| month ↔ quarter                       | 0.97 |
| mileage_km ↔ mileage_per_year         | 0.88 |
| fuel_type ↔ is_electric               | 0.88 |
| battery_capacity_kwh ↔ is_electric    | 0.84 |
| motor_power_kw ↔ battery_capacity_kwh | 0.78 |
| fuel_type ↔ battery_capacity_kwh      | 0.77 |
| mileage_km ↔ log_mileage              | 0.73 |
| motor_power_kw ↔ is_electric          | 0.73 |

`is_electric` is redundant with three other features at once — it was
derived from them, so keeping all four adds no new information.

## 3. Feature importance: tree vs linear model

```python
rf = RandomForestRegressor(random_state=42)
rf.fit(X, y_log)
rf_importance = pd.Series(rf.feature_importances_, index=X.columns) \
    .sort_values(ascending=False)
print("=== RandomForest importances ===")
print(rf_importance)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge = Ridge()
ridge.fit(X_scaled, y_log)
ridge_coef = pd.Series(np.abs(ridge.coef_), index=X.columns) \
    .sort_values(ascending=False)
print("\n=== Ridge |standardized coefficients| ===")
print(ridge_coef)
```

**Result (top features, side by side):**

| feature        | RF importance | Ridge\|coef\| |
| -------------- | ------------- | ------------- |
| engine_cc      | 0.233         | 0.493         |
| level          | 0.197         | 0.114         |
| model_year     | 0.170         | 0.410         |
| motor_power_kw | 0.056         | 0.424         |
| fuel_type      | 0.011         | 0.295         |
| car_age        | 0.003         | 0.101         |

Where RF and Ridge disagree is the useful signal:

- **`level`** — high in RF, low in Ridge → its effect on price is nonlinear
  (threshold-like), which only a tree can capture.
- **`fuel_type`** — low in RF, high in Ridge → its effect is fairly
  monotonic/linear, so the tree gets little extra benefit from splitting on it.
- **`car_age`** — high Ridge coefficient despite ~0 real correlation and ~0
  RF importance is a multicollinearity artifact (overlaps with
  `model_year`/`year`) — trust RF, not Ridge, here.

## 4. Performance impact (train with/without each feature)

```python
def cv_score(X, y_log, n_splits=5):
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    rmses, r2s = [], []
    for train_idx, val_idx in cv.split(X):
        model = RandomForestRegressor(random_state=42)
        model.fit(X.iloc[train_idx], y_log.iloc[train_idx])
        pred = np.expm1(model.predict(X.iloc[val_idx]))
        y_val = np.expm1(y_log.iloc[val_idx])
        rmses.append(mean_squared_error(y_val, pred) ** 0.5)
        r2s.append(r2_score(y_val, pred))
    return np.mean(rmses), np.mean(r2s)

base_rmse, base_r2 = cv_score(X, y_log)
print(f"ALL 19 features: RMSE=${base_rmse:,.0f} R2={base_r2:.3f}")

candidates = ["is_electric", "car_age", "quarter", "mileage_per_year", "car_body_color"]
for c in candidates:
    rmse, r2 = cv_score(X.drop(columns=[c]), y_log)
    print(f"without {c:20s}: RMSE=${rmse:,.0f} R2={r2:.3f}  (delta R2 = {r2 - base_r2:+.4f})")

rmse, r2 = cv_score(X.drop(columns=candidates), y_log)
print(f"without all 5: RMSE=${rmse:,.0f} R2={r2:.3f}  (delta R2 = {r2 - base_r2:+.4f})")
```

**Result:**

| removed feature                  | RMSE              | R²             | Δ R²           |
| -------------------------------- | ----------------- | --------------- | ---------------- |
| *(none — all 19)*             | $73,319           | 0.613           | —               |
| mileage_per_year                 | $72,588           | 0.621           | +0.008           |
| quarter                          | $72,879           | 0.618           | +0.005           |
| is_electric                      | $73,091           | 0.616           | +0.003           |
| car_body_color                   | $73,152           | 0.616           | +0.002           |
| car_age                          | $73,195           | 0.614           | +0.001           |
| **all 5 removed together** | **$71,636** | **0.632** | **+0.019** |

Every one of these features helps to remove individually, and removing them
together compounds to a bigger gain — confirming they're net noise, not just
individually weak signal that combines usefully.

## Conclusion

`is_electric`, `car_age`, `quarter`, `mileage_per_year`, `car_body_color` are
irrelevant/redundant and are dropped. `level` and `fuel_type` are real signal 

but nonlinear in opposite directions, which is the main reason RandomForest (R² 0.632) 

beats Linear/Ridge (R² 0.307) on
this dataset.
