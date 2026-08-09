# Ensemble Methods — Head-to-Head Comparison

This document runs **every ensemble method covered so far** (bagging, boosting variants, voting, stacking) on the same task — predicting `price` from `china_used_cars.csv` — using the same train/test split and feature set, for a fair, apples-to-apples comparison.

See also: `README.md` (SHAP + multi-output), `README_ensemble_methods.md` (voting), `README_stacking.md`, `README_bagging.md`, `README_boosting.md` for deep dives on each individual technique.

---

## Setup

All models are trained on the same 80/20 split (`random_state=42`) with the same dropped columns to avoid leakage (`price` itself, plus `mileage_km`, `log_mileage`, `mileage_per_year`, `year`, `month`).

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    BaggingRegressor, RandomForestRegressor, AdaBoostRegressor,
    GradientBoostingRegressor, VotingRegressor, StackingRegressor
)
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb

df = pd.read_csv('china_used_cars.csv')
drop_cols = ['price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate(name, model):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(f"{name:20s} MAE: {mean_absolute_error(y_test, pred):10.1f}  R2: {r2_score(y_test, pred):.4f}")
```

---

## All methods, run together

```python
evaluate("Decision Tree", DecisionTreeRegressor(random_state=42))

evaluate("Bagging", BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=42),
    n_estimators=200, random_state=42, n_jobs=-1
))

evaluate("Random Forest", RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1
))

evaluate("AdaBoost", AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=4, random_state=42),
    n_estimators=200, learning_rate=0.5, random_state=42
))

evaluate("Gradient Boosting", GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42
))

evaluate("XGBoost", xgb.XGBRegressor(
    n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42, n_jobs=-1
))

base = [
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('lr', make_pipeline(StandardScaler(), LinearRegression())),
]

evaluate("Voting", VotingRegressor(estimators=base, weights=[2, 2, 1]))

evaluate("Stacking", StackingRegressor(
    estimators=base, final_estimator=RidgeCV(), cv=5, n_jobs=-1
))
```

---

## Final results — sorted by R² (higher is better)

| Rank | Model | MAE | R² |
|---|---|---|---|
| 1 | **Bagging** | 12,027 | **0.7401** |
| 2 | Stacking | 17,240 | 0.7062 |
| 3 | Random Forest | 12,358 | 0.7044 |
| 4 | Voting | 17,493 | 0.5977 |
| 5 | Decision Tree (single, no ensemble) | 12,486 | 0.5948 |
| 6 | AdaBoost | 40,258 | 0.5645 |
| 7 | XGBoost | 15,671 | 0.5435 |
| 8 | Gradient Boosting | 18,772 | 0.4449 |

---

## What this run shows on this dataset

- **Bagging wins outright** on both metrics — the best MAE *and* the best R². Simple row-level ensembling of decision trees beat every fancier method here.
- **Random Forest and Stacking are essentially tied for 2nd/3rd** — makes sense, since Random Forest *is* bagging with feature randomness added, and Stacking's meta-model learned to lean heavily on Random Forest as its strongest input.
- **Boosting methods (AdaBoost, Gradient Boosting, XGBoost) all underperform bagging here**, none using tuned hyperparameters — boosting is generally more hyperparameter-sensitive, so this isn't "boosting is worse," it's "these default-ish settings weren't tuned for this dataset."
- **Voting comes in below even a single decision tree** — a repeat of the pattern seen in `README_ensemble_methods.md`: averaging in a weak Gradient Boosting/Linear Regression model dragged down what would otherwise be a strong ensemble.

## Bottom line

Row-level ensembling (bagging, Random Forest) suits this price-prediction problem better than sequential error-correction (boosting) or manual-weight combination (voting) — at least without further tuning. If you wanted to push boosting past bagging's 0.74, the next step would be proper hyperparameter search (Optuna) specifically for XGBoost/Gradient Boosting, since their untuned numbers here are likely leaving real performance on the table.

## Quick reference: which method to reach for

| Situation | Recommended starting point |
|---|---|
| Need a strong baseline fast, minimal tuning | Random Forest or plain Bagging |
| Base models vary a lot in strength, want automatic weighting | Stacking |
| Base models are all roughly comparable in quality | Voting |
| Willing to invest in tuning (`learning_rate`, `n_estimators`, depth) for best possible accuracy | XGBoost / Gradient Boosting |
| Want the most interpretable ensemble | Bagging or Random Forest (feature importances readily available) |
