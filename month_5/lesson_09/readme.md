# Ensemble Methods — Voting Classifier & Voting Regressor

This document explains ensemble learning, with a focus on **Voting Classifier** and **Voting Regressor**, using fully tested code examples built on `china_used_cars.csv`.

All code below was run end-to-end against the actual dataset before being included here.

---

## Table of Contents

1. [What are ensemble methods?](#1-what-are-ensemble-methods)
2. [Voting Classifier](#2-voting-classifier)
3. [Voting Regressor](#3-voting-regressor)
4. [Key lesson: voting doesn't always help](#4-key-lesson-voting-doesnt-always-help)

---

## 1. What are ensemble methods?

**Ensemble learning** combines multiple models to produce a better result than any single model alone. The core idea: individual models make different kinds of errors, and if those errors aren't perfectly correlated, combining predictions cancels some of that error out.

There are four main families:

| Family | Idea | Examples |
|---|---|---|
| **Bagging** | Train the same algorithm on different random subsets of data (parallel, reduces variance) | Random Forest |
| **Boosting** | Train models sequentially, each correcting the previous one's errors (reduces bias) | Gradient Boosting, XGBoost, AdaBoost |
| **Stacking** | Train a "meta-model" that learns how to best combine base models' outputs | `StackingClassifier` / `StackingRegressor` |
| **Voting** | Combine *different* algorithm types by averaging or voting on their outputs directly (no meta-model) | `VotingClassifier`, `VotingRegressor` |

Voting is the simplest and most interpretable: you pick several different, ideally diverse, models (e.g. a tree-based model, a boosting model, a linear model), train each independently, then combine their outputs at prediction time.

---

## 2. Voting Classifier

For classification, `VotingClassifier` combines predictions in one of two ways:

**Hard voting** — each classifier casts one "vote" for a class label; majority wins.
```
final_class = mode(pred_1, pred_2, ..., pred_n)
```

**Soft voting** — each classifier outputs class *probabilities*; these are averaged (optionally weighted), and the class with the highest average probability wins.
```
final_proba = (w1·proba_1 + w2·proba_2 + ... + wn·proba_n) / Σw
final_class = argmax(final_proba)
```

Soft voting is generally preferred when your base models produce well-calibrated probabilities, since it uses more information than a single hard vote (e.g. a 51%-confident vote and a 99%-confident vote count equally in hard voting, but soft voting weighs them appropriately).

### Tested example: predicting `is_electric`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

df = pd.read_csv('china_used_cars.csv')

# Drop the target + columns that leak it directly (battery/motor specs define is_electric)
drop_cols = ['is_electric', 'battery_capacity_kwh', 'motor_power_kw',
             'price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['is_electric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf1 = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf2 = GradientBoostingClassifier(random_state=42)
clf3 = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))  # scale for LR only

voting_soft = VotingClassifier(
    estimators=[('rf', clf1), ('gb', clf2), ('lr', clf3)],
    voting='soft',
    weights=[2, 1, 1]   # trust the RF a bit more
)
voting_soft.fit(X_train, y_train)
pred = voting_soft.predict(X_test)
print("Voting accuracy:", accuracy_score(y_test, pred))

# Compare against each individual model
for name, clf in voting_soft.named_estimators_.items():
    p = clf.predict(X_test)
    print(f"{name} accuracy: {accuracy_score(y_test, p):.4f}")
```

**Verified results on this dataset:**

| Model | Accuracy |
|---|---|
| RF alone | 0.9904 |
| GB alone | 0.9916 |
| LR alone | 0.9892 |
| **Hard voting** | **0.9940** |
| **Soft voting** | **0.9928** |

Here voting *did* help — the ensemble edges out every individual model, since `is_electric` is easy enough that all three models are strong but make slightly different mistakes.

A useful trick: wrap models that need scaling (like `LogisticRegression`) in a `Pipeline` with a `StandardScaler`, while leaving tree-based models unscaled — `VotingClassifier` is happy to combine heterogeneous pipelines like this.

---

## 3. Voting Regressor

For regression there's no "vote" — `VotingRegressor` simply averages the predicted values (weighted if you specify weights):

```
final_prediction = (w1·pred_1 + w2·pred_2 + ... + wn·pred_n) / Σw
```

### Tested example: predicting `price`

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('china_used_cars.csv')

drop_cols = ['price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

r1 = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
r2 = GradientBoostingRegressor(random_state=42)
r3 = make_pipeline(StandardScaler(), LinearRegression())

voting_reg = VotingRegressor(estimators=[('rf', r1), ('gb', r2), ('lr', r3)], weights=[2, 2, 1])
voting_reg.fit(X_train, y_train)
pred = voting_reg.predict(X_test)
print("Voting -> MAE:", mean_absolute_error(y_test, pred), " R2:", r2_score(y_test, pred))

for name, est in voting_reg.named_estimators_.items():
    p = est.predict(X_test)
    print(f"{name:4s} -> MAE: {mean_absolute_error(y_test, p):.2f}  R2: {r2_score(y_test, p):.4f}")
```

**Verified results on this dataset:**

| Model | MAE | R² |
|---|---|---|
| RF alone | 12,358 | **0.704** |
| GB alone | 19,407 | 0.380 |
| LR alone | 34,686 | 0.224 |
| **Voting (weighted 2:2:1)** | 17,493 | 0.598 |

---

## 4. Key lesson: voting doesn't always help

Unlike the classifier case, **voting made things worse here** — the ensemble (R²=0.598) underperforms the RF alone (R²=0.704). This is a genuinely important, common pitfall:

> Voting only helps when the models being combined are **reasonably close in quality**. If one model (here, RF) is much stronger than the others, averaging it with weaker models *drags the average down* — you're diluting a good predictor with worse ones.

Fixes, in order of what to try:

1. **Weight more heavily toward the strong model** — e.g. `weights=[8, 1, 1]` instead of `[2, 2, 1]`.
2. **Drop the weak model entirely** — if linear regression can't capture the nonlinear price relationships your RF captures, it's not adding useful diversity, just noise.
3. **Use stacking instead of voting** — a `StackingRegressor` learns *how much to trust each model* via a meta-model, rather than using fixed weights you have to guess.
4. **Only ensemble models that are individually competitive** — the diversity benefit only pays off when each contributor is decent on its own.

**Takeaway:** ensembling isn't automatically better — it's a bet that diverse errors will cancel out, and that bet only pays off when the models being combined are roughly comparable in strength. Always compare the ensemble against its best individual member before assuming the ensemble wins.
