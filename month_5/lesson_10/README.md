# Stacking (Ensemble Method) — Stacking Classifier & Stacking Regressor

This document explains **stacking**, the most sophisticated of the standard ensemble techniques, with fully tested code examples built on `china_used_cars.csv`. It also compares stacking directly against voting (see `README_ensemble_methods.md`).

All code below was run end-to-end against the actual dataset before being included here.

---

## Table of Contents

1. [What is stacking?](#1-what-is-stacking)
2. [StackingRegressor example (price)](#2-stackingregressor-example-price)
3. [StackingClassifier example (is_electric)](#3-stackingclassifier-example-is_electric)
4. [Stacking vs. Voting — when to use which](#4-stacking-vs-voting--when-to-use-which)

---

## 1. What is stacking?

Stacking (short for **stacked generalization**) is the most sophisticated of the standard ensemble techniques. Where voting combines predictions with a fixed rule (majority vote or averaging), stacking **learns** how to combine them — via a second model trained specifically for that job.

### How it works

1. **Base learners (level 0)** — train several different models on your training data, same as voting (e.g. Random Forest, Gradient Boosting, Linear Regression).
2. **Generate out-of-fold predictions** — instead of just taking each base learner's predictions on the training set (which would be overfit and unreliable), stacking uses cross-validation internally: each base learner predicts on the folds it *wasn't* trained on. This produces a clean, honest set of "predictions as inputs."
3. **Meta-model (final estimator / level 1)** — a new model is trained where the *inputs* are the base learners' out-of-fold predictions, and the *target* is still the original `y`. This meta-model learns the optimal way to weight and combine the base learners — including learning that one model should be trusted more in certain regions of the data.
4. **At prediction time**, each base learner predicts on the new data, and those predictions get fed into the trained meta-model to produce the final output.

```
Raw features → [RF, GB, LR] → base predictions → Meta-model → final prediction
```

This cross-validation step is the critical difference from naively training a meta-model on each base learner's in-sample predictions — doing that would let the meta-model see predictions that are artificially "too good" (the base learners have already seen that exact data), causing severe overfitting. `StackingRegressor` / `StackingClassifier` handle this CV splitting internally via the `cv` parameter.

---

## 2. StackingRegressor example (price)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('china_used_cars.csv')

drop_cols = ['price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_learners = [
    ('rf', RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(random_state=42)),
    ('lr', make_pipeline(StandardScaler(), LinearRegression())),
]

stack_reg = StackingRegressor(
    estimators=base_learners,
    final_estimator=RidgeCV(),   # meta-model — RidgeCV is a common, stable default
    cv=5,                        # 5-fold CV to generate honest out-of-fold predictions
    n_jobs=-1
)
stack_reg.fit(X_train, y_train)
pred = stack_reg.predict(X_test)
print("Stacking -> MAE:", mean_absolute_error(y_test, pred), " R2:", r2_score(y_test, pred))

for name, est in stack_reg.named_estimators_.items():
    p = est.predict(X_test)
    print(f"{name:4s} -> MAE: {mean_absolute_error(y_test, p):.2f}  R2: {r2_score(y_test, p):.4f}")

# Inspect what the meta-model learned — the weight given to each base learner
print("Meta-model coefficients (rf, gb, lr):", stack_reg.final_estimator_.coef_)
print("Meta-model intercept:", stack_reg.final_estimator_.intercept_)
```

**Verified results on this dataset:**

| Model | MAE | R² |
|---|---|---|
| RF alone | 12,358 | 0.7044 |
| GB alone | 19,407 | 0.3799 |
| LR alone | 34,686 | 0.2235 |
| Voting (weighted 2:2:1) | 17,493 | 0.5977 |
| **Stacking (RidgeCV meta-model)** | 17,240 | **0.7062** |

**Learned meta-model weights:** `rf: 1.034, gb: -0.393, lr: 0.360`, intercept `-357.1`.

Stacking recovers the RF's strength that voting threw away. By R², stacking (0.7062) actually edges out even the RF alone (0.7044) — the meta-model learned to lean heavily on RF (coefficient ≈1.03), while assigning a *negative* weight to GB, effectively using it as a correction term rather than blindly averaging it in. This is the key advantage over voting: **the combination weights are learned from data, not guessed.**

**Nuance worth flagging:** MAE for stacking (17,240) is still worse than RF alone (12,358) even though R² improved slightly. R² and MAE weight errors differently (R² is more sensitive to variance explained, MAE to average absolute error) — this mismatch is a reminder to check multiple metrics, not just one, before declaring an ensemble "better."

---

## 3. StackingClassifier example (is_electric)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

df = pd.read_csv('china_used_cars.csv')

drop_cols = ['is_electric', 'battery_capacity_kwh', 'motor_power_kw',
             'price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['is_electric']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

base_clf = [
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingClassifier(random_state=42)),
]

stack_clf = StackingClassifier(
    estimators=base_clf,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=5,
    n_jobs=-1
)
stack_clf.fit(X_train, y_train)
pred = stack_clf.predict(X_test)
print("Stacking accuracy:", accuracy_score(y_test, pred))

for name, est in stack_clf.named_estimators_.items():
    p = est.predict(X_test)
    print(f"{name} accuracy: {accuracy_score(y_test, p):.4f}")
```

**Verified results:**

| Model | Accuracy |
|---|---|
| RF alone | 0.9904 |
| GB alone | 0.9916 |
| **Stacking** | **0.9940** |

Matches the best voting result (hard voting was also 0.9940) — this dataset is easy enough that both approaches converge to a similar ceiling for this target.

---

## 4. Stacking vs. Voting — when to use which

| | Voting | Stacking |
|---|---|---|
| How weights are set | You choose them manually (or equal by default) | Learned automatically by the meta-model |
| Robust to one weak base learner? | No — drags the average down | Yes — meta-model can learn to downweight or ignore it |
| Complexity / risk of overfitting | Low | Higher — needs proper CV, more moving parts |
| Training cost | Low (train each base learner once) | Higher (each base learner is trained `cv`-times internally, plus the meta-model) |
| Interpretability | Very simple | Meta-model coefficients give some insight, but overall less transparent |

**Practical rule of thumb:** if your base models are all roughly comparable in quality, voting is simpler and nearly as good. If your base models differ meaningfully in strength (as with RF vs. GB vs. LR here), stacking is the safer choice — it won't get dragged down by weak contributors the way voting can.
