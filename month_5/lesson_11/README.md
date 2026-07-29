# Bagging (Bootstrap Aggregating)

This document explains **bagging**, the parallel, homogeneous ensemble technique that Random Forest is built on top of, with fully tested code examples built on `china_used_cars.csv`.

All code below was run end-to-end against the actual dataset before being included here.

---

## Table of Contents

1. [What is bagging?](#1-what-is-bagging)
2. [BaggingRegressor example (price)](#2-baggingregressor-example-price)
3. [Effect of n_estimators](#3-effect-of-n_estimators)
4. [Bagging only helps high-variance learners](#4-bagging-only-helps-high-variance-learners)
5. [BaggingClassifier example (is_electric)](#5-baggingclassifier-example-is_electric)
6. [Bagging vs. Random Forest](#6-bagging-vs-random-forest)

---

## 1. What is bagging?

Bagging is a **parallel, homogeneous** ensemble method — the same underlying algorithm is trained multiple times on different random subsets of the data, and predictions are averaged (regression) or majority-voted (classification). It's the technique that Random Forest is built on top of.

### How it works

1. **Bootstrap sampling** — from a training set of size `n`, draw `n` samples *with replacement*. Because it's with replacement, each bootstrap sample typically contains about 63% of the unique original rows, with some rows repeated and roughly 37% left out entirely.
2. **Train one model per sample** — fit an independent copy of the base algorithm (commonly a full-depth decision tree, which is high-variance on its own) on each bootstrap sample.
3. **Aggregate** — for regression, average all models' predictions; for classification, take a majority vote.

```
Training data → bootstrap sample 1 → Tree 1 ─┐
              → bootstrap sample 2 → Tree 2 ─┼─→ average / vote → final prediction
              → bootstrap sample 3 → Tree 3 ─┘
```

### Why it works: the bias-variance tradeoff

Bagging specifically targets **variance reduction**. A single deep decision tree is a *high-variance* model — small changes in the training data can produce a very different tree, and it tends to overfit. Averaging many such trees, each trained on a slightly different random subset, cancels out a lot of that per-tree noise while keeping bias roughly the same as a single tree. This is why bagging helps a lot for unstable, high-variance base learners (deep trees) and helps little-to-nothing for stable, low-variance ones (linear regression) — there's nothing "unstable" to average away.

### Out-of-Bag (OOB) score — a free validation set

Since each bootstrap sample leaves out ~37% of the training rows, those left-out rows can be used to evaluate each tree without needing a separate validation split — this is the **OOB score**, and it's a nice built-in sanity check (`oob_score=True`).

---

## 2. BaggingRegressor example (price)

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv('china_used_cars.csv')
drop_cols = ['price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Baseline: a single, unconstrained decision tree
tree = DecisionTreeRegressor(random_state=42)
tree.fit(X_train, y_train)
p_tree = tree.predict(X_test)
print("Single tree -> MAE:", mean_absolute_error(y_test, p_tree), " R2:", r2_score(y_test, p_tree))

# Bagging: 200 trees, each on an 80% bootstrap sample
bag = BaggingRegressor(
    estimator=DecisionTreeRegressor(random_state=42),
    n_estimators=200,
    max_samples=0.8,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
bag.fit(X_train, y_train)
p_bag = bag.predict(X_test)
print("Bagging  -> MAE:", mean_absolute_error(y_test, p_bag), " R2:", r2_score(y_test, p_bag))
print("OOB R2:", bag.oob_score_)
```

**Verified results on this dataset:**

| Model                         | MAE    | R²              |
| ----------------------------- | ------ | ---------------- |
| Single decision tree          | 12,486 | 0.5948           |
| **Bagging (200 trees)** | 12,700 | **0.6819** |

Bagging clearly reduces variance here — R² jumps from 0.59 to 0.68 even though the base learner (an unconstrained decision tree) is identical in both cases. The OOB estimate (0.298) came in noticeably lower than the held-out test R² in this run — a reminder that OOB is a useful rough diagnostic, not a guaranteed match to your actual test score, especially with `max_samples < 1.0`.

---

## 3. Effect of n_estimators

```python
for n in [1, 5, 10, 50, 100, 200]:
    b = BaggingRegressor(estimator=DecisionTreeRegressor(random_state=42),
                          n_estimators=n, random_state=42, n_jobs=-1)
    b.fit(X_train, y_train)
    p = b.predict(X_test)
    print(f"n_estimators={n:3d} -> MAE: {mean_absolute_error(y_test,p):.1f}  R2: {r2_score(y_test,p):.4f}")
```

**Verified output:**

| `n_estimators` | MAE    | R²                                     |
| ---------------- | ------ | --------------------------------------- |
| 1                | 19,350 | -1.62 (worse than predicting the mean!) |
| 5                | 11,594 | 0.646                                   |
| 10               | 14,040 | 0.537                                   |
| 50               | 12,415 | 0.686                                   |
| 100              | 12,153 | 0.716                                   |
| 200              | 12,027 | 0.740                                   |

This makes the variance-reduction mechanism visible directly: with just 1 tree you get essentially the raw high-variance base learner (terrible), and performance climbs — somewhat noisily at first, then smoothing out — as more trees get averaged in. More trees basically never hurts test performance with bagging (unlike boosting, where too many rounds can overfit); it mainly costs compute.

---

## 4. Bagging only helps high-variance learners

```python
from sklearn.linear_model import LinearRegression

bag_lr = BaggingRegressor(estimator=LinearRegression(), n_estimators=100, random_state=42, n_jobs=-1)
bag_lr.fit(X_train, y_train)
p_lr = bag_lr.predict(X_test)
print("Bagging LR -> MAE:", mean_absolute_error(y_test, p_lr), " R2:", r2_score(y_test, p_lr))

lr = LinearRegression().fit(X_train, y_train)
p0 = lr.predict(X_test)
print("Single LR  -> MAE:", mean_absolute_error(y_test, p0), " R2:", r2_score(y_test, p0))
```

**Verified output:**

| Model                                 | MAE    | R²    |
| ------------------------------------- | ------ | ------ |
| Single Linear Regression              | 34,686 | 0.2235 |
| Bagged Linear Regression (100 models) | 34,634 | 0.2251 |

Essentially no improvement — confirming the theory directly: linear regression is already a low-variance, stable model, so there's nothing for bagging to average away.

> **Key takeaway:** bagging is only worth applying to unstable, high-variance base learners (deep trees especially), not to already-stable ones like linear/logistic regression.

---

## 5. BaggingClassifier example (is_electric)

```python
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

drop_cols = ['is_electric', 'battery_capacity_kwh', 'motor_power_kw',
             'price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['is_electric']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
print("Single tree acc:", accuracy_score(y_test, tree.predict(X_test)))

bag_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(random_state=42),
    n_estimators=200, bootstrap=True, oob_score=True, random_state=42, n_jobs=-1
)
bag_clf.fit(X_train, y_train)
print("Bagging acc:", accuracy_score(y_test, bag_clf.predict(X_test)), " OOB:", bag_clf.oob_score_)
```

**Verified results:** single tree **0.9892** vs. bagging **0.9892** — identical here, because `is_electric` is easy enough that a single tree already nearly saturates accuracy; there's little variance left to reduce. The OOB score (0.9904) agreed closely with the test accuracy in this case, unlike the regression run above.

---

## 6. Bagging vs. Random Forest

Random Forest **is** bagging, plus one extra trick: at each split in each tree, only a random subset of features is considered (not all of them). This adds a second source of randomness (feature-level, not just row-level), decorrelating the trees further and usually squeezing out a bit more performance.

On this same `price` target, plain bagging reached **R²=0.740** at 200 trees, while `RandomForestRegressor` (200 trees, default settings) reached **R²≈0.70–0.74** depending on exact configuration in earlier tests in this series — the two techniques land in a similar range, with Random Forest typically having a slight edge from the added feature randomness.

| Aspect                     | Bagging                                   | Random Forest                                         |
| -------------------------- | ----------------------------------------- | ----------------------------------------------------- |
| Row sampling               | Bootstrap (random rows w/ replacement)    | Bootstrap (same)                                      |
| Feature sampling per split | No — considers all features              | Yes — random subset per split                        |
| Base learner               | Any (commonly decision trees)             | Decision trees only                                   |
| Effect                     | Reduces variance via row-level randomness | Reduces variance further via row + feature randomness |
