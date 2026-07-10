# Overfitting and Underfitting in Machine Learning

A clear, practical guide to understanding, detecting, and fixing the two most common model performance problems — with code examples using sklearn.

---

## Table of Contents

- [The core idea](#the-core-idea)
- [Underfitting](#underfitting)
- [Overfitting](#overfitting)
- [The sweet spot](#the-sweet-spot)
- [Detecting both problems](#detecting-both-problems)
- [How to fix underfitting](#how-to-fix-underfitting)
- [How to fix overfitting](#how-to-fix-overfitting)
- [Bias-variance tradeoff](#bias-variance-tradeoff)
- [Quick reference](#quick-reference)

---

## The core idea

As model complexity increases, two things happen:

- **Training error always falls** — a more complex model can always fit the training data better
- **Validation error forms a U-shape** — it starts high (underfitting), reaches a minimum (sweet spot), then rises again (overfitting)

```
Error
  │
  │  ╲ val error                         ╱
  │   ╲                                 ╱
  │    ╲           sweet spot          ╱
  │     ╲               │             ╱
  │      ╲______________│____________╱
  │       ──────────────────────────── train error
  │
  └──────────────────────────────────────→ Complexity
      underfit    optimal    overfit
```

Your goal is to find and stay near the minimum of the validation error curve.

---

## Underfitting

The model is **too simple** to capture the real pattern. It performs badly on **both** training data and new data — it hasn't even learned what it was shown.

### Signs of underfitting

| Signal                    | What you see                                |
| ------------------------- | ------------------------------------------- |
| Training accuracy         | Low                                         |
| Validation accuracy       | Low                                         |
| Gap between train and val | Small (both are equally bad)                |
| Predictions               | Nearly constant — model ignores most input |

### Code example

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Extreme regularization forces coefficients close to zero
# max_iter=10 is too few iterations to converge
model = LogisticRegression(C=0.00001, max_iter=10)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))   # ~0.64
print("Test  accuracy:", model.score(X_test,  y_test))    # ~0.64
# Both scores are similar and both are poor → UNDERFIT
```

### Why it happens

- Model is too simple for the problem (e.g. linear model on a curved relationship)
- Too much regularization (very high `alpha` or very low `C`)
- Too few training iterations (`max_iter` too small)
- Important features removed during preprocessing
- Too few features — the model doesn't have enough information

---

## Overfitting

The model is **too complex** and memorizes the training data — including its noise and random quirks. It performs very well on training data but fails on new, unseen data.

### Signs of overfitting

| Signal                    | What you see                            |
| ------------------------- | --------------------------------------- |
| Training accuracy         | Very high (sometimes 100%)              |
| Validation accuracy       | Noticeably lower                        |
| Gap between train and val | Large                                   |
| Predictions               | Great on training set, poor on new data |

### Code example

```python
from sklearn.tree import DecisionTreeClassifier

# No depth limit — tree grows until every training sample is pure
model = DecisionTreeClassifier(max_depth=None, min_samples_leaf=1)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))   # ~1.00 (perfect!)
print("Test  accuracy:", model.score(X_test,  y_test))    # ~0.82 (much worse)

# Large gap between train and test = OVERFITTING
# The model memorized the training data, including its noise
```

### Why it happens

- Model is too complex (very deep tree, many neurons, high-degree polynomial)
- Not enough training data
- Too little regularization
- Training for too many epochs (neural networks)
- Too many features, including irrelevant or noisy ones

---

## The sweet spot

Neither underfit nor overfit. Both training error and validation error are low, and the gap between them is small.

### Finding the sweet spot by comparing depths

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

results = []
for depth in [2, 4, 6, 8, 10, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    cv_score    = cross_val_score(model, X_train, y_train, cv=5).mean()
    gap         = train_score - cv_score
    results.append((depth, train_score, cv_score, gap))
    print(f"max_depth={str(depth):5s} | train={train_score:.3f} | cv={cv_score:.3f} | gap={gap:.3f}")

# depth=None → train=1.000, cv=0.820, gap=0.180  ← OVERFIT
# depth=6    → train=0.920, cv=0.905, gap=0.015  ← SWEET SPOT
# depth=2    → train=0.740, cv=0.738, gap=0.002  ← UNDERFIT
```

The depth where `cv_score` peaks and `gap` is small is your sweet spot.

---

## Detecting both problems

### Learning curves — the best diagnostic tool

Plot training and validation scores against the size of the training set. The shape of the curves tells you exactly what's wrong.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

def plot_learning_curve(model, X, y, title):
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    val_mean   = val_scores.mean(axis=1)

    plt.figure(figsize=(8, 5))
    plt.plot(train_sizes, train_mean, label='Training score',   color='steelblue')
    plt.plot(train_sizes, val_mean,   label='Validation score', color='tomato')
    plt.fill_between(train_sizes, train_mean - train_scores.std(axis=1),
                                  train_mean + train_scores.std(axis=1), alpha=0.1, color='steelblue')
    plt.fill_between(train_sizes, val_mean - val_scores.std(axis=1),
                                  val_mean + val_scores.std(axis=1), alpha=0.1, color='tomato')
    plt.xlabel('Training set size')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Underfit model
plot_learning_curve(
    LogisticRegression(C=0.00001, max_iter=10),
    X_train, y_train, 'Underfit: both curves low and flat'
)

# Overfit model
plot_learning_curve(
    DecisionTreeClassifier(max_depth=None),
    X_train, y_train, 'Overfit: large gap between curves'
)

# Good model
plot_learning_curve(
    DecisionTreeClassifier(max_depth=6, min_samples_leaf=10),
    X_train, y_train, 'Good fit: curves close and high'
)
```

**Reading the chart:**

| Pattern                                     | Diagnosis                                    |
| ------------------------------------------- | -------------------------------------------- |
| Both curves are low and flat                | Underfitting — model too simple             |
| Train is high, val is much lower, large gap | Overfitting — model memorized training data |
| Both curves are high and close together     | Good generalization — sweet spot            |
| Val curve is still rising                   | More training data would help                |

### Validation score vs training score

```python
from sklearn.model_selection import cross_val_score

model = DecisionTreeClassifier(max_depth=10, random_state=42)
model.fit(X_train, y_train)

train_score = model.score(X_train, y_train)
cv_score    = cross_val_score(model, X_train, y_train, cv=5).mean()
gap         = train_score - cv_score

print(f"Train score : {train_score:.3f}")
print(f"CV score    : {cv_score:.3f}")
print(f"Gap         : {gap:.3f}")

if gap > 0.10:
    print("→ Likely OVERFITTING")
elif train_score < 0.80:
    print("→ Likely UNDERFITTING")
else:
    print("→ Looking good")
```

---

## How to fix underfitting

```python
# 1. Use a more powerful model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

model = RandomForestClassifier(n_estimators=200, random_state=42)
# or
model = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1)

# 2. Reduce regularization (increase C for LR, decrease alpha for Ridge)
from sklearn.linear_model import LogisticRegression, Ridge

model = LogisticRegression(C=10.0)   # higher C = less penalty
model = Ridge(alpha=0.01)            # lower alpha = less penalty

# 3. Add more / better features (feature engineering)
import pandas as pd
df['power_ratio'] = df['p1'] / (df['p2'] + 1e-9)   # domain-informed feature
df['tau_sum']     = df[['tau1','tau2','tau3','tau4']].sum(axis=1)

# 4. Train for more iterations
model = LogisticRegression(max_iter=1000)   # default is 100, often too low

# 5. Try polynomial features for linear models
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lr',   LogisticRegression(max_iter=1000)),
])
```

---

## How to fix overfitting

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge

# 1. Limit model complexity
model = DecisionTreeClassifier(
    max_depth=6,          # limit depth
    min_samples_leaf=10,  # no tiny leaf nodes
    min_samples_split=20  # require more samples to split
)

# 2. Add regularization
model = LogisticRegression(C=0.1)    # smaller C = more L2 regularization
model = Ridge(alpha=10.0)            # larger alpha = more L2 regularization

# 3. Use ensemble methods (Random Forest averages many trees → less overfit)
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=4,
    max_features='sqrt',   # use subset of features per split
    random_state=42
)

# 4. Drop irrelevant features
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=8)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected  = selector.transform(X_test)

# 5. Get more training data (most reliable fix of all)
# More data = less chance the model memorizes noise

# 6. Cross-validate to measure the real generalization score
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV mean: {scores.mean():.3f} ± {scores.std():.3f}")
# High std across folds also signals overfitting
```

---

## Bias-variance tradeoff

Overfitting and underfitting are two sides of a deeper concept called the bias-variance tradeoff:

```
Total Error = Bias² + Variance + Irreducible Noise
```

| Term                        | Meaning                                                              | High when                          |
| --------------------------- | -------------------------------------------------------------------- | ---------------------------------- |
| **Bias**              | Error from wrong assumptions — model consistently misses the target | Model is too simple (underfitting) |
| **Variance**          | Error from sensitivity to small changes in training data             | Model is too complex (overfitting) |
| **Irreducible noise** | Random error in the data itself                                      | Always present, can't be reduced   |

```
Bias vs Variance

                        Total Error
                       ╱            ╲
            Variance  ╱              ╲  Bias²
                     ╱                ╲___
           ─────────╱─────────────────────── Complexity
          underfit      sweet spot      overfit
```

Making a model more complex reduces bias but increases variance. The sweet spot minimizes their sum.

|             | Bias | Variance | Result              |
| ----------- | ---- | -------- | ------------------- |
| Too simple  | High | Low      | Underfitting        |
| Just right  | Low  | Low      | Good generalization |
| Too complex | Low  | High     | Overfitting         |

---

## Quick reference

```
Problem       │ Train error │ Val error │ Gap   │ Fix
──────────────┼─────────────┼───────────┼───────┼───────────────────────────────
Underfitting  │ High        │ High      │ Small │ More complex model, less regularization,
              │             │           │       │ more/better features, more iterations
──────────────┼─────────────┼───────────┼───────┼───────────────────────────────
Overfitting   │ Low         │ High      │ Large │ Simpler model, more regularization,
              │             │           │       │ more data, feature selection, ensemble
──────────────┼─────────────┼───────────┼───────┼───────────────────────────────
Good fit      │ Low         │ Low       │ Small │ You're done — tune further with
              │             │           │       │ hyperparameter search if needed
```

**Diagnostic steps (in order):**

1. Train a baseline model
2. Compare `train_score` vs `cv_score` — measure the gap
3. Plot learning curves to visualize the diagnosis
4. Apply the right fix based on which problem you see
5. Repeat until train and validation scores are both high with a small gap

---

Part of the ML Engineering self-study roadmap. See the full course repo at [github.com/otabekrazzoqov/Machine-Learning-full-course-main](https://github.com/otabekrazzoqov/Machine-Learning-full-course-main)
