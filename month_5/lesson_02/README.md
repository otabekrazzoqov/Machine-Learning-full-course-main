# Class Imbalance Problem: Oversampling & Undersampling

A reference guide to detecting, understanding, and fixing class imbalance in classification problems — with a worked example on the Smart Grid Stability dataset.

---

## Table of Contents

- [What is class imbalance?](#what-is-class-imbalance)
- [Why it breaks models](#why-it-breaks-models)
- [The golden rule: resample after splitting](#the-golden-rule-resample-after-splitting)
- [Oversampling techniques](#oversampling-techniques)
  - [Random Oversampling](#1-random-oversampling)
  - [SMOTE](#2-smote-synthetic-minority-oversampling-technique)
  - [ADASYN](#3-adasyn-adaptive-synthetic-sampling)
- [Undersampling techniques](#undersampling-techniques)
  - [Random Undersampling](#1-random-undersampling)
  - [ClusterCentroids](#2-clustercentroids)
  - [NearMiss](#3-nearmiss)
- [Worked example: Smart Grid Stability dataset](#worked-example-smart-grid-stability-dataset)
- [Decision framework](#decision-framework)
- [A free alternative: class_weight='balanced'](#a-free-alternative-class_weightbalanced)
- [Quick reference](#quick-reference)

---

## What is class imbalance?

Class imbalance happens when one class (the **majority class**) vastly outnumbers another (the **minority class**) in your target variable — sometimes 100:1, sometimes 1000:1.

It shows up almost everywhere in real-world data:

| Use case | Typical minority share | What's rare |
|---|---|---|
| Fraud detection | ~0.1% | fraudulent transactions |
| Disease screening | ~2-5% | patients with the condition |
| Spam filtering | ~5-15% | spam emails in an inbox |
| Layoff risk prediction | ~3-8% | employees actually laid off |

## Why it breaks models

A model trained on imbalanced data can get **high accuracy while being completely useless**.

> If 99% of transactions are legit, a model that *always* predicts "legit" — without looking at the data at all — gets **99% accuracy** and **0% fraud caught**.

This happens because the loss function barely notices the minority class — getting every minority sample wrong costs almost nothing on average. The result:

- Minority classes get **ignored** — the model has no incentive to learn their patterns
- **Poor minority-class performance** — low recall/precision exactly where it matters most
- **Biased, overfit behavior** — a high overall score hides a model that learned almost nothing useful

**This is why accuracy is the wrong metric for imbalanced problems.** Use precision, recall, F1-score, or per-class ROC-AUC instead, and always check a confusion matrix.

## The golden rule: resample after splitting

**Always split train/test first. Resample only the training set.**

If you resample before splitting, duplicated or synthetic minority samples can leak into your test set, giving you a falsely optimistic evaluation. Your test set must reflect the real-world class distribution.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Resample X_train, y_train only. Leave X_test, y_test untouched.
```

---

## Oversampling techniques

Oversampling adds data to the minority class until it matches (or approaches) the majority class size.

### 1. Random Oversampling

Duplicates random rows from the minority class until classes are balanced.

```python
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)
```

| Pros | Cons |
|---|---|
| Simple and fast to apply | Exact duplicates risk overfitting — model can memorize repeated rows |
| No assumptions about data structure | No new information added |

### 2. SMOTE (Synthetic Minority Oversampling Technique)

Instead of duplicating, SMOTE creates **new synthetic points**. For each minority sample, it finds its *k*-nearest minority neighbors and generates a new point along the line connecting them.

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
```

| Pros | Cons |
|---|---|
| Adds variety instead of exact duplicates | Can create unrealistic points if minority data is spread out or near majority clusters |
| Lower overfitting risk than random oversampling | Numeric features only (use `SMOTENC` for categorical features) |

### 3. ADASYN (Adaptive Synthetic Sampling)

A refinement of SMOTE. Generates **more** synthetic samples in regions where the minority class is hardest to learn (near the decision boundary — surrounded by many majority neighbors), and **fewer** where it's already easy.

```python
from imblearn.over_sampling import ADASYN

adasyn = ADASYN(random_state=42, n_neighbors=5)
X_train_res, y_train_res = adasyn.fit_resample(X_train, y_train)
```

| Pros | Cons |
|---|---|
| Focuses effort where the model actually struggles | More sensitive to noise — an outlier surrounded by majority points gets oversampled too |
| Often improves classification right at the decision boundary | |

---

## Undersampling techniques

Undersampling removes data from the majority class until it matches (or approaches) the minority class size.

### 1. Random Undersampling

Drops random rows from the majority class until classes are balanced.

```python
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_train_res, y_train_res = rus.fit_resample(X_train, y_train)
```

| Pros | Cons |
|---|---|
| Fast — smaller dataset trains quicker | Throws away potentially useful majority-class information |
| Simple and effective | Can hurt performance if the majority class wasn't actually redundant |

### 2. ClusterCentroids

Clusters the majority class with K-Means into *k* clusters (where *k* is the target size), then replaces each cluster with its **centroid** — a synthetic representative point — instead of picking real rows.

```python
from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=42)
X_train_res, y_train_res = cc.fit_resample(X_train, y_train)
```

| Pros | Cons |
|---|---|
| Retains the overall shape/distribution of the majority class better than random dropping | Centroids are synthetic — real data points are lost entirely |
| | Computationally heavier (runs K-Means internally) |

### 3. NearMiss

A more surgical undersampler. Keeps the majority samples **closest to the minority class** (near the decision boundary) and removes the "easy," far-away majority points. Three versions (`version=1,2,3`) differ in exactly how they select which points to keep.

```python
from imblearn.under_sampling import NearMiss

nm = NearMiss(version=1, n_neighbors=3)
X_train_res, y_train_res = nm.fit_resample(X_train, y_train)
```

| Pros | Cons |
|---|---|
| Keeps the majority samples that actually define the decision boundary | Sensitive to noisy points right at the boundary |
| Often the best balance among undersampling methods in practice | Slower than random undersampling |

---

## Worked example: Smart Grid Stability dataset

The [Smart Grid Stability dataset](https://www.kaggle.com/datasets/pcbreviglieri/smart-grid-stability) (60,000 rows, binary target `stabf`: stable/unstable) is naturally close to balanced (~64/36), so to demonstrate these techniques meaningfully, a severe **96% / 4%** imbalance was manufactured by downsampling the `stable` class to 1,500 rows.

**Setup:** Logistic Regression, same untouched test set for every method.

### Results — minority-class (`stable`) F1 score

| Method | F1 (stable) | Recall | Precision |
|---|---|---|---|
| **Baseline (no resampling)** | **0.344** | 0.21 | 0.94 |
| NearMiss | 0.330 | 0.66 | 0.22 |
| SMOTE | 0.231 | ~0.79 | ~0.13 |
| Random Undersampling | 0.226 | ~0.79 | ~0.13 |
| ADASYN | 0.224 | ~0.79 | ~0.13 |
| Random Oversampling | 0.223 | ~0.79 | ~0.13 |
| ClusterCentroids | 0.217 | 0.78 | 0.13 |

### What actually happened

This is a deliberately honest result, and it's a more realistic lesson than "resampling always helps":

- **Baseline** had the *highest* minority F1 — but terrible recall (21%). It rarely flags `stable` cases, but when it does, it's almost always right (94% precision).
- **All oversampling methods** (Random, SMOTE, ADASYN) and most undersampling methods dramatically improved recall (21% → ~79%), but precision collapsed (94% → ~13%). The model now catches almost all real `stable` cases, but also wrongly flags many `unstable` cases as `stable`. Net F1 actually went *down* slightly versus baseline.
- **NearMiss** struck the best balance: better recall than baseline (66% vs. 21%) while keeping precision reasonable (22%) — giving it the best F1 among all the resampling techniques tried.

**The takeaway:** resampling is not a magic fix. It shifts the tradeoff between precision and recall. Whether that tradeoff is "better" depends entirely on which mistake costs more in your specific use case.

---

## Decision framework

| Use case | What matters more | Prefer |
|---|---|---|
| Fraud detection | Catching fraud (recall) | Oversampling / NearMiss |
| Spam filtering | Not blocking real emails (precision) | Baseline or mild techniques |
| Medical screening | Recall — missing a disease is costly | Oversampling |
| Recommendation flagging | Precision — false positives erode trust | Baseline / careful undersampling |

**Always look at precision *and* recall, not just F1 or accuracy alone**, and pick the strategy that matches what mistake is more costly for your problem.

---

## A free alternative: `class_weight='balanced'`

Before reaching for resampling, try this — it costs nothing in terms of data manipulation, training time, or synthetic-data risk. It simply tells the model to penalize minority misclassifications more heavily during training.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)
```

Available on most sklearn classifiers: `LogisticRegression`, `SVC`, `RandomForestClassifier`, and more. In many real projects, this alone gets most of the benefit of resampling with none of the added complexity. **Worth trying first, every time.**

---

## Quick reference

```python
# 1. Split FIRST
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 2. Try the free option first
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(class_weight='balanced')

# 3. If you need resampling, pick based on your priority:
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids, NearMiss

# Few classes, plenty of data, want max recall  -> SMOTE / ADASYN / Random Oversampling
# Want a balanced precision/recall tradeoff      -> NearMiss
# Huge dataset, majority class is redundant      -> Random Undersampling / ClusterCentroids

# 4. Always evaluate on the untouched test set
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

### Other techniques worth knowing

- **Threshold tuning** — instead of the default 0.5 cutoff on `predict_proba`, lower the threshold for the minority class to trade precision for recall.
- **Combined methods** — `SMOTEENN` and `SMOTETomek` in `imblearn` combine SMOTE with a cleanup undersampling step to remove noisy/ambiguous points after oversampling.

---

*Generated as part of an ML engineering self-study roadmap. See the accompanying Jupyter notebook (`imbalance_smartgrid.ipynb`) for the full runnable code behind the worked example.*
