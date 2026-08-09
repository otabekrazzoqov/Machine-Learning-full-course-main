# Lesson 01 — Introduction to Machine Learning

## Table of Contents

1. [What is Machine Learning?](#1-what-is-machine-learning)
2. [Why use Machine Learning?](#2-why-use-machine-learning)
3. [Types of Machine Learning](#3-types-of-machine-learning)
4. [Supervised Machine Learning — in depth](#4-supervised-machine-learning--in-depth)
5. [Supervised learning: simple code example](#5-supervised-learning-simple-code-example)
6. [Key terms cheat sheet](#6-key-terms-cheat-sheet)

---

## 1. What is Machine Learning?

**Machine Learning (ML)** is a branch of Artificial Intelligence where a system learns patterns from data and improves its performance on a task without being explicitly programmed with fixed rules for every case.

The classic distinction is:

- **Traditional programming**: `Rules + Data → Output` (a human writes the logic).
- **Machine Learning**: `Data + Output → Rules` (the algorithm discovers the logic itself, by finding patterns in examples).

**Example:** instead of hand-coding a rule like "if the car is under 2 years old and has less than 20,000 km, price it above $15,000," you show a model thousands of past car listings (with their actual prices), and it learns the underlying pricing pattern on its own.

A working definition worth remembering (Tom Mitchell's, one of the most cited in ML textbooks):

> A computer program is said to *learn* from experience `E`, with respect to some task `T` and performance measure `P`, if its performance on `T`, as measured by `P`, improves with experience `E`.

---

## 2. Why use Machine Learning?

ML is useful when:

- **The rules are too complex to hand-write** — e.g. recognizing handwriting, understanding speech, detecting spam.
- **The rules change over time** — e.g. fraud patterns, market prices, spam techniques.
- **You need to find patterns humans wouldn't notice** — e.g. subtle correlations across thousands of data points.
- **You need to make predictions or decisions at scale**, faster than a human could evaluate each case manually.

---

## 3. Types of Machine Learning

There are four broad categories:

| Type                               | What it learns from                                                               | Goal                                                                                                  | Examples                                                  |
| ---------------------------------- | --------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Supervised Learning**      | Labeled data (input**and** correct output given)                            | Predict the output for new, unseen inputs                                                             | Price prediction, spam detection, disease diagnosis       |
| **Unsupervised Learning**    | Unlabeled data (only inputs, no correct answers given)                            | Discover hidden structure or groupings in the data                                                    | Customer segmentation, anomaly detection, topic discovery |
| **Semi-Supervised Learning** | A small amount of labeled data + a large amount of unlabeled data                 | Combine both to learn more efficiently than pure supervised learning when labels are expensive to get | Medical imaging (few labeled scans, many unlabeled ones)  |
| **Reinforcement Learning**   | An agent interacting with an environment, receiving rewards/penalties for actions | Learn a strategy (policy) that maximizes cumulative reward over time                                  | Game-playing agents, robotics, self-driving car control   |

### Quick way to tell them apart

- If your dataset has a clear "answer column" you're trying to predict → **supervised**.
- If your dataset has no answer column, and you're just trying to find structure/groups → **unsupervised**.
- If you have a *little* labeled data and a lot more unlabeled data → **semi-supervised**.
- If there's no fixed dataset at all, but an agent that acts, observes outcomes, and gets rewarded/punished → **reinforcement learning**.

---

## 4. Supervised Machine Learning — in depth

Supervised learning is the most common and most widely used type of ML in practice. It works with **labeled data**: every training example consists of an input (features, `X`) and a known correct output (label/target, `y`).

```
Input Features (X)              Label (y)
------------------------------- --------
mileage=13000, age=1, power=310 -> price=24288
mileage=10000, age=0, power=0   -> price=8787
mileage=90000, age=1, power=0   -> price=2914
```

The model's job during training is to learn a function `f` such that `f(X) ≈ y`, then use that learned function to predict `y` for brand-new inputs it has never seen.

Supervised learning splits into two main sub-types, based on what kind of thing you're predicting:

### 4.1 Regression

Predicting a **continuous numeric value**.

- Example: predicting a used car's `price` (any number, e.g. $2,914, $8,787, $24,288...).
- Other examples: predicting house prices, temperature, stock prices, someone's age.
- Common algorithms: Linear Regression, Decision Tree Regressor, Random Forest Regressor, Gradient Boosting Regressor.
- Common evaluation metrics: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE, R² (R-squared).

### 4.2 Classification

Predicting a **category / discrete label**.

- Example: predicting whether a car `is_electric` (0 or 1) — this is **binary classification**.
- Example: predicting a car's `fuel_type` out of several possible categories — this is **multi-class classification**.
- Other examples: spam vs. not spam, disease vs. no disease, digit recognition (0–9).
- Common algorithms: Logistic Regression, Decision Tree Classifier, Random Forest Classifier, Gradient Boosting Classifier, KNN, SVM.
- Common evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.

### 4.3 The general supervised learning workflow

1. **Collect labeled data** — inputs (`X`) paired with correct outputs (`y`).
2. **Split the data** — typically into a training set (to learn from) and a test set (to check performance on unseen data), e.g. 80/20.
3. **Choose a model/algorithm** — depends on whether it's regression or classification, and the nature of the data.
4. **Train (fit) the model** — the algorithm adjusts its internal parameters to minimize error on the training data.
5. **Evaluate on the test set** — measure how well the model generalizes to data it has never seen.
6. **Tune and iterate** — adjust hyperparameters, features, or the algorithm itself to improve performance.
7. **Deploy** — use the trained model to make predictions on new, real-world data.

### 4.4 Why the train/test split matters

If you evaluate a model only on the data it was trained on, it can appear to perform very well simply by memorizing that data — this is called **overfitting**, and it doesn't reflect how the model will perform on new data. Splitting off a test set (data the model never sees during training) gives an honest measure of real-world performance.

---

## 5. Supervised learning: simple code example

A minimal, runnable example showing both regression and classification, using scikit-learn:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

df = pd.read_csv('china_used_cars.csv')

# --- Regression example: predict a continuous value (price) ---
X = df.drop(columns=['price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

reg_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
reg_model.fit(X_train, y_train)               # learn from labeled examples
predictions = reg_model.predict(X_test)        # predict on unseen data

print("Regression MAE:", mean_absolute_error(y_test, predictions))
print("Regression R2:", r2_score(y_test, predictions))

# --- Classification example: predict a category (is the car electric?) ---
Xc = df.drop(columns=['is_electric', 'battery_capacity_kwh', 'motor_power_kw',
                       'price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month'])
yc = df['is_electric']

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.2, random_state=42, stratify=yc)

clf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
clf_model.fit(Xc_train, yc_train)
clf_predictions = clf_model.predict(Xc_test)

print("Classification Accuracy:", accuracy_score(yc_test, clf_predictions))
```

Both blocks follow the exact same supervised learning pattern: **labeled data in → `.fit()` to learn → `.predict()` on new data → measure how correct it was.** The only difference is the type of label (`price` is continuous → regression; `is_electric` is a category → classification).

---

## 6. Key terms cheat sheet

| Term                         | Meaning                                                                                |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| **Feature(s) / X**     | The input variables used to make a prediction                                          |
| **Label / Target / y** | The correct output the model is trying to predict                                      |
| **Training set**       | Data used to teach the model                                                           |
| **Test set**           | Data used to evaluate the model, unseen during training                                |
| **Model**              | The learned function that maps inputs to outputs                                       |
| **Overfitting**        | Model memorizes training data but fails to generalize to new data                      |
| **Underfitting**       | Model is too simple to capture the real pattern, performs poorly even on training data |
| **Regression**         | Supervised learning where the target is a continuous number                            |
| **Classification**     | Supervised learning where the target is a category/class                               |

---
