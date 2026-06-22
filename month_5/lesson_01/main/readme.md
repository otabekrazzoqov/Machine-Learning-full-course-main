# Multiclass Classification with One-vs-Rest (OvR) and One-vs-One (OvO)

## Project Overview

This project demonstrates how to perform **multiclass classification** using two popular strategies:

* **One-vs-Rest (OvR)**
* **One-vs-One (OvO)**

Both methods use a **Support Vector Machine (SVM)** as the underlying binary classifier and are implemented using  **Scikit-Learn** .

The project uses the **Smart Grid Stability** dataset and converts it into a multiclass classification problem to compare the performance of OvR and OvO approaches.

---

## Objectives

The main goals of this project are:

* Understand how multiclass classification works.
* Learn the difference between One-vs-Rest and One-vs-One strategies.
* Train SVM-based multiclass classifiers.
* Evaluate model performance using:
  * Accuracy
  * Classification Report
  * Confusion Matrix
  * F1-Score Comparison

---

## Dataset

### Smart Grid Stability Dataset

The original dataset contains:

* Input features describing a smart electrical grid.
* A continuous stability score (`stab`).
* A binary stability label (`stabf`).

### Creating a Multiclass Problem

Since the original dataset is binary, a new multiclass target variable is created.

The continuous stability score (`stab`) is divided into four categories using quartiles:

| Stability Score Group | Label         |
| --------------------- | ------------- |
| Lowest Quartile       | very_unstable |
| Lower-Middle Quartile | unstable      |
| Upper-Middle Quartile | stable        |
| Highest Quartile      | very_stable   |

This creates a balanced multiclass classification problem with four classes.

---

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn

### Main Libraries

```python
pandas
numpy
scikit-learn
```

---

## Project Workflow

### 1. Load Dataset

The dataset is loaded using Pandas.

```python
df = pd.read_csv("smart_grid_stability_augmented.csv")
```

---

### 2. Create Multiclass Labels

The continuous stability score is transformed into four categorical classes.

```python
df["stab_class"] = pd.qcut(
    df["stab"],
    q=4,
    labels=[
        "very_stable",
        "stable",
        "unstable",
        "very_unstable"
    ]
)
```

---

### 3. Prepare Features and Target

Input features are separated from the target variable.

The following columns are removed:

* `stab`
* `stabf`
* `stab_class`

Reason:

* `stab` and `stabf` contain direct information about stability.
* Keeping them would cause  **data leakage** .

```python
X = df.drop(columns=["stab", "stabf", "stab_class"])
y = df["stab_class"]
```

---

### 4. Train-Test Split

The dataset is divided into:

* 70% Training Data
* 30% Testing Data

Stratified sampling is used to preserve class distribution.

```python
train_test_split(
    X,
    y,
    test_size=0.3,
    stratify=y,
    random_state=42
)
```

---

### 5. Feature Scaling

SVMs are sensitive to feature magnitudes.

Standardization is applied using:

```python
StandardScaler()
```

This transforms each feature to:

* Mean = 0
* Standard Deviation = 1

---

## One-vs-Rest (OvR)

### Concept

For a problem with  **4 classes** , OvR trains  **4 binary classifiers** .

Each classifier learns:

| Classifier | Task                        |
| ---------- | --------------------------- |
| Model 1    | very_stable vs all others   |
| Model 2    | stable vs all others        |
| Model 3    | unstable vs all others      |
| Model 4    | very_unstable vs all others |

### Training

```python
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

ovr_model = OneVsRestClassifier(
    SVC(kernel="linear", random_state=42)
)

ovr_model.fit(X_train_scaled, y_train)
```

### Prediction

```python
y_pred_ovr = ovr_model.predict(X_test_scaled)
```

### Number of Models

For:

```
N classes
```

OvR trains:

```
N classifiers
```

For 4 classes:

```
4 classifiers
```

---

## One-vs-One (OvO)

### Concept

OvO trains a classifier for every possible pair of classes.

For 4 classes:

| Pair                         |
| ---------------------------- |
| very_stable vs stable        |
| very_stable vs unstable      |
| very_stable vs very_unstable |
| stable vs unstable           |
| stable vs very_unstable      |
| unstable vs very_unstable    |

Total:

```
N(N−1)/2
```

For 4 classes:

```
4 × 3 / 2 = 6 classifiers
```

### Training

```python
from sklearn.multiclass import OneVsOneClassifier

ovo_model = OneVsOneClassifier(
    SVC(kernel="linear", random_state=42)
)

ovo_model.fit(X_train_scaled, y_train)
```

### Prediction

```python
y_pred_ovo = ovo_model.predict(X_test_scaled)
```

---

## Model Evaluation

Both models are evaluated using:

### Accuracy

Measures overall prediction correctness.

```python
accuracy_score(y_test, predictions)
```

---

### Classification Report

Provides:

* Precision
* Recall
* F1-Score
* Support

```python
classification_report(y_test, predictions)
```

---

### Confusion Matrix

Shows:

* Correct predictions
* Misclassifications
* Class-wise performance

```python
confusion_matrix(y_test, predictions)
```

---

## F1-Score Comparison

A side-by-side comparison is performed to compare:

* OvR F1 scores
* OvO F1 scores

for each class.

This helps identify:

* Which classes are easier to classify.
* Which strategy performs better for overlapping classes.

---

## Why OvR and OvO Perform Differently

### One-vs-Rest

Advantages:

* Simpler
* Faster training
* Fewer models

Disadvantages:

* Each classifier must separate one class from all remaining classes.
* Can struggle when classes overlap.

---

### One-vs-One

Advantages:

* Focuses only on two classes at a time.
* Often achieves better separation between similar classes.
* Can improve performance on difficult class boundaries.

Disadvantages:

* Requires many more classifiers.
* Higher computational cost.

---

## Expected Findings

In many multiclass datasets:

* Extreme classes (such as `very_stable` and `very_unstable`) are easier to distinguish.
* Middle classes (`stable` and `unstable`) are more likely to overlap.
* OvO may outperform OvR on overlapping classes because it learns pairwise boundaries.

---

## Mathematical Comparison

For a dataset with `N` classes:

### One-vs-Rest

Number of classifiers:

```text
N
```

### One-vs-One

Number of classifiers:

```text
N(N−1)/2
```

Example for 4 classes:

| Method | Number of Classifiers |
| ------ | --------------------- |
| OvR    | 4                     |
| OvO    | 6                     |

---

## Key Takeaways

* Multiclass problems can be solved using binary classifiers through OvR and OvO strategies.
* OvR trains one model per class.
* OvO trains one model for every class pair.
* Feature scaling is critical for SVM performance.
* Classification reports and confusion matrices provide detailed evaluation beyond accuracy.
* OvO often performs better when classes have overlapping decision boundaries, while OvR is computationally more efficient.

---

## Author

**Otabek Razzokov**

Master of Science in Computer Science and Engineering

Areas of Interest:

* Machine Learning
* Data Science
* Deep Learning
* Smart Grid Analytics
* Artificial Intelligence

---
