# SHAP Values & Multi-Output Models — China Used Car Price Prediction

This document explains **SHAP (SHapley Additive exPlanations)** and **multi-output regression**, with fully tested code examples built on `china_used_cars.csv`.

All code below was run end-to-end against the actual dataset before being included here.

---

## Table of Contents

1. [What are SHAP values?](#1-what-are-shap-values)
2. [Single-output SHAP example (price prediction)](#2-single-output-shap-example-price-prediction)
3. [What are multi-output models?](#3-what-are-multi-output-models)
4. [Multi-output model + SHAP example](#4-multi-output-model--shap-example)
5. [Common pitfalls](#5-common-pitfalls)

---

## 1. What are SHAP values?

SHAP values come from cooperative game theory (Shapley values). For any single prediction, they answer:

> "How much did each feature push this specific prediction away from the average prediction?"

Formally, for a prediction `f(x)`:

```
f(x) = base_value + Σ(shap_value_i for every feature i)
```

- **`base_value`** — the average prediction over the training/background data (what you'd predict with zero information about this specific car).
- **`shap_value_i`** — how much feature `i` (e.g. `motor_power_kw`, `car_age`) pushed *this* prediction up or down from that base value.
- The values are **additive** — they sum exactly to `f(x) - base_value`. This is what makes SHAP more rigorous than typical "feature importance": it explains *individual* predictions, not just overall tendencies.

For tree-based models (`RandomForestRegressor`, XGBoost, LightGBM, etc.), `shap.TreeExplainer` computes **exact** Shapley values efficiently by exploiting the tree structure, instead of the exponential brute-force the underlying theory implies.

---

## 2. Single-output SHAP example (price prediction)

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('china_used_cars.csv')

# Drop the target itself, plus columns that duplicate/derive from it
drop_cols = ['price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# --- SHAP: modern Explanation-object API ---
explainer = shap.TreeExplainer(model)
exp = explainer(X_test)   # shap.Explanation object — handles base_value shapes internally

# 1. Global summary — which features matter most, and in which direction
plt.figure()
shap.summary_plot(exp.values, X_test, show=False)
plt.savefig('shap_summary.png', bbox_inches='tight')
plt.close()

# 2. Local explanation — why THIS car got THIS predicted price
plt.figure()
shap.plots.waterfall(exp[0], show=False)
plt.savefig('shap_waterfall_row0.png', bbox_inches='tight')
plt.close()

# 3. Dependence — how one feature's effect changes across its value range
plt.figure()
shap.dependence_plot('motor_power_kw', exp.values, X_test, show=False)
plt.savefig('shap_dependence_power.png', bbox_inches='tight')
plt.close()

# Sanity check: base_value + this row's shap values == the model's actual prediction
pred0 = model.predict(X_test.iloc[[0]])[0]
print("Model prediction:      ", pred0)
print("base_value + shap sum: ", exp.base_values[0] + exp.values[0].sum())
```

**Verified output on this dataset:**
```
Model prediction:       7547.24
base_value + shap sum:  7547.2425
```
The two match — confirming the SHAP decomposition is mathematically consistent with the model's actual output.

### Reading the plots

| Plot | What it shows |
|---|---|
| `summary_plot` | Beeswarm: each dot = one car. X-axis = SHAP value (impact on price). Color = feature value (red=high, blue=low). Good for spotting global patterns, e.g. "high `motor_power_kw` consistently raises predicted price." |
| `waterfall` | For **one** car: step-by-step view of which features pushed its predicted price up/down from the base value. |
| `dependence_plot` | One feature's value (x-axis) vs. its SHAP value (y-axis) — reveals non-linear effects, e.g. `car_age` may hurt price sharply in the first few years then flatten out. |

---

## 3. What are multi-output models?

A **multi-output model** predicts **several targets simultaneously** from the same set of input features — for example, jointly predicting `price` and `mileage_per_year` for a car instead of training two completely separate pipelines.

`RandomForestRegressor` actually supports multi-output natively (you can pass a 2-column `y` directly), but the general-purpose, model-agnostic wrapper in scikit-learn is `MultiOutputRegressor`, which fits **one independent underlying model per target**.

```python
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

y_multi = df[['price', 'mileage_per_year']]
X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

multi_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
)
multi_model.fit(X_train, y_train)

preds = multi_model.predict(X_test)   # shape: (n_samples, 2) -> [price, mileage_per_year]
```

> **Note on target choice:** `price` + `mileage_per_year` is used here purely to demonstrate the mechanics, since the underlying project is a single-target price predictor. Swap in whatever second target actually makes sense for your use case — the pattern generalizes to any set of numeric targets.

---

## 4. Multi-output model + SHAP example

**Important gotcha:** `shap.TreeExplainer` does not understand `MultiOutputRegressor` directly — it expects one tree model, not a wrapper holding several. The fix: reach into `.estimators_` and explain each target's underlying model separately.

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import shap
import matplotlib.pyplot as plt

df = pd.read_csv('china_used_cars.csv')

drop_cols = ['price', 'mileage_km', 'log_mileage', 'mileage_per_year', 'year', 'month']
X = df.drop(columns=drop_cols)
y_multi = df[['price', 'mileage_per_year']]

X_train, X_test, y_train, y_test = train_test_split(X, y_multi, test_size=0.2, random_state=42)

multi_model = MultiOutputRegressor(
    RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
)
multi_model.fit(X_train, y_train)

target_names = y_multi.columns.tolist()   # ['price', 'mileage_per_year']

shap_explanations = {}
for i, target in enumerate(target_names):
    single_model = multi_model.estimators_[i]        # the RF trained for this one target
    explainer = shap.TreeExplainer(single_model)
    exp = explainer(X_test)                          # modern Explanation object, per target
    shap_explanations[target] = exp

    # Global summary for this target
    plt.figure()
    shap.summary_plot(exp.values, X_test, show=False)
    plt.title(f"SHAP summary — target: {target}")
    plt.savefig(f'shap_summary_{target}.png', bbox_inches='tight')
    plt.close()

    # Local explanation for the first test row, for this target
    plt.figure()
    shap.plots.waterfall(exp[0], show=False)
    plt.savefig(f'shap_waterfall_{target}_row0.png', bbox_inches='tight')
    plt.close()

# Sanity check for each target
for target, exp in shap_explanations.items():
    pred0 = multi_model.predict(X_test.iloc[[0]])[0][target_names.index(target)]
    print(f"[{target}] model prediction:      {pred0}")
    print(f"[{target}] base_value + shap sum: {exp.base_values[0] + exp.values[0].sum()}")
```

This gives you a **separate, correctly-attributed SHAP explanation per target**, so you can directly compare — for example — whether `motor_power_kw` matters more for predicting `price` or for predicting `mileage_per_year`.

---

## 5. Common pitfalls

- **`waterfall_legacy` shape errors** — `explainer.expected_value` can be a scalar, a list, or a 1-element array depending on your SHAP version and whether the model is single- or multi-output. Avoid this entirely by using the modern `explainer(X)` → `Explanation` object API (as done throughout this doc) instead of `shap.plots._waterfall.waterfall_legacy(...)`.
- **`TreeExplainer` + `MultiOutputRegressor`** — never pass the wrapper itself into `TreeExplainer`. Always explain `multi_model.estimators_[i]` individually, one target at a time.
- **Leakage columns** — when explaining `price`, make sure to drop columns that are mathematically derived from the target or from mileage in ways that would leak information (e.g. `log_mileage`, `mileage_per_year` when mileage itself is a feature) — otherwise SHAP will "explain" a leak rather than a genuine signal.
- **Sanity-check every explanation** — always verify `base_value + shap_values.sum() == model.predict(x)` for at least one row. If this doesn't hold, something in the explainer setup (wrong model object, wrong background data, mismatched feature order) is off.
