
import numpy as np
import pandas as pd
 
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
 

 
class OrdinalMapper(BaseEstimator, TransformerMixin):

    def __init__(self, mapping: dict):
        self.mapping = mapping

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self
 
    def transform(self, X, y=None):
        X_flat = pd.Series(np.array(X).ravel()).map(self.mapping)
        return X_flat.values.astype(float).reshape(-1, 1)
 
 
 
NUMERIC_FEATURES = [
    "Age",
    "Years_of_Experience",
    "Routine_Task_Percentage",
    "Creativity_Requirement",
    "Human_Interaction_Level",
    "Number_of_AI_Tools_Used",
    "AI_Usage_Hours_Per_Week",
    "Tasks_Automated_Percentage",
    "AI_Training_Hours",
]
 
# Ordinal: meaningful order → map to integers
ORDINAL_CONFIG = {
    "AI_Adoption_Level": {"Low": 0, "Medium": 1, "High": 2},
    "Job_Level":         {"Entry": 0, "Mid": 1, "Senior": 2},
    "Company_Size":      {"Small": 0, "Medium": 1, "Large": 2},
}
 
# Nominal: no natural order → one-hot encode
NOMINAL_FEATURES = ["Education_Level", "Industry", "Job_Role"]
 
TARGET = "Layoff_Risk"
 
 
# Numeric: fill NaN with median → z-score normalise
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
 
# Nominal: fill NaN with most-frequent → one-hot encode
nominal_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
 
 
def make_ordinal_pipeline(mapping: dict) -> Pipeline:
    """Impute → map ordinal strings to ints. One per ordinal column."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("mapper",  OrdinalMapper(mapping)),
    ])
 
 
 
def build_preprocessor() -> ColumnTransformer:
    transformers = [
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("nom", nominal_pipeline, NOMINAL_FEATURES),
    ]
    for col, mapping in ORDINAL_CONFIG.items():
        transformers.append((f"ord_{col}", make_ordinal_pipeline(mapping), [col]))
 
    return ColumnTransformer(transformers=transformers, remainder="drop")
 
 
 
def build_pipeline() -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1,
        )),
    ])
 
 
 
def run(data_path: str = None, verbose: bool = True) -> Pipeline:
    if data_path is None:
        data_path = r"C:\Users\mrcoo\Machine-Learning-full-course-main\month_4\model_building_lesson_10\main\ai-impact-jobs-layoff-risk-dataset.csv"
 
    df = pd.read_csv(data_path)
    X  = df.drop(columns=[TARGET])
    y  = df[TARGET]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
 
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
 
    if verbose:
        print("=" * 55)
        print("  Layoff Risk Prediction — Pipeline Results")
        print("=" * 55)
        print(classification_report(y_test, preds))
 
    return pipeline
 
 
 
def predict_new(pipeline: Pipeline, employees: list[dict]) -> pd.DataFrame:
    """
    Predict Layoff_Risk for raw employee records.
    No manual preprocessing needed — the pipeline handles everything.
    """
    df_new = pd.DataFrame(employees)
    preds  = pipeline.predict(df_new)
    proba  = pipeline.predict_proba(df_new)
 
    result = df_new[["Job_Role", "Industry"]].copy()
    result["Predicted_Risk"] = preds
    for i, cls in enumerate(pipeline.classes_):
        result[f"P({cls})"] = proba[:, i].round(3)
    return result
 
 
 
if __name__ == "__main__":
    import os
    # Support running from project root or from same directory as data
    data_path = r"C:\Users\mrcoo\Machine-Learning-full-course-main\month_4\model_building_lesson_10\main\ai-impact-jobs-layoff-risk-dataset.csv"
    if not os.path.exists(data_path):
        data_path = "/mnt/user-data/uploads/ai-impact-jobs-layoff-risk-dataset.csv"
 
    pipeline = run(data_path=data_path)
 
    new_employees = [
        {  # High routine, many AI tools → likely High risk
            "Age": 28, "Education_Level": "Bachelor's",
            "Years_of_Experience": 3, "Industry": "IT",
            "Job_Role": "Software Engineer", "Company_Size": "Large",
            "Job_Level": "Entry", "Routine_Task_Percentage": 70,
            "Creativity_Requirement": 30, "Human_Interaction_Level": 20,
            "AI_Adoption_Level": "High", "Number_of_AI_Tools_Used": 8,
            "AI_Usage_Hours_Per_Week": 30, "Tasks_Automated_Percentage": 60,
            "AI_Training_Hours": 5,
        },
        {  # Low routine, PhD, high creativity → likely Low risk
            "Age": 45, "Education_Level": "PhD",
            "Years_of_Experience": 18, "Industry": "Finance",
            "Job_Role": "Auditor", "Company_Size": "Medium",
            "Job_Level": "Senior", "Routine_Task_Percentage": 20,
            "Creativity_Requirement": 80, "Human_Interaction_Level": 70,
            "AI_Adoption_Level": "Low", "Number_of_AI_Tools_Used": 1,
            "AI_Usage_Hours_Per_Week": 3, "Tasks_Automated_Percentage": 10,
            "AI_Training_Hours": 40,
        },
    ]
 
    print("\nNew employee predictions:")
    print(predict_new(pipeline, new_employees).to_string(index=False))