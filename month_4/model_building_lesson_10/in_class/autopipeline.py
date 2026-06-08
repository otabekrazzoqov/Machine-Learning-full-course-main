import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer

class PlayerPipeline(BaseEstimator):   
    

    def __init__(self, df: pd.DataFrame, target: str, model=None):
        self.df = df.copy()
        self.target = target
        self.model_algorithm = model
        self.model = None
        self.preprocessor = None
    

    def _prepare_features(self):
        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]

        num_col = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_col = X.select_dtypes(exclude=[np.number]).columns.tolist()

    
        numeric_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', MinMaxScaler())
        ])

        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        self.preprocessor = ColumnTransformer([
            ('num', numeric_pipeline, num_col),
            ('cat', categorical_pipeline, cat_col)
        ])

        return X, y

    def fit(self):
        X, y = self._prepare_features()

        if self.model_algorithm is None:
            raise ValueError("No model specified. Pass a scikit-learn estimator.")

        
        self.model = Pipeline([
            ('preprocessor', self.preprocessor),
            ('estimator', self.model_algorithm)
        ])

        self.model.fit(X, y)
        return self

    def predict(self, X=None):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if X is None:
            X = self.df.drop(columns=[self.target])
        return self.model.predict(X)

    def score(self, X=None, y=None):
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        if X is None and y is None:
            X = self.df.drop(columns=[self.target])
            y = self.df[self.target]
        return self.model.score(X, y)
