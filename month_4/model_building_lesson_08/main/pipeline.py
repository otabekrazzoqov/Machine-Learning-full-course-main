from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataPipeline:
   
    def __init__(self, data: pd.DataFrame | str | Path | None = None, target_column: str | None = None):
        self.target_column = target_column
        self.df: pd.DataFrame | None = None
        self.original_df: pd.DataFrame | None = None
        self.scaler: StandardScaler | MinMaxScaler | None = None
        self.scaled_columns: list[str] = []

        if data is not None:
            if isinstance(data, pd.DataFrame):
                self.set_df(data)
            else:
                self.load_csv(data)

    def load_csv(self, file_path: str | Path) -> pd.DataFrame:
        self.df = pd.read_csv(file_path)
        self.original_df = self.df.copy()
        return self.df

    def set_df(self, df: pd.DataFrame) -> pd.DataFrame:
        self.df = df.copy()
        self.original_df = df.copy()
        return self.df

    def handle_missing_data(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        fill_values: dict[str, object] | None = None,
    ) -> pd.DataFrame:
      
        df = self._require_df()
        fill_values = fill_values or {}

        for column, value in fill_values.items():
            if column in df.columns:
                df[column] = df[column].fillna(value)

        numeric_columns = self._feature_columns(df.select_dtypes(include="number").columns)
        categorical_columns = self._feature_columns(df.select_dtypes(exclude="number").columns)

        for column in numeric_columns:
            if column in fill_values:
                continue
            if numeric_strategy == "mean":
                value = df[column].mean()
            elif numeric_strategy == "median":
                value = df[column].median()
            elif numeric_strategy == "zero":
                value = 0
            else:
                raise ValueError('numeric_strategy must be "mean", "median", or "zero".')
            df[column] = df[column].fillna(value)

        for column in categorical_columns:
            if column in fill_values:
                continue
            if categorical_strategy == "mode":
                mode_value = df[column].mode(dropna=True)
                value = mode_value.iloc[0] if not mode_value.empty else "Unknown"
            elif categorical_strategy == "unknown":
                value = "Unknown"
            else:
                raise ValueError('categorical_strategy must be "mode" or "unknown".')
            df[column] = df[column].fillna(value)

        self.df = df
        return self.df

    def encode_categorical(self, columns: Iterable[str] | None = None, drop_first: bool = False) -> pd.DataFrame:
        df = self._require_df()

        if columns is None:
            columns = self._feature_columns(df.select_dtypes(include=["object", "category", "bool"]).columns)
        else:
            columns = [column for column in columns if column in df.columns and column != self.target_column]

        self.df = pd.get_dummies(df, columns=list(columns), drop_first=drop_first, dtype=int)
        return self.df

    def scale_numeric(
        self,
        columns: Iterable[str] | None = None,
        method: str = "standard",
    ) -> pd.DataFrame:
       
        df = self._require_df()

        if columns is None:
            columns = self._feature_columns(df.select_dtypes(include="number").columns)
        else:
            columns = [column for column in columns if column in df.columns and column != self.target_column]

        self.scaled_columns = list(columns)
        if not self.scaled_columns:
            return df

        if method == "standard":
            self.scaler = StandardScaler()
        elif method == "minmax":
            self.scaler = MinMaxScaler()
        else:
            raise ValueError('method must be "standard" or "minmax".')

        df[self.scaled_columns] = self.scaler.fit_transform(df[self.scaled_columns])
        self.df = df
        return self.df

    def fit_transform(
        self,
        numeric_strategy: str = "median",
        categorical_strategy: str = "mode",
        scale_method: str = "standard",
        drop_first: bool = False,
    ) -> pd.DataFrame:
        self.handle_missing_data(
            numeric_strategy=numeric_strategy,
            categorical_strategy=categorical_strategy,
        )
        self.encode_categorical(drop_first=drop_first)
        self.scale_numeric(method=scale_method)
        return self.get_df()

    def get_df(self) -> pd.DataFrame:
        return self._require_df().copy()

    def get_original_df(self) -> pd.DataFrame:
        if self.original_df is None:
            raise ValueError("No original DataFrame found. Load a CSV or set a DataFrame first.")
        return self.original_df.copy()

    def _feature_columns(self, columns: Iterable[str]) -> list[str]:
        return [column for column in columns if column != self.target_column]

    def _require_df(self) -> pd.DataFrame:
        if self.df is None:
            raise ValueError("No DataFrame found. Use load_csv() or set_df() first.")
        return self.df.copy()
