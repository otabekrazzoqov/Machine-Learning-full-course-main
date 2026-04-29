from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

class Preprocessing:
    def __init__(self, df):
        self.df = df.copy() 

    def handling_missing_values(self):
        for col in self.df.columns:
            if self.df[col].isnull().any():
                if self.df[col].dtype == "object":
                    self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
                else:
                    self.df[col] = self.df[col].fillna(self.df[col].mean())
        return self
    
    
    def encoding(self):
        encoder = LabelEncoder()
        for col in self.df.colums:
            if self.df[col].dtype == "object":
                if self.df[col].nunique() <=5:
                    dummies = pd.get_dummies(self.df[col], prefix=col, dtype=int)
                    self.df = pd.concat(self.df.drop(columns=col), dummies, axis=1)
                else:
                    self.df[col] = encoder.fit_transform(self.df[col])
        return self


    