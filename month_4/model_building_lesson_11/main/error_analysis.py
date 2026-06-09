import pandas as pd
import os

class DataLoader:

    def __init__(self, filename: str):
        self.filename = filename

    def load(self, filename: str):
        return pd.read_csv(f"{filename}")

        