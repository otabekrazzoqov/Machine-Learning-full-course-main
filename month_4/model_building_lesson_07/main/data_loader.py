import os 
import pandas as pd 
from logger import get_logger

logger = get_logger(__name__)

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DataLoader:
    SUPPORTED_FORMATS = [".csv", ".parquet", ".json"]

    def __init__(self, subfolder: str = "raw"):
        self.subfolder = subfolder
        self.df = None
        self.filepath = None
        logger.info(f"DataLoader initialized | subfolder: {subfolder}")

    
    def load(self, filename: str) -> "DataLoader":
        self.filepath = os.path.join(ROOT_DIR, "data", self.subfolder, filename)
        ext = os.path.splitext(filename)[1].lower()
        logger.info(f"Loading file | path: {self.filepath}")

        try:
            if ext == ".csv":
                self.df = pd.read_csv(self.filepath)
            elif ext == ".parquet":
                self.df = pd.read_parquet(self.filepath)
            elif ext == ".json":
                self.df = pd.read_json(self.filepath)

            logger.info(f"file loaded successfully | shape: {self.df.shape} | columns: {self.df.columns.to_list()} ")

        except FileNotFoundError:
            logger.error(f"File Not Found: {self.filepath}")
            raise
        except Exception as e:
            logger.error(f"Failed to load file: {e}", exc_info=True)
            raise
        return self
    

    def validate(self) -> "DataLoader":
        logger.info("Running validation checks...")

        if self.df is None:
            logger.error("No data loaded. Call load() before validate()")
            raise RuntimeError("No Data loaded. Call load() first.")
        
        if self.df.empty:
            logger.critical("Loaded Dataframe is empty!")
            raise ValueError("DataFrame is empty after loading.")
        
        null_counts = self.df.isnull().sum()
        null_cols = null_counts[null_counts>0]
        if not null_counts.empty:
            logger.warning(f"Columns with missing values:\n{null_cols.to_string()}")
        else:
            logger.info("No missing values found")

        dupe_count = self.df.duplicated().sum()
        if dupe_count > 0:
            logger.warning(f"Found {dupe_count} duplicate rows")
        else:
            logger.info("No duplicate rows found")

        logger.info(f"Validation passed | shape: {self.df.shape}")
        logger.info(f"Dtypes:\n{self.df.dtypes.to_string()}")
        return self


    def save(self, filename: str, subfolder: str = "preprocessed") -> "DataLoader":
        path = os.path.join(ROOT_DIR, "data", subfolder, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        ext = os.path.splitext(filename)[1].lower()
        logger.info(f"Saving data | path: {path}")

        try:
            if ext == ".csv":
                self.df.to_csv(path, index=False)
            elif ext == ".parquet":
                self.df.to_parquet(path, index=False)
            else:
                logger.error(f"Cannot save: unsupported format'{ext}'")
                raise ValueError(f"Unsupported save format: {ext}")
            
            logger.info(f"Data saved successfully | shape: {self.df.shape}")

        except Exception as e:
            logger.error(f"Failed to save data: {e}", exc_info=True)
            raise
        return self
    

    def get_df(self) -> pd.DataFrame:
        if self.df is None:
            logger.error("No data available. Call load() first.")
            raise RuntimeError("No data loaded")
        return self.df.copy()
    

    def info(self) -> "DataLoader":
        if self.df is not None:
            logger.info("─" * 50)
            logger.info(f"Shape    : {self.df.shape}")
            logger.info(f"Columns  : {self.df.columns.tolist()}")
            logger.info(f"Nulls    :\n{self.df.isnull().sum().to_string()}")
            logger.info(f"Dtypes   :\n{self.df.dtypes.to_string()}")
            logger.info(f"Sample   :\n{self.df.head(3).to_string()}")
            logger.info("─" * 50)
        else:
            logger.warning("info() called but no data is loaded yet")
        return self
    
