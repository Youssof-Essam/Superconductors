from sklearn.model_selection import train_test_split
from src.pipeline_step_base import PipleLineStepBase
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class LoadSplitData(PipleLineStepBase):
    def __init__(self, path: str, filename: str, target_column: str, test_size=0.2, random_state=42):
        self.path = path
        self.target_column = target_column
        self.test_size = test_size
        self.random_state = random_state
        
        logger.info(f"Loading data from {path}/{filename}")
        try:
            # We load the data here, but process the split in run()
            self.data = pd.read_csv(f"{path}/{filename}").drop_duplicates()
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def run(self):
        # 1. Separate Features and Target
        if self.target_column not in self.data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataset.")
            
        X = self.data.drop(columns=[self.target_column])
        y = self.data[self.target_column]

        # 2. Perform the split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # 3.Save splits for persistence/audit
        X_train.assign(**{self.target_column: y_train}).to_csv(f"{self.path}/train_split.csv", index=False)
        X_test.assign(**{self.target_column: y_test}).to_csv(f"{self.path}/test_split.csv", index=False)
        
        logger.info(f"Data split and saved successfully to {self.path}")

        return X_train, X_test, y_train, y_test