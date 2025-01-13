import logging
from typing import Tuple
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from orchestrations.cleaning_logic import (
    DataPreprocessStrategy, RDataDivisionStrategy, CDataDivisionStrategy, DataCleaning
)
from .config import ModelConfig

@step
def clean_data(data: pd.DataFrame, config: ModelConfig) -> Tuple[
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series],  # Regression task output
    Tuple[pd.DataFrame, pd.DataFrame]  # Clustering task output
]:
    try:
        preprocess_strat = DataPreprocessStrategy()
        data_cleaner = DataCleaning(data=data, strategy=preprocess_strat)
        processed_data = data_cleaner.process_data()
        if config.model_task == 'reg':
            division_strategy = RDataDivisionStrategy
            data_divider = DataCleaning(data=processed_data, strategy=division_strategy)
            X_train, X_test, y_train, y_test = data_divider.process_data()
            logging.info("Data division completed successfully.")
            return X_train, X_test, y_train, y_test
        else:
            division_strategy = CDataDivisionStrategy
            data_divider = DataCleaning(data=processed_data, strategy=division_strategy)
            training_data, testing_data = data_divider.process_data()
            logging.info("Data division completed successfully.")
            return training_data, testing_data
    
    except Exception as e:
            logging.error(f"Error in handling data cleaning and preprocessing: {e}")
            raise e


