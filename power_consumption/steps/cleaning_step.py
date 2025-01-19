import logging
from typing import Tuple, Union
from typing_extensions import Annotated
import pandas as pd
from zenml import step
from orchestrations.processing import (
    DataPreprocessStrategy, DataCleaning
)
from .config import ModelConfig
from zenml.integrations.pandas.materializers.pandas_materializer import PandasMaterializer
    

@step
def clean_data(data: pd.DataFrame) -> pd.DataFrame:
     try: 
          strategy = DataPreprocessStrategy()
          data_cleaner = DataCleaning(data=data, strategy=strategy)
          processed_data = data_cleaner.process_data()

          return processed_data
     
     except Exception as e:
          logging.error(f"Error in handling data cleaning and preprocessing: {e}")
          raise e


if __name__ == '__main__':
    raw_data = pd.read_csv("/Users/shaneryan_1/Downloads/MLOps/power_consumption/data/household_power_consumption.txt",
                        sep=';', low_memory=False)
    config = ModelConfig(task='reg')
    result = clean_data(raw_data)
    print(result.columns)