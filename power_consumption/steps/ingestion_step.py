import logging
import pandas as pd
import numpy as np
from zenml import step
from orchestrations.ingestion import DataIngester
        
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    try:
        ingester = DataIngester(data_path=data_path)
        data = ingester.ingest()
        for column in data.columns:
            if column not in ['Date', 'Time']:
                data[column] = pd.to_numeric(data[column], errors='coerce')
        
        logging.info("Data ingestion successful.")
        
        return data
    
    except Exception as e:
        raise e
    

if __name__ == '__main__':
    data_path = "/Users/shaneryan_1/Downloads/power_consumption/data/household_power_consumption.txt"
    data = ingest_data(data_path)
    print(data.columns)