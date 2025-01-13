import logging
import pandas as pd
import numpy as np
from zenml import step

class DataIngester():
    def __init__(self, data_path: str):
        self.data_path = data_path

    def ingest_data(self):
        logging.info(f"Ingesting data from source: {self.data_path}")
        try: 
            data = pd.read_csv(self.data_path, sep=";")
            return data
        except ImportError:
            data = pd.read_fwf(self.data_path)
            return data
        except Exception as e:
            logging.error(f"Error occured when attempting to ingest data from path: \n\
                          {self.data_path}")
            raise e
        
@step
def ingest_data(data_path: str) -> pd.DataFrame:
    try:
        ingester = DataIngester(data_path=data_path)
        df = ingester.ingest_data()
        logging.info("Data ingestion successful.")
        return df
    except Exception as e:
        raise e