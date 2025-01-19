import logging
import pandas as pd

class DataIngester():
    def __init__(self, data_path: str):
        self.data_path = data_path

    def ingest(self):
        logging.info(f"Ingesting data from source: {self.data_path}")
        try: 
            data = pd.read_csv(self.data_path, sep=";")
            return data
        
        except Exception as e:
            logging.error(f"Error occured when attempting to ingest data from path: \n\
                          {self.data_path}")
            raise e
        

if __name__ == '__main__':
    ingester = DataIngester("/Users/shaneryan_1/Downloads/power_consumption/data/household_power_consumption.txt")
    data = ingester.ingest()
    print(data.columns)