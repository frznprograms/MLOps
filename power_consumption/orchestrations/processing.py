import logging
from typing import Union
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class DataStrategy(ABC):
    @abstractmethod
    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreprocessStrategy(DataStrategy):
    ''' Determines preprocessing of data and data engineering'''
    def process_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data['Date_Time'] = pd.to_datetime(data['Date'] + ' ' + data['Time'], format='%d/%m/%Y %H:%M:%S')
            data.drop(columns=['Date', 'Time'], inplace=True)

            french_holidays = [
                "01-01",  # New Year's Day (Jour de l'An)
                "04-12",  # Easter Sunday (Pâques, varies yearly)
                "05-01",  # Labour Day (Fête du Travail)
                "05-08",  # Victory Day (Fête de la Victoire 1945)
                "07-14",  # Bastille Day (Fête Nationale)
                "08-15",  # Assumption Day (Assomption)
                "11-01",  # All Saints' Day (La Toussaint)
                "11-11",  # Armistice Day (Jour d'Armistice)
                "12-25"   # Christmas Day (Noël)
            ]

            na_summary = data.isna().sum()[data.isna().sum() > 0].to_dict()
            n = len(data)
            for key, value in na_summary.items():
                prop_missing = value / n
                if prop_missing > 0.05:
                    logging.info(f"Significant NA values detected in column {key} with \
                                proportion {prop_missing}. Recommend other missing data \
                                handling strategies.")
                    removal_flag = False

            logging.info("No significant NA presence detected.")

            removal_flag = True

            # remove na values
            if removal_flag:
                data = data.dropna()

            # identify periods of time when people are more likely to be at home

            # weekends
            data['day_of_week'] = data['Date_Time'].dt.dayofweek
            data['is_weekend'] = data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

            # holidays
            data['month_day'] = data['Date_Time'].dt.strftime('%m-%d')
            data['is_holiday'] = data['month_day'].isin(french_holidays)
            data['is_holiday'] = data['is_holiday'].apply(lambda x: 1 if True else 0)

            # identify periods of time when consumption may be higher than usual
            # (i.e. sub-metering-3 may go up in water after use of water heaters, use of air con may go up in summer)

            data['month'] = data['Date_Time'].dt.month

            def month_to_season(month):
                season_mappings = {
                    1: 4, 2: 4, 3: 1,
                    4: 1, 5: 1, 6: 2,
                    7: 2, 8: 2, 9: 3,
                    10: 3, 11: 3, 12: 1
                }
                return season_mappings[month]

            data['season'] = data['month'].apply(month_to_season)

            data.drop(columns=[
                'Date_Time',
                'day_of_week',
                'month_day',
                'month'
            ], axis=1, inplace=True)

            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    
            return data
        
        except Exception as e:
            logging.error("Error encountered during data preprocessing.")

class RDataDivisionStrategy(DataStrategy):
    ''' Determines data division for regression tasks'''
    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(columns=['Global_reactive_power'], axis=1)
            y = data['Global_reactive_power']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            logging.error("Error occured in data division.")
            raise e
        

class CDataDivisionStrategy(DataStrategy):
    ''' Determines data division for clustering tasks'''
    def process_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            train_data = data.sample(frac=0.8, random_state=1)
            test_data = data.drop(train_data.index)
            # Reset index for better readability
            train_data = train_data.reset_index(drop=True)
            test_data = test_data.reset_index(drop=True)
            return train_data, test_data
        
        except Exception as e:
            logging.error("Error occured in data division.")
            raise e
        

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def process_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.process_data(self.data)
        
        except Exception as e:
            logging.error(f"Error in handling data cleaning and preprocessing: {e}")
            raise e
        

### LOCALISED TESTING ###
if __name__ == "__main__":
    data = pd.read_csv("/Users/shaneryan_1/Downloads/power_consumption/data/household_power_consumption.txt", 
                       sep=";")
    data_cleaning = DataCleaning(data, DataPreprocessStrategy())
    new_data = data_cleaning.process_data()
    print(new_data.columns)