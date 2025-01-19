import logging
import pandas as pd
import numpy as np
from abc import ABC
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.model_selection import GridSearchCV
from steps.config import RegConfig, ClusterConfig
from sklearn.base import RegressorMixin, ClusterMixin

class Model(ABC):
    def train_optimise(self, X_train: pd.DataFrame, y_train: pd.Series , cv: int = 5):
        pass


class LinearRegressionModel(Model):
    def train_optimise(self, X_train: pd.DataFrame, y_train: pd.Series , cv: int = 10) -> RegressorMixin:
        try:
            model = LinearRegression()
            params = RegConfig()
            grid = GridSearchCV(
                estimator=model,
                param_grid={
                    "positive": params.pos_coeff
                },
                cv=cv,
                error_score='raise'
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logging.info("Model training and hyperparameter optimisation completed successfully.")
            
            return best_model
        
        except Exception as e:
            logging.error("Error encountered when attempting training and grid search.")
            raise e


class KMeansClusteringModel(Model):
    def train_optimise(self, X_train: pd.DataFrame, cv: int = 10) -> ClusterMixin:
        try:
            model = KMeans()
            params = ClusterConfig()
            grid = GridSearchCV(
                estimator=model,
                param_grid={
                    "n_clusters": params.n_clusters,
                    "max_iter": params.max_iter,
                    "tol": params.tol
                },
                cv=cv,
                error_score='raise'
            )
            grid.fit(X_train)
            best_model = grid.best_estimator_
            logging.info("Model training and hyperparameter optimisation completed successfully.")
            
            return best_model
        
        except Exception as e:
            logging.error("Error encountered when attempting training and grid search.")
            raise e
        


if __name__ == '__main__':
    data = pd.read_csv("/Users/shaneryan_1/Downloads/power_consumption/data/power_consumption_clean.csv")
    data.drop(columns=['Date_Time'], axis=1, inplace=True)
    X_train = data.drop(columns=['Global_reactive_power'], axis=1)
    y_train = data['Global_reactive_power']
    model = LinearRegressionModel()
    trained_model = model.train_optimise(X_train=X_train, y_train=y_train)
    if trained_model:
        print(isinstance(trained_model, RegressorMixin))
    else:
        print("The model is a NoneType!")
