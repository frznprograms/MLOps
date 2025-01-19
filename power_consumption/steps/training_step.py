import logging
import pandas as pd
import numpy as np
from zenml import step
from zenml.client import Client
from orchestrations.models import LinearRegressionModel, KMeansClusteringModel
from sklearn.base import RegressorMixin, ClusterMixin
from .config import ModelConfig
import mlflow
from orchestrations.processing import (
    RDataDivisionStrategy, CDataDivisionStrategy, DataCleaning
)
from .datatypes import ModelAndData

experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def train_model_reg(data: pd.DataFrame, config: ModelConfig) -> ModelAndData:
    if config.task == 'reg':
        logging.info("Regression task acknowledged. Preparing data for training.")
        division_strat = RDataDivisionStrategy()
        divider = DataCleaning(data=data, strategy=division_strat)
        X_train, X_test, y_train, y_test = divider.process_data()
        X_train = X_train.select_dtypes(include=[np.number])
        X_test = X_test.select_dtypes(include=[np.number])
        logging.info("Data division for regression task completed. Beginning training now.")
        
        model = LinearRegressionModel()
        mlflow.sklearn.autolog()
        trained_model = model.train_optimise(X_train=X_train, y_train=y_train)
        mlflow.sklearn.log_model(trained_model, 'LinearRegressionModel')
        
        return ModelAndData(trained_model=trained_model, X_test=X_test, y_test=y_test)

    
@step(experiment_tracker=experiment_tracker.name)
def train_model_cluster(data: pd.DataFrame, config: ModelConfig) -> ClusterMixin:
    if config.task != 'reg':
        logging.info("Clustering task acknowledged. Preparing data for training.")
        division_strat = CDataDivisionStrategy()
        divider = DataCleaning(data=data, strategy=division_strat)
        training_data, testing_data = divider.process_data()
        training_data = training_data.select_dtypes(include=[np.number])
        testing_data = testing_data.select_dtypes(include=[np.number])
        logging.info("Data division for regression task completed. Beginning training now.")

        model = KMeansClusteringModel()
        mlflow.sklearn.autolog()
        trained_model = model.train_optimise(training_data, testing_data)
        mlflow.sklearn.log(model, trained_model, 'KMeansClusteringModel')
        
        return trained_model


if __name__ == '__main__':
    config = ModelConfig(task='reg')
    data = pd.read_csv("/Users/shaneryan_1/Downloads/power_consumption/data/power_consumption_clean.csv")
    data.drop(columns=['Date_Time'], axis=1, inplace=True)
    res = train_model_reg(data=data, config=config)
    print(f"Output is of correct class ModelAndData: {isinstance(res, ModelAndData)}")
    print(f"Output model is of correct class RegressorMixin: {isinstance(res.trained_model, RegressorMixin)}")
    print(f"Output data is of correct class: {isinstance(res.X_test, pd.DataFrame)}")