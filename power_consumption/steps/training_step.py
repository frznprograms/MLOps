import logging
import pandas as pd
from zenml import step
from zenml.client import Client
from orchestrations.models import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelConfig
import mlflow

experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def train_model_reg(X_train: pd.DataFrame, X_test: pd.DataFrame,
                    y_train: pd.Series, y_test: pd.Series, config: ModelConfig) -> RegressorMixin:
    try:
        if config.model_task == 'reg':
            model = LinearRegressionModel()
            mlflow.sklearn.autolog()
            trained_model = model.train_optimise(X_train, y_train)
            
            return trained_model
    
    except Exception as e:
        logging.error("Error encountered when training model for {config.model_task} task.")
        raise e



