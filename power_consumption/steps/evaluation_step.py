import logging
import pandas as pd
import numpy as np
from zenml import step
from zenml.client import Client
from .datatypes import ModelAndData
from orchestrations.evaluations import RegressionMetrics
from sklearn.base import RegressorMixin, ClusterMixin
from sklearn.metrics import silhouette_score
import mlflow


experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def evaluate_regressor(res: ModelAndData) -> dict:
    try:
        predictions = res.trained_model.predict(res.X_test)
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        evaluator = RegressionMetrics()
        metrics = evaluator.score(res.y_test, predictions)
        mlflow.log_metric("MSE", metrics["MSE"])
        mlflow.log_metric("RMSE", metrics["RMSE"])
        mlflow.log_metric("R2 Score", metrics["R2 Score"])
        logging.info("Model evaluation completed, metrics have been logged to mlflow.")
        logging.info(f"Summary of model performance: {metrics}")
        
        return metrics
    
    except Exception as e:
        logging.error("Error encountered when calculating model metrics.")
        raise e
    
### THIS NEEDS TO BE MODIFIED -> INVESTIGATE WHAT THE LABELS COULD BE

@step(experiment_tracker=experiment_tracker.name)
def evaluate_clustering(res: ModelAndData) -> dict:
    inertia = res.trained_model.inertia_
    labels = res.trained_model.predict(res.X_test)
    silhouette = silhouette_score(res.X_test, labels) #-> WHAT ARE THE LABELS FOR THIS ONE?
    
    mlflow.log_metric("Inertia", inertia)
    mlflow.log_metric("Silhouette Score", silhouette)
    logging.info("Clustering evaluation completed, metrics logged to mlflow.")
    
    return {"Inertia": inertia, "Silhouette Score": silhouette}
    