import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, root_mean_squared_error

class Evaluation(ABC):
    @abstractmethod
    def score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        pass

class RegressionMetrics(Evaluation):
    def score(self, y_true: np.ndarray, y_pred:np.ndarray) -> float:
        try:
            logging.info("Calculating metrics for regression model.")
            mse = mean_squared_error(y_true=y_true, y_pred=y_pred)
            rmse = root_mean_squared_error(y_true=y_true, y_pred=y_pred)
            r2 = r2_score(y_true=y_true, y_pred=y_pred)
            summary = {
                "MSE": mse,
                "RMSE": rmse,
                "R2 Score": r2
            }
            
            return summary
        
        except Exception as e:
            logging.error("Error encountered when calculating regression metrics.")
            raise e
        
