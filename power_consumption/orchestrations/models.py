import logging
from abc import ABC
from sklearn.linear_model import LinearRegression, GridSearchCV
from sklearn.cluster import KMeans
from steps.config import RegConfig, ClusterConfig

class Model(ABC):
    def train_optimise(self, X_train, y_train, cv=5):
        pass


class LinearRegressionModel(Model):
    def train_optimise(self, X_train, y_train, cv=10):
        try:
            model = LinearRegression()
            params = RegConfig()
            grid = GridSearchCV(
                estimator=model,
                param_grid={
                    "positive": params.pos_coeff,
                    "random_state": params.random_state
                },
                cv=cv
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logging.info("Model training and hyperparameter optimisation completed successfully.")
            
            return best_model
        
        except Exception as e:
            logging.error("Error encountered when attempting training and grid search.")
            raise e


class KMeansClusteringModel(Model):
    def train_optimise(self, X_train, y_train, cv=5):
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
                cv=cv
            )
            grid.fit(X_train, y_train)
            best_model = grid.best_estimator_
            logging.info("Model training and hyperparameter optimisation completed successfully.")
            
            return best_model
        
        except Exception as e:
            logging.error("Error encountered when attempting training and grid search.")
            raise e
