from zenml.steps import BaseParameters

class ModelConfig(BaseParameters):
    model_task: str = "reg"

class RegConfig(BaseParameters):
    pos_coeff: list = [False, True]
    random_state: int = 1

class ClusterConfig(BaseParameters):
    n_clusters: list = [5, 8, 10]
    max_iter: list = [300, 400, 500]
    tol: list = [0.005, 0.001, 0.0005]
    random_state: int = 1