from pydantic import BaseModel

class ModelConfig(BaseModel):
    task: str = "reg"

class RegConfig(BaseModel):
    pos_coeff: list = [False, True]
    random_state: list = [1]

class ClusterConfig(BaseModel):
    n_clusters: list = [5, 8, 10]
    max_iter: list = [300, 400, 500]
    tol: list = [0.005, 0.001, 0.0005]
    random_state: list = [1]
