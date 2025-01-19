from typing_extensions import Union
from dataclasses import dataclass
import pandas as pd
from sklearn.base import RegressorMixin, ClusterMixin

@dataclass
class ModelAndData:
    trained_model: Union[RegressorMixin, ClusterMixin]
    X_test: pd.DataFrame
    y_test: pd.Series