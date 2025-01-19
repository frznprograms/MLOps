from zenml.pipelines import pipeline
from steps.ingestion_step import ingest_data
from steps.cleaning_step import clean_data
from steps.training_step import train_model_cluster, train_model_reg
from steps.evaluation_step import evaluate_clustering, evaluate_regressor
from steps.config import ModelConfig

@pipeline(enable_cache=True)
def train_pipeline(data_path: str, config: ModelConfig) -> dict:
    data = ingest_data(data_path)
    config = ModelConfig(task='reg')
    cleaned_data = clean_data(data)
    if config.task == 'reg':
        res = train_model_reg(cleaned_data, config=config)
        performance_metrics = evaluate_regressor(res)
        
        return performance_metrics

