import pandas as pd
import logging
import numpy as np
from zenml.config import DockerSettings
from zenml import step, pipeline, get_step_context
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import MLFlowDeploymentConfig
from zenml.client import Client
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from mlflow.tracking import MlflowClient
from pydantic import BaseModel
from steps.cleaning_step import clean_data
from steps.ingestion_step import ingest_data
from steps.training_step import train_model_reg
from steps.evaluation_step import evaluate_regressor
from steps.config import ModelConfig

docker_settings = DockerSettings(required_integrations=[MLFLOW])
DATA_PATH = "/Users/shaneryan_1/Downloads/MLOps/power_consumption/data/household_power_consumption.txt"
MODEL_CONFIG = 'reg'

# Deployment Trigger Configuration
class DeploymentTriggerConfig(BaseModel):
    min_mse: float = 0.1

@step
def deployment_trigger(mse: float, config: DeploymentTriggerConfig) -> bool:
    """Step to decide whether to deploy the model based on performance."""
    return mse < config.min_mse

@step
def deploy_model_service(
    run_id: str,
    pipeline_name: str,
    pipeline_step_name: str,
    model_name: str = "model"
) -> MLFlowDeploymentService:
    """Deploy a model using the MLflow Model Deployer."""
    # Fetch the artifact URI for the model using the run_id
    client = MlflowClient()
    model_uri = client.get_run(run_id).info.artifact_uri + f"/{model_name}"
    
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer

    mlflow_deployment_config = MLFlowDeploymentConfig(
        name="mlops-model-deployment",
        description="Deployed regression model",
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_uri=model_uri,
        model_name=model_name,
        workers=3,
        mlserver=False,
        timeout=DEFAULT_SERVICE_START_STOP_TIMEOUT,
    )

    service = model_deployer.deploy_model(
        config=mlflow_deployment_config, 
        service_type=MLFlowDeploymentService.SERVICE_TYPE
    )
    logging.info(f"Deployed model service: {service.endpoint}")
    return service

@step
def prediction_service_loader(
    pipeline_name: str, pipeline_step_name: str, running: bool = True, model_name: str = "model"
) -> MLFlowDeploymentService:
    """Load an MLflow prediction service that is already running."""
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    existing_services = model_deployer.find_model_server(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        model_name=model_name,
        running=running
    )

    if not existing_services:
        raise RuntimeError(
            f"No MLFlow prediction service deployed by the "
            f"{pipeline_step_name} step in the {pipeline_name} pipeline "
            f"for the {model_name} model is currently running."
        )
    return existing_services[0]

@step
def predictor(service: MLFlowDeploymentService, data: pd.DataFrame) -> np.ndarray:
    """Use the deployed model service to make predictions."""
    service.start(timeout=10)
    data.drop(columns=["Global_reactive_power"], axis=1, inplace=True)
    data = data.to_numpy()
    predictions = service.predict(data)
    return predictions


##################################################################
# Continuous Deployment Pipeline
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment_pipeline(min_mse: float = 0.1, workers: int = 1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    """Pipeline for continuous deployment of models."""
    data = ingest_data(DATA_PATH)
    config = ModelConfig(task=MODEL_CONFIG)
    cleaned_data = clean_data(data)

    if config.task == "reg":
        res = train_model_reg(cleaned_data, config=MODEL_CONFIG)
        performance_metrics = evaluate_regressor(res)

    # Decide on deployment
    deploy_decision = deployment_trigger(mse=performance_metrics, config=DeploymentTriggerConfig(min_mse=min_mse))
    
    # if deploy_decision is True
    if deploy_decision:
        deploy_model_service(
            model_uri="runs:/<run_id>/model" or "models:/<model_name>/<model_version>",
            pipeline_name=get_step_context().pipeline_name,
            pipeline_step_name=get_step_context().step_name,
            model_name="regression model",
        )

# Inference Deployment Pipeline
@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_deployment_pipeline(pipeline_name: str, pipeline_step_name: str):
    """Pipeline for inference using a deployed model."""
    data = ingest_data(DATA_PATH)
    service = prediction_service_loader(
        pipeline_name=pipeline_name, pipeline_step_name=pipeline_step_name
    )
    predictions = predictor(service=service, data=data)
    return predictions
