import pandas as pd
import logging
import numpy as np
from zenml.config import DockerSettings
from zenml import step, pipeline, get_step_context
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services.mlflow_deployment import MLFlowDeploymentService
from zenml.client import Client
from zenml.integrations.mlflow.steps.mlflow_deployer import mlflow_model_deployer_step
from mlflow.tracking import MlflowClient, artifact_utils
from pydantic import BaseModel
from steps.cleaning_step import clean_data
from steps.ingestion_step import ingest_data
from steps.training_step import train_model_reg
from steps.evaluation_step import evaluate_regressor
from steps.config import ModelConfig

docker_settings = DockerSettings(required_integrations=[MLFLOW])
DATA_PATH = "/Users/shaneryan_1/Downloads/MLOps/power_consumption/data/household_power_consumption.txt"
MODEL_CONFIG = 'reg'
URI = None

class DeploymentTriggerConfig(BaseModel):
    min_mse: float = 0.1

class MLFlowDeploymentConfig(BaseModel):
    name: str = 'shane-first-mlops-model'
    description: str = 'example of regression model in production'
    pipeline_name: str = get_step_context().pipeline_name
    pipeline_step_name: str = get_step_context.step_name
    model_uri: str = "runs:/<run_id>/model" or "models:/<model_name>/<model_version>"
    model_name: str = 'regression model'
    workers: int = 3
    mlserver: bool = False
    timeoutL: int = DEFAULT_SERVICE_START_STOP_TIMEOUT

@step
def deployment_trigger(mse: float, config: DeploymentTriggerConfig) -> bool:
    # if condition is met, model can be deployed
    return mse < config.min_mse

@step(enable_cache=False)
def prediction_service_loader(
    pipeline_name: str, pipeline_step_name: str, running: bool = True, model_name: str = "model"
) -> MLFlowDeploymentConfig:
    ''' Get the prediction service started by the deployment pipeline '''
    # get the mlflow model deployer stack component
    model_deployer = MLFlowModelDeployer.get_active_model_deployer()

    # fetch existing services with the same pipeline name, step name and model name
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

    print(existing_services)
    print(type(existing_services))
    return existing_services[0]

@step
def predictor(service: MLFlowDeploymentService, data: pd.DataFrame) -> np.ndarray:
    service.start(timeout=10)
    data.drop(columns=['Global_reactive_power'], axis=1, inplace=True)
    data = data.to_numpy()
    prediction = service.predict(data)
    return prediction

@step
# if artifact uri is known
def deploy_model() -> MLFlowDeploymentService:
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    mlflow_deployment_config = MLFlowDeploymentConfig(
        name= 'shane-first-mlops-model',
        description = 'example of regression model in production',
        pipeline_name= get_step_context().pipeline_name,
        pipeline_step_name = get_step_context.step_name,
        model_uri = "runs:/<run_id>/model" or "models:/<model_name>/<model_version>",
        model_name = 'regression model',
        workers = 3,
        mlserver = False,
        timeout = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    )

    service = model_deployer.deploy_model(
        config=mlflow_deployment_config,
        service_type=MLFlowDeploymentService.SERVICE_TYPE
    )
    logging.info(f"The deployed service info: {model_deployer.get_model_server_info(service)}")
    
    return service


@step
# if artifact uri is known
def deploy_model() -> MLFlowDeploymentService:
    # Deploy a model using the MLflow Model Deployer
    zenml_client = Client()
    model_deployer = zenml_client.active_stack.model_deployer
    experiment_tracker = zenml_client.active_stack.experiment_tracker
    # Let's get the run id of the current pipeline
    mlflow_run_id = experiment_tracker.get_run_id(
        experiment_name=get_step_context().pipeline_name,
        run_name=get_step_context().run_name,
    )
    # Once we have the run id, we can get the model URI using mlflow client
    experiment_tracker.configure_mlflow()
    client = MlflowClient()
    model_name = "model" # set the model name that was logged
    model_uri = artifact_utils.get_artifact_uri(
        run_id=mlflow_run_id, artifact_path=model_name
    )
    mlflow_deployment_config = MLFlowDeploymentConfig(
        name= 'shane-first-mlops-model',
        description = 'example of regression model in production',
        pipeline_name= get_step_context().pipeline_name,
        pipeline_step_name = get_step_context.step_name,
        model_uri = "runs:/<run_id>/model" or "models:/<model_name>/<model_version>",
        model_name = 'regression model',
        workers = 3,
        mlserver = False,
        timeout = DEFAULT_SERVICE_START_STOP_TIMEOUT,
    )
    service = model_deployer.deploy_model(
        config=mlflow_deployment_config, 
        service_type=MLFlowDeploymentService.SERVICE_TYPE
    )
    return service







##################################################################
@pipeline(enable_cache=True, settings={"docker": docker_settings})
def continuous_deployment(min_acc: float = 0.5, workers: int = 1, timeout: int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    data = ingest_data(DATA_PATH)
    config = ModelConfig(task=MODEL_CONFIG)
    cleaned_data = clean_data(data)
    if config.task == 'reg':
        res = train_model_reg(cleaned_data, config=MODEL_CONFIG)
        performance_metrics = evaluate_regressor(res)
    
    deployment_decision = deployment_trigger(acc = performance_metrics)
    mlflow_model_deployer_step(
        model=res,
        deploy_decision=deployment_decision,
        workers=workers,
        timeout=timeout
    )

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def inference_deployment(pipeline_name: str, pipeline_step_name: str):
    # Link all the steps artifacts together
    data = ingest_data(DATA_PATH)
    model_deployment_service = prediction_service_loader(
        pipeline_name=pipeline_name,
        pipeline_step_name=pipeline_step_name,
        running=False,
    )
    predictor(service=model_deployment_service, data=data) 