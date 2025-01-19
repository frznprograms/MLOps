from pipelines.train_pipeline import train_pipeline
from zenml.client import Client
from steps.config import ModelConfig

if __name__ == "__main__":
    #print(Client().active_stack.experiment_tracker.get_tracking_uri())
    config = ModelConfig(task='reg')
    train_pipeline("/Users/shaneryan_1/Downloads/power_consumption/data/household_power_consumption.txt",
                   config=config)