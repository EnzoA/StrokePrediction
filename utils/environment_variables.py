from enum import Enum

class EnvironmentVariables(Enum):
    RAW_DATASET_URL = 'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset'

    S3_RAW_DATASET = 's3://data/raw/healthcare-dataset-stroke-data.csv'
    
    DATA_JSON_LOCAL_PATH = 'data_info/data.json'
    
    MLFLOW_BASE_URL = 'http://mlflow:5000'
    MLFLOW_EXPERIMENT_NAME = 'Stroke Prediction'
