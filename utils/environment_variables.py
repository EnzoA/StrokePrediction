from enum import Enum

class EnvironmentVariables(Enum):
    RAW_DATASET_URL = 'https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset'

    S3_RAW_DATASET = 's3://data/raw/healthcare-dataset-stroke-data.csv'
    S3_DATA_JSON = 'data_info/data.json'
    
    DATA_JSON_LOCAL_PATH = '/app/files/data.json'
    MODEL_PKL_LOCAL_PATH = '/app/files/model.pkl'
    
    MLFLOW_BASE_URL = 'http://mlflow:5000'
    MLFLOW_EXPERIMENT_NAME = 'Stroke Prediction'