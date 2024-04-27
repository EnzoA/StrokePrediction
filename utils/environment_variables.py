from enum import Enum

class EnvironmentVariables(Enum):
    RAW_DATASET_PATH = '/opt/airflow/dags/data/healthcare-dataset-stroke-data.csv'

    S3_RAW_DATASET = 's3://data/raw/healthcare-dataset-stroke-data.csv'
    S3_DATA_JSON = 'data_info/data.json'

    S3_X_TRAIN = 's3://data/preprocessed/x_train.csv'
    S3_Y_TRAIN = 's3://data/preprocessed/y_train.csv'
    
    S3_X_TEST = 's3://data/preprocessed/x_test.csv'
    S3_Y_TEST = 's3://data/preprocessed/y_test.csv'
    
    DATA_JSON_LOCAL_PATH = '/app/files/data.json'
    MODEL_PKL_LOCAL_PATH = '/app/files/model.pkl'
    
    MLFLOW_BASE_URL = 'http://mlflow:5000'
    MLFLOW_EXPERIMENT_NAME = 'Stroke Prediction'
    MLFLOW_MODEL_NAME_DEV = 'stroke_model_dev'
    MLFLOW_MODEL_NAME_PROD = 'stroke_model_prod'