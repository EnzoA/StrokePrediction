from enum import Enum

class S3Paths(Enum):
    RAW_DATASET = 's3://data/raw/healthcare-dataset-stroke-data.csv'
    
class LocalPaths(Enum):
    DATA_JSON_PATH = 'data_info/data.json'

class MlFlowConstants(Enum):
    BASE_URL = 'http://mlflow:5000'
    EXPERIMENT_NAME = 'Stroke Prediction'