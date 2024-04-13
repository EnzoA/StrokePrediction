from utils.environment_variables import EnvironmentVariables

def get_raw_dataset():
    import pandas as pd
    import awswrangler as wr
    import logging

    try:
        _ = wr.s3.read_csv(EnvironmentVariables.S3_RAW_DATASET)
    except FileNotFoundError:
        logging.info(f'El dataset crudo no ha sido creado aún en la ruta {EnvironmentVariables.S3_RAW_DATASET}.')
        raw_dataset = pd.read_csv(EnvironmentVariables.RAW_DATASET_URL)
        wr.s3.to_csv(raw_dataset, EnvironmentVariables.S3_RAW_DATASET, index=False)
