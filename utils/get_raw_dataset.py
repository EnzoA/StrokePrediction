def get_raw_dataset():
    import pandas as pd
    import awswrangler as wr
    from utils.environment_variables import EnvironmentVariables

    try:
        _ = wr.s3.read_csv(EnvironmentVariables.S3_RAW_DATASET.value)
        
        print(f'El dataset crudo fue correctamente leído en la ruta {EnvironmentVariables.S3_RAW_DATASET.value}.')
    except wr.exceptions.NoFilesFound:
        print(f'El dataset crudo no ha sido creado aún en la ruta {EnvironmentVariables.S3_RAW_DATASET.value}.')
        raw_dataset = pd.read_csv(EnvironmentVariables.RAW_DATASET_PATH.value)
        
        wr.s3.to_csv(raw_dataset, EnvironmentVariables.S3_RAW_DATASET.value, index=False)
        print(f'El dataset crudo fue correctamente persistido en la ruta {EnvironmentVariables.S3_RAW_DATASET.value}')
        
