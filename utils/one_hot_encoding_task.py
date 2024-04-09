# type: ignore
from utils.environment_variables import EnvironmentVariables

def set_one_hot_encoding_variables():
        '''
        Converts categorical variables into one-hot encoding.
        '''
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        import pandas as pd
        import numpy as np

        # TODO: En este punto del dag, la ruta probablemente sea la de un dataset parcialmente preprocesado, no la del raw.
        dataset = wr.s3.read_csv(EnvironmentVariables.S3_RAW_DATASET)

        columns_to_encode = ['gender', 'work_type', 'smoking_status', 'Residence_type', 'bmi', 'avg_glucose_level']
        columns_drop_first = ['Residence_type', 'bmi', 'avg_glucose_level']
        columns_to_drop = ['work_type_Never_worked', 'smoking_status_Unknown']
        
        for column in columns_to_encode:
            drop_first = column in columns_drop_first
            one_hot_encoded = pd.get_dummies(dataset[column], prefix=column, dtype=float, drop_first=drop_first)
            dataset = pd.concat([dataset, one_hot_encoded], axis=1)
        
        dataset.drop(columns=columns_to_encode + columns_to_drop, axis=1, inplace=True)

        wr.s3.to_csv(df=dataset,
                     path=EnvironmentVariables.S3_RAW_DATASET,
                     index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON)
            result = client.get_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON)
            text = result['Body'].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != '404':
                # Something else has gone wrong.
                raise e

        # Upload JSON String to an S3 Object
        # TODO: Verificar esta lógica. Al momento de ejecutar esta task, el json ya existiría en s3? Tendría todos los campos esperados?
        #       Esta task sólo debería agregar el field columns_after_one_hot_encoding?
        data_dict['columns_after_one_hot_encoding'] = dataset.columns.drop(['stroke'], axis=1).to_list()

        category_dummies_dict = {}
        for column in columns_to_encode:
            category_dummies_dict[column] = np.sort(dataset[column].unique()).tolist()

        data_dict['categories_values_per_categorical'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key=EnvironmentVariables.S3_DATA_JSON,
            Body=data_string
        )

        mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL)
        experiment = mlflow.set_experiment(EnvironmentVariables.MLFLOW_EXPERIMENT_NAME)

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={'experiment': 'etl', 'dataset': EnvironmentVariables.MLFLOW_EXPERIMENT_NAME},
                         log_system_metrics=True)

        target_col = 'stroke'

        mlflow_dataset = mlflow.data.from_pandas(dataset,
                                                 source=EnvironmentVariables.RAW_DATASET_URL,
                                                 targets=target_col,
                                                 name='stroke_data_complete')
        mlflow_dataset_yes_no_encoding = mlflow.data.from_pandas(dataset,
                                                                 source=EnvironmentVariables.RAW_DATASET_URL,
                                                                 targets=target_col,
                                                                 name='stroke_data_complete_with_yes_no_encoding')
        mlflow.log_input(mlflow_dataset, context='Dataset')
        mlflow.log_input(mlflow_dataset_yes_no_encoding, context='Dataset')