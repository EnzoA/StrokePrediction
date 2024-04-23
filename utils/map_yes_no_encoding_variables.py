def map_yes_no_encoding_variables(s3_path, dataset_type):
    '''
    Maps binary yes-no variables into a numerical 1-0 representation.
    '''
    from utils.environment_variables import EnvironmentVariables

    import json
    import datetime
    import boto3
    import botocore.exceptions
    import mlflow

    import awswrangler as wr
    import pandas as pd
    import numpy as np

    dataset = wr.s3.read_csv(s3_path)

    column_to_map = 'ever_married'
    dataset[column_to_map] = dataset[column_to_map].map({ 'Yes': 1, 'No': 0 })

    wr.s3.to_csv(df=dataset,
                 path=s3_path,
                 index=False)

    # Save information of the dataset
    client = boto3.client('s3')

    data_dict = {}
    try:
        client.head_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON.value)
        result = client.get_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON.value)
        text = result['Body'].read().decode()
        data_dict = json.loads(text)
    except botocore.exceptions.ClientError as e:
        if e.response['Error']['Code'] != '404':
            # Something else has gone wrong.
            raise e

    data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
    data_string = json.dumps(data_dict, indent=2)

    client.put_object(
        Bucket='data',
        Key=EnvironmentVariables.S3_DATA_JSON.value,
        Body=data_string
    )

    #mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL.value)
    #experiment = mlflow.set_experiment(EnvironmentVariables.MLFLOW_EXPERIMENT_NAME.value)

    #mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
    #                    experiment_id=experiment.experiment_id,
    #                    tags={'experiment': 'etl', 'dataset': EnvironmentVariables.MLFLOW_EXPERIMENT_NAME.value},
    #                    log_system_metrics=True)

    #target_col = 'stroke'

    #mlflow_dataset = mlflow.data.from_pandas(dataset,
    #                                         source=s3_path,
    #                                         targets=target_col,
    #                                         name=f'stroke_{dataset_type}_data_complete')
    #mlflow_dataset_one_hot_encoding = mlflow.data.from_pandas(dataset,
    #                                                          source=s3_path,
    #                                                          targets=target_col,
    #                                                          name=f'stroke_{dataset_type}_data_complete_with_yes_no_encoding')
    #mlflow.log_input(mlflow_dataset, context='Dataset')
    #mlflow.log_input(mlflow_dataset_one_hot_encoding, context='Dataset')

#def map_yes_no_encoding_variables():
    '''
    Maps binary yes-no variables into a numerical 1-0 representation.
    '''
#    from utils.environment_variables import EnvironmentVariables

#    import json
#    import datetime
#    import boto3
#    import botocore.exceptions
#    import mlflow

#    import awswrangler as wr
#    import pandas as pd
#    import numpy as np

#    dataset = wr.s3.read_csv(EnvironmentVariables.S3_RAW_DATASET)

#    column_to_map = 'ever_married'
#    dataset[column_to_map] = dataset[column_to_map].map({ 'Yes': 1, 'No': 0 })

 #   wr.s3.to_csv(df=dataset,
  #               path=EnvironmentVariables.S3_RAW_DATASET,
  #               index=False)

    # Save information of the dataset
  #  client = boto3.client('s3')

   # data_dict = {}
   # try:
   #     client.head_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON)
   #     result = client.get_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON)
   #     text = result['Body'].read().decode()
   #     data_dict = json.loads(text)
   # except botocore.exceptions.ClientError as e:
   #     if e.response['Error']['Code'] != '404':
            # Something else has gone wrong.
   #         raise e

   # data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
   # data_string = json.dumps(data_dict, indent=2)

   # client.put_object(
   #     Bucket='data',
   #     Key=EnvironmentVariables.S3_DATA_JSON,
   #     Body=data_string
   # )

    #mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL)
    #experiment = mlflow.set_experiment(EnvironmentVariables.MLFLOW_EXPERIMENT_NAME)

    #mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
    #                    experiment_id=experiment.experiment_id,
    #                    tags={'experiment': 'etl', 'dataset': EnvironmentVariables.MLFLOW_EXPERIMENT_NAME},
    #                    log_system_metrics=True)

    #target_col = 'stroke'

    #mlflow_dataset = mlflow.data.from_pandas(dataset,
    #                                         source=EnvironmentVariables.S3_RAW_DATASET,
    #                                         targets=target_col,
    #                                         name='stroke_data_complete')
    #mlflow_dataset_one_hot_encoding = mlflow.data.from_pandas(dataset,
    #                                                          source=EnvironmentVariables.S3_RAW_DATASET,
    #                                                          targets=target_col,
    #                                                          name='stroke_data_complete_with_one_hot_encoding')
    #mlflow.log_input(mlflow_dataset, context='Dataset')
    #mlflow.log_input(mlflow_dataset_one_hot_encoding, context='Dataset')