def map_yes_no_encoding_variables(s3_path):
    '''
    Maps binary yes-no variables into a numerical 1-0 representation.
    '''
    from utils.environment_variables import EnvironmentVariables

    import json
    import datetime
    import boto3
    import botocore.exceptions
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


    data_dict['yes_no_encoding_columns'] = [column_to_map, 'hypertension', 'heart_disease']
    data_string = json.dumps(data_dict, indent=2)

    client.put_object(
        Bucket='data',
        Key=EnvironmentVariables.S3_DATA_JSON.value,
        Body=data_string
    )