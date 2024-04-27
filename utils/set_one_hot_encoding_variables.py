def set_one_hot_encoding_variables(s3_path, dataset_type):
        '''
        Converts categorical variables into one-hot encoding.
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

        columns_to_encode = ['gender', 'work_type', 'smoking_status', 'Residence_type', 'bmi', 'avg_glucose_level']
        columns_drop_first = ['Residence_type', 'bmi', 'avg_glucose_level']
        columns_to_drop = ['work_type_Never_worked', 'smoking_status_Unknown']
        
        for column in columns_to_encode:
            drop_first = column in columns_drop_first
            one_hot_encoded = pd.get_dummies(dataset[column], prefix=column, dtype=float, drop_first=drop_first)
            dataset = pd.concat([dataset, one_hot_encoded], axis=1)
        
        dataset.drop(columns=columns_to_encode + columns_to_drop, inplace=True)

        # Write the processed dataset back to S3
        wr.s3.to_csv(df=dataset, path=s3_path, index=False)

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

        data_dict['columns_one_hot_encoded'] = dataset.columns.to_list()

        categories_list = dataset.columns.to_list()
        category_dummies_dict = {}
        for category in categories_list:
            category_dummies_dict[category] = np.sort(dataset[category].unique()).tolist()

        data_dict['dummy_values_per_one_hot_encoded'] = category_dummies_dict

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key=EnvironmentVariables.S3_DATA_JSON.value,
            Body=data_string
        )