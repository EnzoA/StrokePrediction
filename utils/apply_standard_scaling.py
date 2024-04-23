def apply_standard_scaling():
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        from sklearn.preprocessing import StandardScaler

        from utils.environment_variables import EnvironmentVariables

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN.value)
        X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST.value)

        sc_X = StandardScaler(with_mean=True, with_std=True)
        X_train_arr = sc_X.fit_transform(X_train)
        X_test_arr = sc_X.transform(X_test)

        X_train = pd.DataFrame(X_train_arr, columns=X_train.columns)
        X_test = pd.DataFrame(X_test_arr, columns=X_test.columns)

        save_to_csv(X_train, EnvironmentVariables.S3_X_TRAIN.value)
        save_to_csv(X_test, EnvironmentVariables.S3_X_TEST.value)

        # Save information of the dataset
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON.value)
            result = client.get_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON.value)
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = sc_X.mean_.tolist()
        data_dict['standard_scaler_std'] = sc_X.scale_.tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key=EnvironmentVariables.S3_DATA_JSON.value,
            Body=data_string
        )

        #mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL.value)
        #experiment = mlflow.set_experiment("Stroke")

        # Obtain the last experiment run_id to log the new information
        #list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        #with mlflow.start_run(run_id=list_run[0].info.run_id):

        #    mlflow.log_param("Train observations", X_train.shape[0])
        #    mlflow.log_param("Test observations", X_test.shape[0])
        #    mlflow.log_param("Standard Scaler feature names", sc_X.feature_names_in_)
        #    mlflow.log_param("Standard Scaler mean values", sc_X.mean_)
        #    mlflow.log_param("Standard Scaler scale values", sc_X.scale_)

