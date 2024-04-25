def load_the_champion_model():
    import mlflow

    model_name = 'stroke_model_prod'
    alias = 'champion'

    client = mlflow.MlflowClient()
    model_data = client.get_model_version_by_alias(model_name, alias)

    champion_version = mlflow.sklearn.load_model(model_data.source)

    return champion_version

def load_the_train_test_data():
    import awswrangler as wr
    from environment_variables import EnvironmentVariables

    X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN.value)
    y_train = wr.s3.read_csv(EnvironmentVariables.S3_Y_TRAIN.value)
    X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST.value)
    y_test = wr.s3.read_csv(EnvironmentVariables.S3_Y_TEST.value)

    return X_train, X_test, y_train, y_test

def mlflow_track_experiment(model, X):
    import mlflow
    import datetime
    from environment_variables import EnvironmentVariables
    from mlflow.models import infer_signature

    experiment = mlflow.set_experiment(EnvironmentVariables.MLFLOW_EXPERIMENT_NAME.value)

    mlflow.start_run(run_name='Challenger_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                     experiment_id=experiment.experiment_id,
                     tags={ 'experiment': 'challenger models', 'dataset': 'Stroke disease'},
                     log_system_metrics=True)

    params = model.get_params()
    params['model'] = type(model).__name__

    mlflow.log_params(params)

    artifact_path = 'model'

    signature = infer_signature(X, model.predict(X))

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=artifact_path,
        signature=signature,
        serialization_format='cloudpickle',
        registered_model_name='stroke_model_dev',
        metadata={ 'model_data_version': 1 }
    )

    # Obtain the model URI
    return mlflow.get_artifact_uri(artifact_path)

def register_challenger(model, f1_score, model_uri):
    import mlflow

    client = mlflow.MlflowClient()
    name = 'stroke_model_prod'

    # Save the model params as tags
    tags = model.get_params()
    tags['model'] = type(model).__name__
    tags['f1-score'] = f1_score

    # Save the version of the model
    result = client.create_model_version(
        name=name,
        source=model_uri,
        run_id=model_uri.split('/')[-3],
        tags=tags
    )

    # Save the alias as challenger
    client.set_registered_model_alias(name, 'challenger', result.version)

def train_the_challenger_model():
    import mlflow
    from environment_variables import EnvironmentVariables
    from sklearn.base import clone
    from sklearn.metrics import f1_score

    mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL)

    champion_model = load_the_champion_model()

    challenger_model = clone(champion_model)

    X_train, X_test, y_train, y_test = load_the_train_test_data()

    challenger_model.fit(X_train, y_train.to_numpy().ravel())

    y_pred = challenger_model.predict(X_test)
    f1_score = f1_score(y_test.to_numpy().ravel(), y_pred)

    artifact_uri = mlflow_track_experiment(challenger_model, X_train)

    register_challenger(challenger_model, f1_score, artifact_uri)