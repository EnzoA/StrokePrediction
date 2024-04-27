def train_the_challenger_model():
    import mlflow
    from sklearn.base import clone
    from sklearn.metrics import f1_score
    from utils.environment_variables import EnvironmentVariables
    from utils.load_the_champion_model import load_the_champion_model
    from utils.load_the_train_test_data import load_the_train_test_data
    from utils.mlflow_track_experiment import mlflow_track_experiment

    def register_challenger(model, f1_score, model_uri):
        client = mlflow.MlflowClient()
        name = EnvironmentVariables.MLFLOW_MODEL_NAME_PROD.value

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

    mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL.value)

    champion_model = load_the_champion_model()

    challenger_model = clone(champion_model)

    X_train, X_test, y_train, y_test = load_the_train_test_data()

    challenger_model.fit(X_train, y_train.to_numpy().ravel())

    y_pred = challenger_model.predict(X_test)
    f1_score = f1_score(y_test.to_numpy().ravel(), y_pred)

    artifact_uri = mlflow_track_experiment(challenger_model, X_train)

    register_challenger(challenger_model, f1_score, artifact_uri)