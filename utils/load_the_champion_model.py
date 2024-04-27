def load_the_champion_model():
    import mlflow;
    from utils.environment_variables import EnvironmentVariables

    try:
        mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL.value)
        client = mlflow.MlflowClient()
        model_name = f'{EnvironmentVariables.MLFLOW_MODEL_NAME.value}_prod'
        alias = 'champion'
        model_data = client.get_model_version_by_alias(model_name, alias)
        champion_version = mlflow.sklearn.load_model(model_data.source)
        return champion_version
    except mlflow.exceptions.MlflowException as ex:
        return None
