def mlflow_track_experiment(model, X):
    from datetime import datetime
    import mlflow
    from mlflow.models import infer_signature
    from utils.environment_variables import EnvironmentVariables

    experiment = mlflow.set_experiment(EnvironmentVariables.MLFLOW_EXPERIMENT_NAME.value)

    mlflow.start_run(run_name='Challenger_run_' + datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
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
        registered_model_name=EnvironmentVariables.MLFLOW_MODEL_NAME_PROD.value,
        metadata={ 'model_data_version': 1 }
    )

    # Obtain the model URI
    return mlflow.get_artifact_uri(artifact_path)