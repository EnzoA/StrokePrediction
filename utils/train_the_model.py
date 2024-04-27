def train_the_model():
    # Librerías
    import mlflow
    from mlflow.models import infer_signature
    from datetime import datetime
    
    # Modelos
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV

    # Selección de features
    from sklearn.feature_selection import VarianceThreshold
    
    # Métricas
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        classification_report,
        confusion_matrix,
        ConfusionMatrixDisplay,
        roc_auc_score, roc_curve)

    # Utils locales
    from utils.get_or_create_experiment import get_or_create_experiment
    from utils.load_the_train_test_data import load_the_train_test_data
    from utils.mlflow_track_experiment import mlflow_track_experiment
    from utils.environment_variables import EnvironmentVariables

    mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL.value)
    experiment_id = get_or_create_experiment(EnvironmentVariables.MLFLOW_EXPERIMENT_NAME.value)
    run_name_parent = 'best_hyperparam_'  + datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name_parent, nested=True):
        X_train, X_test, y_train, y_test = load_the_train_test_data()

        logistic_regression_params = {
            'penalty': ['l1', 'l2'],
            'C': [0.5, 0.75, 1, 1.5],
            'solver': ['lbfgs', 'liblinear', 'saga'],
            'max_iter': [100, 150, 300],
            'n_jobs': [-1]
        }

        model = GridSearchCV(
            estimator=LogisticRegression(),
            cv=5,
            param_grid=logistic_regression_params
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_train)

        artifact_path = 'model'
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=artifact_path,
            signature=signature,
            serialization_format='cloudpickle',
            registered_model_name=EnvironmentVariables.MLFLOW_MODEL_NAME.value + '_dev',
            metadata={ 'model_data_version': 1 }
        )
        model_uri = mlflow.get_artifact_uri(artifact_path)

    client = mlflow.MlflowClient()
    registered_model_name = f'{EnvironmentVariables.MLFLOW_MODEL_NAME.value}_prod'
    client.create_registered_model(name=registered_model_name, description='This classifier detects if a person will have a stroke or not')

    tags = model.get_params()
    tags['model'] = type(model).__name__
    tags['f1-score'] = f1_score

    model_version = client.create_model_version(
        name=registered_model_name,
        source=model_uri,
        run_id=model_uri.split('/')[-3],
        tags=tags
    )

    client.set_registered_model_alias(registered_model_name, 'champion', model_version.version)
