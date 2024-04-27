def evaluate_champion_challenge():
        import mlflow
        import awswrangler as wr

        from sklearn.metrics import f1_score

        from utils.environment_variables import EnvironmentVariables

        mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL.value)

        def load_the_model(alias):
            model_name = EnvironmentVariables.MLFLOW_MODEL_NAME.value

            client = mlflow.MlflowClient()
            model_data = client.get_model_version_by_alias(model_name, alias)

            model = mlflow.sklearn.load_model(model_data.source)

            return model

        def load_the_test_data():
            X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST.value)
            y_test = wr.s3.read_csv(EnvironmentVariables.S3_Y_TEST.value)

            return X_test, y_test

        def promote_challenger(name):

            client = mlflow.MlflowClient()

            # Demote the champion
            client.delete_registered_model_alias(name, "champion")

            # Load the challenger from registry
            challenger_version = client.get_model_version_by_alias(name, "challenger")

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

            # Transform in champion
            client.set_registered_model_alias(name, "champion", challenger_version.version)

        def demote_challenger(name):

            client = mlflow.MlflowClient()

            # delete the alias of challenger
            client.delete_registered_model_alias(name, "challenger")

        # Load the champion model
        champion_model = load_the_model("champion")

        # Load the challenger model
        challenger_model = load_the_model("challenger")

        # Load the dataset
        X_test, y_test = load_the_test_data()

        # Obtain the metric of the models
        y_pred_champion = champion_model.predict(X_test)
        f1_score_champion = f1_score(y_test.to_numpy().ravel(), y_pred_champion)

        y_pred_challenger = challenger_model.predict(X_test)
        f1_score_challenger = f1_score(y_test.to_numpy().ravel(), y_pred_challenger)

        experiment = mlflow.set_experiment(EnvironmentVariables.MLFLOW_EXPERIMENT_NAME.value)

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_metric("test_f1_challenger", f1_score_challenger)
            mlflow.log_metric("test_f1_champion", f1_score_champion)

            if f1_score_challenger > f1_score_champion:
                mlflow.log_param("Winner", 'Challenger')
            else:
                mlflow.log_param("Winner", 'Champion')

        name = EnvironmentVariables.MLFLOW_MODEL_NAME.value
        if f1_score_challenger > f1_score_champion:
            promote_challenger(name)
        else:
            demote_challenger(name)