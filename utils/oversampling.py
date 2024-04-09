def oversampling():
    def smote_oversampler(self):
        

        import awswrangler as wr
        import pandas as pd

        from imblearn.over_sampling import SMOTE

        from utils.environment_variables import EnvironmentVariables

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN)
        y_train = wr.s3.read_csv(EnvironmentVariables.S3_Y_TRAIN)

        smote = SMOTE(random_state=42)

        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        y_train = pd.DataFrame(y_train_resampled, columns=y_train.columns)

        save_to_csv(X_train, EnvironmentVariables.S3_X_TRAIN)
        save_to_csv(y_train, EnvironmentVariables.S3_Y_TRAIN)

        