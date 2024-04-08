def oversampling():
    def smote_oversampler(self):
        

        import awswrangler as wr
        import pandas as pd

        from imblearn.over_sampling import SMOTE

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                         path=path,
                         index=False)

        X_train = wr.s3.read_csv("s3://path_train")
        y_train = wr.s3.read_csv("s3://path_ytrain")

        smote = SMOTE(random_state=42)

        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
        y_train = pd.DataFrame(y_train_resampled, columns=y_train.columns)

        save_to_csv(X_train, "s3://data/X_train.csv")
        save_to_csv(y_train, "s3://data/y_train.csv")

        