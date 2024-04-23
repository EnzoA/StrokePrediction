def apply_smote_oversampling():
    import awswrangler as wr
    import pandas as pd

    from imblearn.over_sampling import SMOTE

    from utils.environment_variables import EnvironmentVariables

    X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN.value)
    y_train = wr.s3.read_csv(EnvironmentVariables.S3_Y_TRAIN.value)

    smote = SMOTE(random_state=42)

    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    X_train = pd.DataFrame(X_train_resampled, columns=X_train.columns)
    y_train = pd.DataFrame(y_train_resampled, columns=y_train.columns)

    wr.s3.to_csv(df=X_train, path=EnvironmentVariables.S3_X_TRAIN.value, index=False)
    wr.s3.to_csv(df=y_train, path=EnvironmentVariables.S3_Y_TRAIN.value, index=False)
