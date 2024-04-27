def load_the_train_test_data():
    import awswrangler as wr
    from utils.environment_variables import EnvironmentVariables

    X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN.value)
    y_train = wr.s3.read_csv(EnvironmentVariables.S3_Y_TRAIN.value)
    X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST.value)
    y_test = wr.s3.read_csv(EnvironmentVariables.S3_Y_TEST.value)

    return X_train, X_test, y_train, y_test