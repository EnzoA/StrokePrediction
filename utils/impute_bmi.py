def impute_bmi():
    import awswrangler as wr
    from sklearn.impute import SimpleImputer
    from utils.environment_variables import EnvironmentVariables

    X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN.value)
    imp_median = SimpleImputer(strategy='median')
    imp_median = imp_median.fit(X_train[['bmi']])
    X_train[['bmi']] = imp_median.transform(X_train[['bmi']])
    
    X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST.value)
    X_test[['bmi']] = imp_median.transform(X_test[['bmi']])

    wr.s3.to_csv(X_train, EnvironmentVariables.S3_X_TRAIN.value,index=False)
    wr.s3.to_csv(X_test, EnvironmentVariables.S3_X_TEST.value,index=False)

