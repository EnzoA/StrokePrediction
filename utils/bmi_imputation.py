from utils.environment_variables import EnvironmentVariables

def bmi_imputation():
        import awswrangler as wr
        from sklearn.impute import SimpleImputer

        X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN)
        imp_median = SimpleImputer(strategy='median')
        imp_median = imp_median.fit(X_train[['bmi']])
        X_train[['bmi']] = imp_median.transform(X_train[['bmi']])
        
        X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST)
        imp_median = SimpleImputer(strategy='median')
        imp_median = imp_median.fit(X_test[['bmi']])
        X_test[['bmi']] = imp_median.transform(X_test[['bmi']])
        

        wr.s3.to_csv(X_train, EnvironmentVariables.S3_X_TRAIN,index=False)
        wr.s3.to_csv(X_test, EnvironmentVariables.S3_X_TEST,index=False)

