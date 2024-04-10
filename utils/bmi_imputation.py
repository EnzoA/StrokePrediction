from utils.environment_variables import EnvironmentVariables

def bmi_imputation():
        import awswrangler as wr
        from sklearn.impute import SimpleImputer

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df,
                        path=path,
                        index=False)
        
        
        X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN)
        imp_median = SimpleImputer(strategy='median')
        imp_median = imp_median.fit(X_train[['bmi']])
        X_train[['bmi']] = imp_median.transform(X_train[['bmi']])
        
        save_to_csv(X_train, EnvironmentVariables.S3_X_TRAIN)
