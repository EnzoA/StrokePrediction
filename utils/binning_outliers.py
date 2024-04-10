from utils.environment_variables import EnvironmentVariables

def binning_outliers():
        import awswrangler as wr
        import pandas as pd


        X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN)
        X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST)

        X_train['bmi'] = pd.cut(X_train['bmi'], bins = [0, 19, 25, 30, 500], labels = ['Underweight', 'Healthy', 'Overweight', 'Obese'])
        X_train['avg_glucose_level'] = pd.cut(X_train['avg_glucose_level'], bins = [0, 90, 170, 230, 500], labels = ['Low', 'Normal', 'Elevated', 'High'])  
        
        X_test['bmi'] = pd.cut(X_test['bmi'], bins = [0, 19, 25, 30, 500], labels = ['Underweight', 'Healthy', 'Overweight', 'Obese'])
        X_test['avg_glucose_level'] = pd.cut(X_test['avg_glucose_level'], bins = [0, 90, 170, 230, 500], labels = ['Low', 'Normal', 'Elevated', 'High'])

        wr.s3.to_csv(X_train, EnvironmentVariables.S3_X_TRAIN,index=False)
        wr.s3.to_csv(X_test, EnvironmentVariables.S3_X_TEST,index=False)
