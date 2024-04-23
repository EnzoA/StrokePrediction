import pandas as pd
from airflow.decorators import dag, task
from airflow.operators.python_operator import PythonVirtualenvOperator
from utils.environment_variables import EnvironmentVariables


    

def map_outliers_to_bins(s3_path):
        import awswrangler as wr
        import pandas as pd
        
        # Read data from S3
        data = wr.s3.read_csv(s3_path)
        
        # Process the dataframe
        data['bmi'] = pd.cut(data['bmi'], bins=[0, 19, 25, 30, 500], labels=['Underweight', 'Healthy', 'Overweight', 'Obese'])
        data['avg_glucose_level'] = pd.cut(data['avg_glucose_level'], bins=[0, 90, 170, 230, 500], labels=['Low', 'Normal', 'Elevated', 'High'])  
        
        # Write the processed dataframe back to S3
        wr.s3.to_csv(data, s3_path, index=False)
    
    # Define tasks for processing training and test datasets
    
    
    



#def map_outliers_to_bins():
 #       import awswrangler as wr
 #       import pandas as pd
 #       from utils.environment_variables import EnvironmentVariables

 #       X_train = wr.s3.read_csv(EnvironmentVariables.S3_X_TRAIN)
 #       X_test = wr.s3.read_csv(EnvironmentVariables.S3_X_TEST)

  #      X_train['bmi'] = pd.cut(X_train['bmi'], bins = [0, 19, 25, 30, 500], labels = ['Underweight', 'Healthy', 'Overweight', 'Obese'])
  #      X_train['avg_glucose_level'] = pd.cut(X_train['avg_glucose_level'], bins = [0, 90, 170, 230, 500], labels = ['Low', 'Normal', 'Elevated', 'High'])  
        
   #     X_test['bmi'] = pd.cut(X_test['bmi'], bins = [0, 19, 25, 30, 500], labels = ['Underweight', 'Healthy', 'Overweight', 'Obese'])
   #     X_test['avg_glucose_level'] = pd.cut(X_test['avg_glucose_level'], bins = [0, 90, 170, 230, 500], labels = ['Low', 'Normal', 'Elevated', 'High'])

   #     wr.s3.to_csv(X_train, EnvironmentVariables.S3_X_TRAIN,index=False)
   #     wr.s3.to_csv(X_test, EnvironmentVariables.S3_X_TEST,index=False)
