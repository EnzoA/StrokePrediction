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