from airflow import DAG
from airflow.operators.python_operator import PythonVirtualenvOperator
from datetime import datetime
from utils.get_raw_dataset import get_raw_dataset
from utils.set_one_hot_encoding_variables import set_one_hot_encoding_variables
from utils.map_yes_no_encoding_variables import map_yes_no_encoding_variables
from utils.apply_standard_scaling import apply_standard_scaling
from utils.apply_smote_oversampling import apply_smote_oversampling
from utils.split_dataset import split_dataset
from utils.impute_bmi import impute_bmi
from utils.map_outliers_to_bins import map_outliers_to_bins

with DAG(dag_id='etl_process_dag',
         start_date=datetime(2024, 1, 1),
         schedule_interval='@daily',
         catchup=False) as dag:

        get_raw_dataset_task = PythonVirtualenvOperator(
              task_id='get_raw_dataset',
              python_callable=get_raw_dataset,
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        dataset_split_task = PythonVirtualenvOperator(
              task_id='split_dataset',
              python_callable=split_dataset,
              requirements=['awswrangler==3.6.0',
                            'scikit-learn==1.3.2'],
              system_site_packages=True
        )

        impute_bmi_task = PythonVirtualenvOperator(
              task_id='bmi_data_imputation',
              python_callable=impute_bmi,
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        map_outliers_to_bins_task = PythonVirtualenvOperator(
              task_id='map_outliers_to_bins', 
              python_callable=map_outliers_to_bins,
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        set_one_hot_encoding_variables_task = PythonVirtualenvOperator(
              task_id='set_one_hot_encoding_variables', 
              python_callable=set_one_hot_encoding_variables,
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        map_yes_no_encoding_variables_task = PythonVirtualenvOperator(
              task_id='map_yes_no_encoding_variables',
              python_callable=map_yes_no_encoding_variables,
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        apply_standard_scaling_task = PythonVirtualenvOperator(
              task_id='apply_standard_scaling',
              python_callable=apply_standard_scaling,
              requirements=['awswrangler==3.6.0',
                            'scikit-learn==1.3.2',
                            'mlflow==2.10.2'],
              system_site_packages=True
        )

        apply_smote_oversampling_task = PythonVirtualenvOperator(
              task_id='apply_smote_oversampling',
              python_callable=apply_smote_oversampling,
              requirements=['awswrangler==3.6.0',
                            'imbalanced-learn==0.11.0'],
              system_site_packages=True
        )

(get_raw_dataset_task
 >> dataset_split_task
 >> impute_bmi_task
 >> map_outliers_to_bins_task
 >> set_one_hot_encoding_variables_task
 >> map_yes_no_encoding_variables_task
 >> apply_standard_scaling_task
 >> apply_smote_oversampling_task)
