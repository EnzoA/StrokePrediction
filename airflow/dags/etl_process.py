from airflow import DAG
from airflow.operators.python_operator import PythonVirtualenvOperator
from utils.get_raw_dataset import get_raw_dataset
from utils.set_one_hot_encoding_variables import set_one_hot_encoding_variables
from utils.map_yes_no_encoding_variables import map_yes_no_encoding_variables
from utils.apply_standard_scaling import apply_standard_scaling
from utils.apply_smote_oversampling import apply_smote_oversampling
from utils.split_dataset import split_dataset
from utils.impute_bmi import impute_bmi
from utils.map_outliers_to_bins import map_outliers_to_bins
from utils.environment_variables import EnvironmentVariables
from airflow.models.baseoperator import chain

with DAG(dag_id='etl_process_dag',
         schedule_interval=None,
         max_active_runs=1,
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

        map_outliers_to_bins_training_task= PythonVirtualenvOperator(
              task_id='map_outliers_to_bins_training_data',
              python_callable=map_outliers_to_bins,
              op_kwargs={'s3_path': EnvironmentVariables.S3_X_TRAIN.value},
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True

        )

        map_outliers_to_bins_testing_task = PythonVirtualenvOperator(
              task_id='map_outliers_to_bins_testing_data',
              python_callable=map_outliers_to_bins,
              op_kwargs={'s3_path': EnvironmentVariables.S3_X_TEST.value},
              requirements=["awswrangler"],
              system_site_packages=True

        )

        set_one_hot_encoding_training_variables_task = PythonVirtualenvOperator(
              task_id='one_hot_encoding_training_variables', 
              python_callable=set_one_hot_encoding_variables,
              op_kwargs={'s3_path': EnvironmentVariables.S3_X_TRAIN.value},
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        set_one_hot_encoding_testing_variables_task = PythonVirtualenvOperator(
              task_id='one_hot_encoding_testing_variables', 
              python_callable=set_one_hot_encoding_variables,
              op_kwargs={'s3_path': EnvironmentVariables.S3_X_TEST.value},
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        map_yes_no_encoding_training_variables_task = PythonVirtualenvOperator(
              task_id='yes_no_encoding_training_variables', 
              python_callable=map_yes_no_encoding_variables,
              op_kwargs={'s3_path': EnvironmentVariables.S3_X_TRAIN.value, 'dataset_type': 'training'},
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
        )

        map_yes_no_encoding_testing_variables_task = PythonVirtualenvOperator(
              task_id='yes_no_encoding_testing_variables', 
              python_callable=map_yes_no_encoding_variables,
              op_kwargs={'s3_path': EnvironmentVariables.S3_X_TEST.value, 'dataset_type': 'testing'},
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


chain(get_raw_dataset_task, 
      dataset_split_task, 
      impute_bmi_task, 
      [map_outliers_to_bins_training_task, map_outliers_to_bins_testing_task], 
      [set_one_hot_encoding_training_variables_task, set_one_hot_encoding_testing_variables_task],
      [map_yes_no_encoding_training_variables_task, map_yes_no_encoding_testing_variables_task],
      apply_standard_scaling_task, 
      apply_smote_oversampling_task
      )

