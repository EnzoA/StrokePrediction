from airflow import DAG
from airflow.operators.python import PythonVirtualenvOperator
from datetime import datetime
from utils.one_hot_encoding_task import set_one_hot_encoding_variables
from utils.yes_no_encoding_task import map_yes_no_encoding_variables
from utils.standard_scaler import standard_scaler
from utils.oversampling import smote_oversampler


with DAG(dag_id="etl_process_dag",
         start_date=datetime(2024,1,1),
         schedule_interval="@daily",
         catchup=False) as dag:

        dataset_split = PythonVirtualenvOperator(
              task_id = "train_test_dataset_split",
              python_callable = some_function #TODO: reemplazar por la funcion verdadera
            
        )    
        
        bmi_imputation = PythonVirtualenvOperator(
              task_id="bmi_data_imputation",
              python_callable = some_other_function 
        )

        outliers = PythonVirtualenvOperator(
              task_id="binning_outliers", 
              python_callable = some_other_function_2
        )

        one_hot_encoding = PythonVirtualenvOperator(
              task_id="set_one_hot_encoding_variables", 
              python_callable = set_one_hot_encoding_variables,
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True
              
        )

        yes_no_encoding = PythonVirtualenvOperator(
              task_id="map_yes_no_encoding_variables",
              python_callable = map_yes_no_encoding_variables,
              requirements=['awswrangler==3.6.0'],
              system_site_packages=True

        )

        data_scaling = PythonVirtualenvOperator(
              task_id="normalize_features",
              python_callable=standard_scaler,
              requirements=["awswrangler==3.6.0",
                            "scikit-learn==1.3.2",
                            "mlflow==2.10.2"],
              system_site_packages=True

        )

        smote_oversampling = PythonVirtualenvOperator(
              task_id="data_train_oversampling",
              python_callable=smote_oversampler,
              requirements=["awswrangler==3.6.0",
                            "imbalanced-learn==0.11.0"],
              system_site_packages=True

        )


dataset_split >> bmi_imputation >> outliers >> one_hot_encoding >> yes_no_encoding >> data_scaling >> smote_oversampling