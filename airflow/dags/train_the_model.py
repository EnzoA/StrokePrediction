from datetime import datetime
from airflow import DAG
from airflow.operators.python_operator import PythonVirtualenvOperator, BranchPythonOperator
from airflow.models.baseoperator import chain
from utils.decide_if_first_train import decide_if_first_train
from utils.train_the_model import train_the_model
from utils.train_the_challenger_model import train_the_challenger_model

with DAG(dag_id='train_the_model',
         start_date=datetime(2024, 1, 1),
         schedule_interval='@daily',
         catchup=False) as dag:
    
    decide_if_first_train_task = BranchPythonOperator(
        task_id='decide_if_first_train_task',
        python_callable=decide_if_first_train
    )

    train_the_model_task = PythonVirtualenvOperator(
        task_id='train_the_model',
        python_callable=train_the_model,
        requirements=[
            'scikit-learn==1.3.2',
            'mlflow==2.10.2',
            'awswrangler==3.6.0'],
        system_site_packages=True
    )

    train_the_challenger_model_task = PythonVirtualenvOperator(
        task_id='train_the_challenger_model',
        python_callable=train_the_challenger_model,
        requirements=[
            'scikit-learn==1.3.2',
            'mlflow==2.10.2',
            'awswrangler==3.6.0'],
        system_site_packages=True
    )

chain(
    decide_if_first_train_task,
    [train_the_model_task, train_the_challenger_model_task]
)