# Stroke Prediction
<h3>Modelo de ML y API REST para la predicción de accidentes cerebrovasculares</h3>

Según la Organización Mundial de la Salud (OMS), el accidente cerebrovascular es la segunda causa de muerte a nivel mundial y es responsable de aproximadamente el 11% del total de muertes. Se entrena un modelo para predecir si es probable que un paciente sufra un accidente cerebrovascular en función de parámetros de entrada como el sexo, la edad, diversas enfermedades y el tabaquismo.

Link al dataset usado para el trabajo: [https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset)

A continuación, se detallan las columnas del dataset:

1- id-Patient ID
2- gender-Gender of Patient
3- age-Age of Patient
4- hypertension-0 - no hypertension, 1 - suffering from hypertension
5- heart_disease-0 - no heart disease, 1 - suffering from heart disease
6- ever_married-Yes/No
7- work_type-Type of occupation
8- Residence_type-Area type of residence (Urban/ Rural)
9- avg_glucose_level-Average Glucose level (measured after meal)
10- bmi-Body mass index
11- smoking_status-patient’s smoking status
12- stroke-0 - no stroke, 1 - suffered stroke


La TP incluye:

- En Apache Airflow

    -DAG que obtiene los datos del repositorio, realiza limpieza y 
     feature engineering, y guarda en el bucket `s3://data` los datos separados para entrenamiento 
     y pruebas. MLflow hace seguimiento de este procesamiento.
     ![MINIO](Minio.png)
    -DAG de entrenamiento y reentrenamiento con la capacidad de detectar si hay modelo registrado, si existe entrena y registra el modelo inicial
     de lo contrario hace reentrenamiento y registro del nuevo modelo en caso de que las metricas sean mejores que el modelo actual, todo con registro del 
     proceso en MLFLOW.
     
- Un servicio de API del modelo, que toma el artefacto de MLflow y lo expone para realizar predicciones.
    ![FasTAPI](FastAPI.png)

Diagrama de la implementacion
![Arquitectura](Arquitectura.png)


## Revision de  Funcionamiento

Los pasos para revisar el funcionamiento son los siguientes:

1. Al momento de tener arriba el sistema multicontenedor, ejecuta en Airflow el DAG 
llamado `etl_process.py`, de esta manera se crearán los datos en el 
bucket `s3://data`.
2. Ejecuta la notebook (ubicada en `train_the_model.py`) para realizar el entrenamiento 
del modelo inicial.
![DAG](DAG_airflow.png)
3. Utiliza el servicio de API.

Además, una vez entrenado el modelo, puedes ejecutar el DAG `train_the_model.py` para probar 
un nuevo modelo que compita con el modelo actual(Champion). Antes de hacer esto, ejecuta el DAG 
`etl_process.py` para que el conjunto de datos sea nuevo.

### API 

Podemos realizar predicciones utilizando la API, accediendo a `http://localhost:8800/`.

Para hacer una predicción, debemos enviar una solicitud al endpoint `Predict` con un 
cuerpo de tipo JSON que contenga un campo de características (`features`) con cada 
entrada para el modelo.

Un ejemplo utilizando `curl` sería:

```bash
curl -X 'POST' \
  'http://localhost:8800/predict/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "features": {
   "age": 67,
    "avg_glucose_level": 210,
    "bmi": 81,
    "ever_married": true,
    "gender": "female",
    "heart_disease": true,
    "hypertension": true,
    "residence_type": "Urban",
    "smoking_status": "formerly smoked",
    "work_type": "Private"
  }
}'
```

La respuesta del modelo será un valor booleano y un mensaje en forma de cadena de texto que 
indicará si el paciente tiene o no una enfermedad cardiaca.

```json
{
  "int_output": true,
  "str_output": "Stroke detected"
}
```

Para obtener más detalles sobre la API, ingresa a `http://localhost:8800/docs`.