# type: ignore
import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated
from utils.environment_variables import EnvironmentVariables
from models.work_type import WorkType
from models.smoking_status import SmokingStatus
from models.residence_type import ResidenceType
from models.gender import Gender

def load_model(model_name: str, alias: str):
    '''
    Loads a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    '''

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL)
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open(EnvironmentVariables.MODEL_PKL_LOCAL_PATH, 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON)
        result_s3 = s3.get_object(Bucket='data', Key=EnvironmentVariables.S3_DATA_JSON)
        text_s3 = result_s3['Body'].read().decode()
        data_dictionary = json.loads(text_s3)

        # TODO: Se necesita esto? Estará disponible?
        data_dictionary['standard_scaler_mean'] = np.array(data_dictionary['standard_scaler_mean'])
        data_dictionary['standard_scaler_std'] = np.array(data_dictionary['standard_scaler_std'])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open(EnvironmentVariables.DATA_JSON_LOCAL_PATH, 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary

def check_model():
    '''
    Checks for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    '''

    global model
    global data_dict
    global version_model

    try:
        model_name = 'stroke_prediction_model_prod'
        alias = 'champion'

        mlflow.set_tracking_uri(EnvironmentVariables.MLFLOW_BASE_URL)
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass

class ModelInput(BaseModel):
    '''
    Input schema for the stroke prediction model.

    This class defines the input fields required by the heart disease prediction model along with their descriptions
    and validation constraints.

    :param age: Age of the patient (0 to 150).
    :param hypertension: Whether the patient has hypertension or not.
    :param heart_disease: Whether the patient has a heart disease or not.
    :param ever_married: Whether the patient has ever been married or not.
    :param gender: The patient's gender: male or female.
    :param work_type: The patient's work type: private, self_employed, children, govt_job or never_worked.
    :param smoking_status: The patient's smoking status: never_smoked, unknown, formerly_smoked or smokes.
    :param residence_type: The patient's residence type: urban or rural.
    :param avg_glucose_level: The average glucose level the patient has.
    :param bmi: The patient's body mass index.
    '''

    age: int = Field(
        description='Age of the patient',
        ge=0,
        le=150,
    )
    hypertension: bool = Field(
        description='Whether the patient has hypertension or not'
    )
    heart_disease: bool = Field(
        description='Whether the patient has a heart disease or not'
    )
    ever_married: bool = Field(
        description='Whether the patient has ever been married or not'
    )
    gender: Gender = Field(
        description="The patient's gender: male or female"
    )
    work_type: WorkType = Field(
        description="The patient's work type: private, self_employed, children, govt_job or never_worked"
    )
    smoking_status: SmokingStatus = Field(
        description="The patient's smoking status: never_smoked, unknown, formerly_smoked or smokes"
    )
    residence_type: ResidenceType = Field(
        description="The patient's residence type: urban or rural"
    )
    avg_glucose_level: float = Field(
        description='The average glucose level the patient has',
        ge=0,
        le=300,
    )
    bmi: float = Field(
        description="The patient's body mass index",
        ge=0.0,
        le=100.0,
    )

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'age': 67,
                    'hypertension': True,
                    'heart_disease': True,
                    'ever_married': True,
                    'gender': Gender.female,
                    'work_type': WorkType.private,
                    'smoking_status': SmokingStatus.formerly_smoked,
                    'residence_type': ResidenceType.urban,
                    'avg_glucose_level': 210.0,
                    'bmi': 81
                }
            ]
        }
    }

class ModelOutput(BaseModel):
    '''
    Output schema for the stroke prediction model.

    This class defines the output fields returned by the stroke prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. True if the patient has a stroke.
    :param str_output: Output of the model in string form. Can be "Healthy patient" or "Stroke detected".
    '''

    int_output: bool = Field(
        description='Output of the model. True if the patient has a stroke',
    )
    str_output: Literal['Healthy patient', 'Stroke detected'] = Field(
        description='Output of the model in string form',
    )

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'int_output': True,
                    'str_output': 'Stroke detected',
                }
            ]
        }
    }

# Load the model before start
model, version_model, data_dict = load_model('stroke_prediction_model_prod', 'champion')

app = FastAPI()

@app.get('/')
async def read_root():
    '''
    Root endpoint of the Stroke Prediction Detector API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    '''
    return JSONResponse(content=jsonable_encoder({ 'message': 'Welcome to the Stroke Prediction Detector API' }))

@app.post('/predict/', response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    '''
    Endpoint for predicting strokes.

    This endpoint receives features related to a patient's health and predicts whether the patient has a stroke
    or not using a trained model. It returns the prediction result in both integer and string formats.
    '''

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    # TODO: Actualizar acorde al dataset.

    # Process categorical features
    for categorical_col in data_dict['categorical_columns']:
        features_df[categorical_col] = features_df[categorical_col].astype(int)
        categories = data_dict['categories_values_per_categorical'][categorical_col]
        features_df[categorical_col] = pd.Categorical(features_df[categorical_col], categories=categories)

    # Convert categorical features into dummy variables
    features_df = pd.get_dummies(data=features_df,
                                 columns=data_dict['categorical_columns'],
                                 drop_first=True)

    # Reorder DataFrame columns
    features_df = features_df[data_dict['columns_after_dummy']]

    # Scale the data using standard scaler
    features_df = (features_df-data_dict['standard_scaler_mean'])/data_dict['standard_scaler_std']

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = 'Healthy patient'
    if prediction[0] > 0:
        str_pred = 'Stroke detected'

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
