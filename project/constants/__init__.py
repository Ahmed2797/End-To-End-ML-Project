import os
import numpy as np


# MongoDB
Data_Base_Name = 'Customer_Churn'
Collection_name = 'Churn'
#Collection_name = 'Churn_Modelling'
MONGODB_URL_KEY = 'MONGODB_URL' 


# Artifacts 
Artifact = 'artifact'
Pipeline_dir = 'pipeline'
final_model = 'final_model'


# Data
Raw_Data = 'raw.csv'
Train_Data = 'train.csv'
Test_Data = 'test.csv' 

# Target_column
Target_Column = 'Exited'


# yaml_file
COLUMN_YAML_FILE_PATH = os.path.join('yaml_file','columns.yaml')
PARAM_YAML_FILE = os.path.join('yaml_file','param.yaml')

# DataIngestion
DATA_INGESTION_DIR:str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORED_DIR:str = 'feature'
DATA_INGESTION_INGESTED_DIR:str = 'ingested'
DATA_INGESTION_SPLIT_RATIO:float = 0.2
Collection_name:str = 'Churn' 


# Data_validation 
DATA_VALIDATION_DIR:str = 'data_validation'
DATA_VALIDATION_REPORT_DIR:str = 'drift_report'
DATA_VALIDATION_REPORT_STATUS:str = 'report.yaml' 


# Data_Transformation
DATA_TRANSFORMATION_DIR = "data_transform"
TRANSFORM_FILE = "transform"
TRANSFORM_OBJECT = "transform_obj"
PREPROCESSING_OBJECT = "preprocessing.pkl"

DATA_TRANSFORMATION_IMPUTER_PARAMS = {
    "n_neighbors": 3,
    "weights": "uniform",
    "missing_values": np.nan
} 


# Model Trainer 
MODEL_TRAINER_DIR = 'model_trainer'
MODEL_TRAINER_FILE_PATH = 'best_model'
BEST_MODEL_OBJECT = 'best_model.pkl' 
EXCEPTED_SCORE:float = 0.7






