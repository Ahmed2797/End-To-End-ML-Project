
from dataclasses import dataclass 
from project.constants import * 
from datetime import datetime 
import os


Timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

@dataclass 
class Project_Configuration:
    artifact: str = os.path.join(Artifact, Timestamp)
    pipeline: str = Pipeline_dir
    timestamp: str = Timestamp
    model_dir: str = os.path.join(final_model, Timestamp)


project_config = Project_Configuration()


# ================================================================
# DATA INGESTION CONFIG 
# Fix: train_path & test_path must NOT duplicate parent dir
# ================================================================
@dataclass 
class Data_Ingestion_Config:
    data_ingestion_dir = os.path.join(project_config.artifact, DATA_INGESTION_DIR)
    data_ingestion_feature_stored_dir = os.path.join(data_ingestion_dir, DATA_INGESTION_FEATURE_STORED_DIR)
    data_ingestion_ingested_dir = os.path.join(data_ingestion_dir, DATA_INGESTION_INGESTED_DIR)

    # FIXED PATHS
    train_path = os.path.join(data_ingestion_dir,data_ingestion_feature_stored_dir, Train_Data)
    test_path = os.path.join(data_ingestion_dir,data_ingestion_feature_stored_dir, Test_Data)

    split_ratio = DATA_INGESTION_SPLIT_RATIO 
    data_ingestion_collection_name = Collection_name 


# ================================================================
# DATA VALIDATION CONFIG
# ================================================================
@dataclass 
class Data_Validation_Config:
    data_validation_dir = os.path.join(project_config.artifact, DATA_VALIDATION_DIR)
    report_dir = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_DIR)
    report_status = os.path.join(data_validation_dir, DATA_VALIDATION_REPORT_STATUS)


# ================================================================
# DATA TRANSFORMATION CONFIG
# ================================================================
@dataclass
class Data_Transformation_Config:
    data_transformation_dir: str = os.path.join(project_config.artifact, DATA_TRANSFORMATION_DIR)
    
    transform_train_path: str = os.path.join(
        data_transformation_dir, 
        TRANSFORM_FILE, 
        Train_Data.replace('csv', 'npy')
    )
    transform_test_path: str = os.path.join(
        data_transformation_dir, 
        TRANSFORM_FILE, 
        Test_Data.replace('csv', 'npy')
    )
    transform_object_path: str = os.path.join(
        data_transformation_dir, 
        TRANSFORM_OBJECT, 
        PREPROCESSING_OBJECT
    )

    final_model_path: str = os.path.join(project_config.model_dir, PREPROCESSING_OBJECT)


# ================================================================
# MODEL TRAINER CONFIG
# Fix: best_model_object path corrected
# ================================================================
@dataclass 
class Model_Trainer_Config:
    model_train_dir: str = os.path.join(project_config.artifact, MODEL_TRAINER_DIR)
    model_train_file_path: str = os.path.join(model_train_dir, MODEL_TRAINER_FILE_PATH)

    # FIXED: best_model_object was incorrectly joined
    best_model_object: str = os.path.join(model_train_dir, BEST_MODEL_OBJECT)

    excepted_score: float = EXCEPTED_SCORE 
    param_yaml = PARAM_YAML_FILE

    mlflow_tracking_uri: str = 'https://dagshub.com/Ahmed2797/End-To-End-ML-Project.mlflow'
    mlflow_experiment_name: str = 'ML_mini'

    final_model_path: str = os.path.join(project_config.model_dir, BEST_MODEL_OBJECT)






'''
from dataclasses import dataclass 
from project.constants import * 
from datetime import datetime 
import os


Timestamp = datetime.now().strftime('%m_%d_%Y_%H_%M_%S')

@dataclass 
class Project_Configuration:
    artifact: str = os.path.join(Artifact, Timestamp)
    pipeline: str = Pipeline_dir
    timestamp: str = Timestamp
    model_dir: str = os.path.join(final_model,Timestamp)

project_config = Project_Configuration()


@dataclass 
class Data_Ingestion_Config:
    data_ingestion_dir = os.path.join(project_config.artifact,DATA_INGESTION_DIR)
    data_ingestion_feature_stored_dir = os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORED_DIR)
    data_ingestion_ingested_dir = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR)
    train_path = os.path.join(data_ingestion_dir,data_ingestion_feature_stored_dir,Train_Data)
    test_path = os.path.join(data_ingestion_dir,data_ingestion_feature_stored_dir,Test_Data)
    split_ratio = DATA_INGESTION_SPLIT_RATIO 
    data_ingestion_collection_name = Collection_name 


@dataclass 
class Data_Validation_Config:
    data_validation_dir = os.path.join(project_config.artifact,DATA_VALIDATION_DIR)
    report_dir = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_DIR)
    report_status = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_STATUS)



@dataclass
class Data_Transformation_Config:
    data_transformation_dir: str = os.path.join(project_config.artifact, DATA_TRANSFORMATION_DIR)
    transform_train_path: str = os.path.join(data_transformation_dir, TRANSFORM_FILE, Train_Data.replace('csv','npy'))
    transform_test_path: str = os.path.join(data_transformation_dir, TRANSFORM_FILE, Test_Data.replace('csv','npy'))
    transform_object_path: str = os.path.join(data_transformation_dir, TRANSFORM_OBJECT, PREPROCESSING_OBJECT)

    final_model_path:str = os.path.join(project_config.model_dir,PREPROCESSING_OBJECT)



@dataclass 
class Model_Trainer_Config:
    model_train_dir:str = os.path.join(project_config.artifact,MODEL_TRAINER_DIR)
    model_train_file_path:str = os.path.join(model_train_dir,MODEL_TRAINER_FILE_PATH)
    best_model_object:str = os.path.join(model_train_file_path,BEST_MODEL_OBJECT)
    excepted_score:float = EXCEPTED_SCORE 
    param_yaml = PARAM_YAML_FILE
    mlflow_tracking_uri: str = 'https://dagshub.com/Ahmed2797/End-To-End-ML-Project.mlflow'
    mlflow_experiment_name: str = 'ML_mini'

    final_model_path:str = os.path.join(project_config.model_dir,BEST_MODEL_OBJECT)

'''




