from dataclasses import dataclass 
from project.constants import *


@dataclass 
class Data_Ingestion_Artifact:
    train_file_path:str 
    test_file_path:str


@dataclass 
class Data_Validation_Artifact:
    validation_status:bool
    message_error:str 
    drift_report_file_path:str


@dataclass
class Data_Transformation_Artifact:
    transform_train_path: str
    transform_test_path: str
    preprocessing_pkl: str


@dataclass
class ClassificationMetricArtifact:
    f1_score:float
    precision_score:float
    recall_score:float
    accuracy_score:float
@dataclass
class Model_Trainer_Artifact:
    trained_model_file_path:str 
    metric_artifact:ClassificationMetricArtifact
