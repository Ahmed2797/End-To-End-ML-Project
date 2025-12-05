import sys
import os
from typing import Tuple
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from neuro_mf import ModelFactory
from project.logger import logging
from project.exception import CustomException
from project.constants import *
from project.entity.artifacts import Data_Transformation_Artifact, Model_Trainer_Artifact, ClassificationMetricArtifact
from project.entity.config import Model_Trainer_Config
from project.entity.estimator import ProjectModel
from project.utils import load_numpy_array, load_object, save_object
import mlflow
import mlflow.sklearn
import joblib
import dagshub

dagshub.init(repo_owner='Ahmed2797', repo_name='End-To-End-ML-Project', mlflow=True)

class Model_Train:
    def __init__(self, data_transformation_artifact: Data_Transformation_Artifact,
                 model_trainer_config: Model_Trainer_Config):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def track_mlflow(self, best_model_path: str, metrics_artifact: ClassificationMetricArtifact):
        try:
            """Log metrics and model to MLflow."""
            mlflow.set_tracking_uri(self.model_trainer_config.mlflow_tracking_uri)
            mlflow.set_experiment(self.model_trainer_config.mlflow_experiment_name)

            with mlflow.start_run():
                mlflow.log_metric('f1_score', metrics_artifact.f1_score)
                mlflow.log_metric('precision_score', metrics_artifact.precision_score)
                mlflow.log_metric('accuracy_score', metrics_artifact.accuracy_score)
                mlflow.log_metric('recall_score', metrics_artifact.recall_score)

                # Log the model file
                mlflow.log_artifact(best_model_path)

                # Log the model as sklearn object
                try:
                    model_obj = joblib.load(best_model_path)
                    mlflow.sklearn.log_model(model_obj, 'model')
                except Exception as e:
                    logging.info(f"[WARNING] Failed to log model to MLflow/DagsHub: {e}")

        except Exception as e:
            raise CustomException(e, sys)

    def get_best_model_and_metrics(self, train: np.ndarray, test: np.ndarray) -> Tuple[object, ClassificationMetricArtifact]:
        try:
            logging.info("Initializing ModelFactory for best model search...")
            
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            logging.info(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}")
            logging.info(f"Unique classes in training: {np.unique(y_train)}")
            logging.info(f"Unique classes in test: {np.unique(y_test)}")

            model_factory = ModelFactory(model_config_path=self.model_trainer_config.param_yaml)
            best_model_details = model_factory.get_best_model(
                X=x_train, y=y_train, base_accuracy=self.model_trainer_config.excepted_score
            )

            model_obj = best_model_details.best_model
            y_pred = model_obj.predict(x_test)

            average_method = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
            metrics = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(y_test, y_pred),
                f1_score=f1_score(y_test, y_pred, average=average_method, zero_division=0),
                precision_score=precision_score(y_test, y_pred, average=average_method, zero_division=0),
                recall_score=recall_score(y_test, y_pred, average=average_method, zero_division=0)
            )
            print(metrics)

            # Save model temporarily to log in MLflow
            best_model_path = os.path.join(self.model_trainer_config.model_train_dir, "best_model.pkl")
            os.makedirs(self.model_trainer_config.model_train_dir, exist_ok=True)
            joblib.dump(model_obj, best_model_path)

            #self.track_mlflow(best_model_path, metrics)

            logging.info(f"Best model found: {type(model_obj).__name__} with accuracy {metrics.accuracy_score:.4f}")
            return best_model_details, metrics

        except Exception as e:
            raise CustomException(e, sys)

    def init_model(self) -> Model_Trainer_Artifact:
        try:
            logging.info("Starting model training process...")

            train_arr = load_numpy_array(self.data_transformation_artifact.transform_train_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transform_test_path)

            best_model_details, metrics = self.get_best_model_and_metrics(train_arr, test_arr)

            if best_model_details.best_score < self.model_trainer_config.excepted_score:
                raise Exception(f"No best model found exceeding expected score. Best: {best_model_details.best_score:.4f}, Expected: {self.model_trainer_config.excepted_score}")

            preprocessor_obj = load_object(self.data_transformation_artifact.preprocessing_pkl)

            prediction_model = ProjectModel(
                transform_object=preprocessor_obj,
                best_model_details=best_model_details.best_model
            )

            # Save final model
            os.makedirs(os.path.dirname(self.model_trainer_config.best_model_object), exist_ok=True)
            save_object(self.model_trainer_config.best_model_object, prediction_model)
            save_object(self.model_trainer_config.final_model_path, prediction_model)

            model_trainer_artifact = Model_Trainer_Artifact(
                trained_model_file_path=self.model_trainer_config.model_train_file_path,
                metric_artifact=metrics
            )

            logging.info("Model training and saving completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            raise CustomException(e, sys)
