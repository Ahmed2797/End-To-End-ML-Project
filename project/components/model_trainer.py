import sys
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from neuro_mf import ModelFactory

from project.logger import logging
from project.exception import CustomException
from project.constants import *
from project.entity.artifacts import (
    Data_Transformation_Artifact, 
    Model_Trainer_Artifact, 
    ClassificationMetricArtifact
)
from project.entity.config import Model_Trainer_Config
from project.entity.estimator import ProjectModel
from project.utils import load_numpy_array, load_object, save_object


class Model_Train:
    """
    Handles training, evaluation, and saving of the best model using preprocessed data.
    Fully compatible with the latest neuro_mf version.
    """

    def __init__(self, 
                 data_transformation_artifact: Data_Transformation_Artifact,
                 model_trainer_config: Model_Trainer_Config):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def get_best_model_and_metrics(self, train: np.ndarray, test: np.ndarray) -> Tuple[object, ClassificationMetricArtifact]:
        """
        Train models using ModelFactory and return best model along with classification metrics.

        Args:
            train (np.ndarray): Preprocessed training data (features + target)
            test (np.ndarray): Preprocessed test data (features + target)

        Returns:
            Tuple[object, ClassificationMetricArtifact]: Best model and metric artifact
        """
        try:
            logging.info("Initializing ModelFactory for best model search...")
            
            x_train, y_train = train[:, :-1], train[:, -1]
            x_test, y_test = test[:, :-1], test[:, -1]

            logging.info(f"Training data shape: {x_train.shape}, Test data shape: {x_test.shape}")
            logging.info(f"Unique classes in training: {np.unique(y_train)}")
            logging.info(f"Unique classes in test: {np.unique(y_test)}")

            # Initialize ModelFactory with error handling
            try:
                model_factory = ModelFactory(model_config_path=self.model_trainer_config.param_yaml)
                logging.info("ModelFactory initialized successfully")
            except Exception as e:
                logging.error(f"Error initializing ModelFactory: {str(e)}")
                raise CustomException(f"ModelFactory initialization failed: {str(e)}", sys)

            # Get best model with error handling
            try:
                best_model_details = model_factory.get_best_model(
                    X=x_train,
                    y=y_train,
                    base_accuracy=self.model_trainer_config.excepted_score
                )
                logging.info(f"Best model details obtained: {best_model_details}")
            except Exception as e:
                logging.error(f"Error in get_best_model: {str(e)}")
                raise CustomException(f"Model training failed: {str(e)}", sys)

            model_obj = best_model_details.best_model

            # Predict and calculate metrics
            y_pred = model_obj.predict(x_test)
            
            # Handle binary/multiclass metrics appropriately
            average_method = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
            
            metrics = ClassificationMetricArtifact(
                accuracy_score=accuracy_score(y_test, y_pred),
                f1_score=f1_score(y_test, y_pred, average=average_method, zero_division=0),
                precision_score=precision_score(y_test, y_pred, average=average_method, zero_division=0),
                recall_score=recall_score(y_test, y_pred, average=average_method, zero_division=0)
            )

            print(best_model_details.best_model)
            print(best_model_details.best_score)
            print(metrics)
            logging.info(f"Best model found: {type(model_obj).__name__} with accuracy {metrics.accuracy_score:.4f}")
            return best_model_details, metrics

        except Exception as e:
            logging.error(f"Error in get_best_model_and_metrics: {str(e)}")
            raise CustomException(e, sys)

    def init_model(self) -> Model_Trainer_Artifact:
        """
        Loads preprocessed data, trains the best model, wraps it with ProjectModel, 
        saves the prediction object, and returns the training artifact.
        """
        try:
            logging.info("Starting model training process...")
            
            # Load train and test arrays
            train_arr = load_numpy_array(self.data_transformation_artifact.transform_train_path)
            test_arr = load_numpy_array(self.data_transformation_artifact.transform_test_path)

            logging.info(f"Loaded train array shape: {train_arr.shape}")
            logging.info(f"Loaded test array shape: {test_arr.shape}")

            # Train and get best model + metrics
            best_model_details, metrics = self.get_best_model_and_metrics(train_arr, test_arr)

            # Check if best model meets expected score
            if best_model_details.best_score < self.model_trainer_config.excepted_score:
                logging.warning(f"No model meets expected score threshold. Best score: {best_model_details.best_score}, Expected: {self.model_trainer_config.excepted_score}")
                raise Exception(f"No best model found exceeding expected score. Best: {best_model_details.best_score:.4f}, Expected: {self.model_trainer_config.excepted_score}")

            # Load preprocessing pipeline
            preprocessor_obj = load_object(self.data_transformation_artifact.preprocessing_pkl)
            # if not isinstance(preprocessor_obj, Pipeline):
            #     preprocessor_obj = Pipeline([
            #         ('preprocessor', preprocessor_obj)
            #     ])

            # Wrap model with preprocessing for consistent prediction
            prediction_model = ProjectModel(
                transform_object=preprocessor_obj,
                best_model_details=best_model_details.best_model
            )

            # Save final prediction model
            save_object(self.model_trainer_config.best_model_object, prediction_model)
            save_object(self.model_trainer_config.final_model_path, prediction_model)

            # Return training artifact
            model_trainer_artifact = Model_Trainer_Artifact(
                trained_model_file_path=self.model_trainer_config.model_train_file_path,
                metric_artifact=metrics
            )

            logging.info("Model training and saving completed successfully.")
            return model_trainer_artifact

        except Exception as e:
            logging.error(f"Error in init_model: {str(e)}")
            raise CustomException(e, sys)
        
