from project.logger import logging                         
from project.exception import CustomException 
from project.entity.config import Data_Validation_Config 
from project.entity.artifacts import Data_Validation_Artifact, Data_Ingestion_Artifact
from project.utils import read_yaml, write_yaml_file
from project.constants import COLUMN_YAML_FILE_PATH
from evidently import Report 
from evidently.presets import DataDriftPreset
import pandas as pd 
import sys
import json


class Data_Validation:
    def __init__(self, data_validation_config: Data_Validation_Config,
                       data_ingestion_artifact: Data_Ingestion_Artifact):
        """
        Initializes Data_Validation with configuration and ingestion artifacts.
        """
        try:
            self.validation_config = data_validation_config
            self.data_ingestion_artifact = data_ingestion_artifact
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)
            logging.info("Data Validation Initialized successfully.")
            
        except Exception as e:
            raise CustomException(e, sys)

    def no_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Check if the number of columns in dataframe matches expected schema.
        """
        try:
            expected_columns = self._column_schema['columns']
            status = len(expected_columns) == len(dataframe.columns)
            logging.info(f"Expected Columns: {len(expected_columns)}, Found: {len(dataframe.columns)}")
            logging.info(f"Column count match status: {status}")

            return status
        except Exception as e:
            raise CustomException(e, sys)

    def is_columns_existed(self, dataframe: pd.DataFrame) -> bool:
        """
        Check if all required numeric and categorical columns exist in dataframe.
        """
        try:
            # Numeric columns
            missing_numeric_columns = [
                col for col in self._column_schema['numerical_columns']
                if col not in dataframe.columns
            ]

            # Categorical columns
            missing_categorical_columns = [
                col for col in self._column_schema['categorical_columns']
                if col not in dataframe.columns
            ]

            if missing_numeric_columns:
                logging.warning(f"Missing numeric columns: {missing_numeric_columns}")

            if missing_categorical_columns:
                logging.warning(f"Missing categorical columns: {missing_categorical_columns}")

            status = not (missing_numeric_columns or missing_categorical_columns)
            logging.info(f"Columns existence check status: {status}")

            return status
        except Exception as e:
            raise CustomException(e, sys)

    def drift_dataset_detect(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> bool:
        """
        Detect drift between train (reference) and test (current) datasets using Evidently.
        """
        try:
            drift_report = Report(metrics=[DataDriftPreset()])
            report = drift_report.run(reference_data=reference_df, current_data=current_df)

            report_json = report.json()
            report_dict = json.loads(report_json)

            # Save drift report as YAML
            write_yaml_file(
                file_path=self.validation_config.report_dir,
                content=report_dict
            )

            # Extract drift information
            n_features = sum(1 for m in report_dict["metrics"] if "ValueDrift" in m["metric_id"])
            drift_metric = next(
                (m for m in report_dict["metrics"] if "DriftedColumnsCount" in m["metric_id"]), None
            )

            n_feature_detect = drift_metric["value"]["count"] if drift_metric else 0
            drift_status = n_feature_detect > 0

            logging.info(f"{n_feature_detect}/{n_features} features show drift.")
            logging.info(f"Drift Detection Status: {drift_status}")

            return drift_status
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Read CSV file as pandas DataFrame.
        """
        try:
            df = pd.read_csv(file_path)
            logging.info(f"Data loaded successfully from {file_path} with shape {df.shape}")

            return df
        except Exception as e:
            raise CustomException(e, sys)

    def init_data_validation(self) -> Data_Validation_Artifact:
        """
        Initialize and run all validation checks.
        """
        try:
            logging.info("===== Starting Data Validation Process =====")

            train_data = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_data = self.read_data(self.data_ingestion_artifact.test_file_path)

            validation_error_message = []

            # Train dataset checks
            if not self.no_of_columns(train_data):
                validation_error_message.append("Error: Train data column count mismatch.")
            if not self.is_columns_existed(train_data):
                validation_error_message.append("Error: Missing columns in train dataset.")

            # Test dataset checks
            if not self.no_of_columns(test_data):
                validation_error_message.append("Error: Test data column count mismatch.")
            if not self.is_columns_existed(test_data):
                validation_error_message.append("Error: Missing columns in test dataset.")

            validation_status = len(validation_error_message) == 0

            # Perform data drift detection if schema validation passed
            if validation_status:
                drift_status = self.drift_dataset_detect(train_data, test_data)
                if drift_status:
                    validation_error_message.append("Error: Data drift detected between train and test datasets.")
                    validation_status = False
                else:
                    validation_error_message.append("Success: No drift detected between train and test datasets.")
            else:
                logging.warning(f"Schema validation failed: {validation_error_message}")

            # Create validation artifact
            data_validation_artifact = Data_Validation_Artifact(
                validation_status=validation_status,
                message_error=validation_error_message,
                drift_report_file_path=self.validation_config.report_dir
            )

            logging.info(f"Data Validation Artifact created successfully: {data_validation_artifact}")
            logging.info("===== Data Validation Completed =====")

            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)
        


