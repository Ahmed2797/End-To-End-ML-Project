from project.entity.config import Data_Transformation_Config 
from project.entity.artifacts import (
    Data_Ingestion_Artifact,
    Data_Validation_Artifact,
    Data_Transformation_Artifact
)
from project.constants import (
    DATA_TRANSFORMATION_IMPUTER_PARAMS,
    COLUMN_YAML_FILE_PATH,
    Target_Column
)
from project.utils import read_yaml, save_object,save_numpy_array
from project.exception import CustomException 
from project.logger import logging

from sklearn.preprocessing import StandardScaler, OneHotEncoder, PowerTransformer
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 
from sklearn.impute import SimpleImputer, KNNImputer 
from imblearn.combine import SMOTEENN 

import pandas as pd 
import numpy as np 
import os 
import sys


class Data_Transformation:
    def __init__(self, 
                    data_transformation_config: Data_Transformation_Config,
                    data_ingestion_artifact: Data_Ingestion_Artifact,
                    data_validation_artifact: Data_Validation_Artifact):
        try:      
            self.transformation_config = data_transformation_config
            self.ingestion_artifact = data_ingestion_artifact
            self.validation_artifact = data_validation_artifact
            self._column_schema = read_yaml(COLUMN_YAML_FILE_PATH)

        except Exception as e:
            raise CustomException(e, sys)

    # Apply log transform before pipeline
    def apply_log_transform_to_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply log transform to specific columns before pipeline"""
        try:
            df_transformed = df.copy()
            log_columns = self._column_schema['log_transform_col']
            
            # Ensure log_columns is a list (even if single column)
            if isinstance(log_columns, str):
                log_columns = [log_columns]
            
            # Apply log1p to each specified column
            for col in log_columns:
                if col in df_transformed.columns:
                    df_transformed[col] = np.log1p(df_transformed[col].clip(lower=0))
                    logging.info(f'Applied log transform to column: {col}')
            
            return df_transformed
        except Exception as e:
            raise CustomException(f"Error in apply_log_transform_to_columns: {e}", sys)


    # Simple Preprocessing Pipeline
    def get_data_transformation(self) -> ColumnTransformer:
        try:
            # Numeric pipeline
            numeric_pipeline = Pipeline(steps=[
                ("imputer", KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)),
                ("scaler", StandardScaler())
            ])

            # Categorical pipeline
            categorical_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", drop='first'))
            ])

            # Transformer pipeline
            transform_pipeline = Pipeline(steps=[
                ('transformer', PowerTransformer(method='yeo-johnson'))
            ])

            
            # Combine pipelines
            preprocessor = ColumnTransformer(transformers=[
                ("num", numeric_pipeline, self._column_schema['numerical_columns']),
                ("cat", categorical_pipeline, self._column_schema['categorical_columns']),
                ('transform',transform_pipeline,self._column_schema['log_transform_col'])
            ])

            preprocessor_ = Pipeline([
                ("preprocessor", preprocessor)
            ])

            logging.info('Preprocessing Object Created Successfully')
            return preprocessor_
        except Exception as e:
            raise CustomException(f"Error in get_data_transformation: {e}", sys)


    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
        
    @staticmethod
    def target_value_mapping(y):
        try:
            mapping = {'Yes': 1, 'No': 0}
            return y.map(mapping)
        except Exception as e:
            raise CustomException(f"Error in target_value_mapping: {e}", sys)


    # Main transformation flow
    def initiate_data_transformation(self):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(self.ingestion_artifact.train_file_path)
            test_df = pd.read_csv(self.ingestion_artifact.test_file_path)

            # Drop unwanted columns
            drop_cols = self._column_schema['drop_columns']
            print('Drop_Columns:----',drop_cols)
            train_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            test_df.drop(columns=drop_cols, inplace=True, errors='ignore')
            print('actural_data_shape:--------',train_df.shape)

            # Apply Log Transform BEFORE pipeline
            # train_df = self.apply_log_transform_to_columns(train_df)
            # test_df = self.apply_log_transform_to_columns(test_df)

            # Separate target
            target_col = Target_Column
            X_train = train_df.drop(columns=[target_col])
            y_train = train_df[target_col]
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]

            # Target mapping
            # y_train = self.target_value_mapping(y_train)
            # y_test = self.target_value_mapping(y_test)

            # Preprocessor pipeline
            preprocessor = self.get_data_transformation()
            X_train_trans = preprocessor.fit_transform(X_train)
            X_test_trans = preprocessor.transform(X_test)

            # Apply SMOTE 
            logging.info("Applying SMOTE to reduces Imbalanced_Data")
            smt = SMOTEENN(sampling_strategy='minority',random_state=42)
            x_train_resample, y_train_resampled = smt.fit_resample(X_train_trans,y_train) 
            #x_test_resample, y_test = smt.fit_resample(X_test_trans,y_test) 
            print('*smote**apply*_data_shape:--------',x_train_resample.shape)

            logging.info(f"Resampled training shape: {x_train_resample.shape}, {y_train_resampled.shape}")

            # For test set we keep original distribution (only transform, do not resample)
            x_test_final = X_test_trans
            y_test_final = y_test

            # Combine & save arrays
            train_arr = np.c_[x_train_resample, np.array(y_train_resampled)]
            test_arr = np.c_[x_test_final, np.array(y_test_final)]

            save_numpy_array(self.transformation_config.transform_train_path, train_arr)
            save_numpy_array(self.transformation_config.transform_test_path, test_arr)

            # Save preprocessor object
            save_object(self.transformation_config.transform_object_path, preprocessor)
            save_object(self.transformation_config.final_model_path, preprocessor)

            # Return artifact
            data_transformation_artifact = Data_Transformation_Artifact(
                transform_train_path=self.transformation_config.transform_train_path,
                transform_test_path=self.transformation_config.transform_test_path,
                preprocessing_pkl=self.transformation_config.transform_object_path
            )
            logging.info("Data transformation completed successfully")

            return data_transformation_artifact
        except Exception as e:
            raise CustomException(f"Error in initiate_data_transformation: {e}", sys)
        

