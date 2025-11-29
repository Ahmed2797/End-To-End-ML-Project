import sys
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from project.exception import CustomException

class ProjectModel:
    """
    A wrapper class for making predictions using a trained ML model 
    with preprocessing pipeline support.
    """
    def __init__(self, transform_object: Pipeline, best_model_details: BaseEstimator | object):
        """
        Args:
            transform_object (Pipeline): Preprocessing pipeline.
            best_model_details (BaseEstimator or object): Trained model or object containing .best_model.
        """
        if not isinstance(transform_object, Pipeline):
            raise ValueError("transform_object must be a scikit-learn Pipeline")
        self.transform_object = transform_object
        self.best_model_details = best_model_details

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms the input dataframe and returns model predictions as a DataFrame.

        Args:
            dataframe (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame: Predictions with column name 'prediction'.
        """
        try:
            if not isinstance(dataframe, pd.DataFrame):
                raise ValueError("Input must be a pandas DataFrame")

            # Transform input features
            transformed_features = self.transform_object.transform(dataframe)

            # Get the actual model
            model = getattr(self.best_model_details, "best_model", self.best_model_details)

            # Make predictions
            predictions = model.predict(transformed_features)

            # Return as DataFrame for consistency
            return pd.DataFrame(predictions, columns=['prediction'])

        except Exception as e:
            raise CustomException(e, sys)

    def __repr__(self):
        return (
            f"Network_model(model={type(self.best_model_details).__name__}, "
            f"transform={'Yes' if self.transform_object else 'No'})"
        )

    def __str__(self):
        return self.__repr__()
    
    
