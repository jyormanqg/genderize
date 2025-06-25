"""
Main module for gender prediction functionality.

This module contains the LatamGenderize class which provides methods
to predict gender based on Latin American names using machine learning models.
"""

import os
import re
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import unidecode


class LatamGenderize:
    """
    A class for predicting gender based on Latin American names.
    
    This class uses a pre-trained machine learning model to predict
    the gender of individuals based on their names, with specific
    optimizations for Latin American naming conventions.
    
    Attributes:
        _model: The loaded TensorFlow model for gender prediction
        _model_path: Path to the model file
    """
    
    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialize the LatamGenderize instance.
        
        Args:
            model_path: Optional path to a custom model file. If not provided,
                       uses the default model included with the package.
        
        Raises:
            FileNotFoundError: If the model file cannot be found
            Exception: If there's an error loading the model
        """
        if model_path is None:
            # Get the directory where this package is installed
            package_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(package_dir, "models", "boyorgirl_CO_ES.h5")
        
        self._model_path = model_path
        self._model = self._load_model(model_path)
    
    def genderize(self, df: pd.DataFrame, name_column: Optional[str] = None) -> pd.DataFrame:
        """
        Predict gender for names in a DataFrame.
        
        Args:
            df: DataFrame containing names to predict gender for
            name_column: Name of the column containing the names. If None,
                        will attempt to auto-detect the column name.
        
        Returns:
            DataFrame with original data plus gender prediction columns:
            - gender_predicted: 'M' for male, 'F' for female
            - gender_probability: Confidence score (0.0 to 1.0)
        
        Raises:
            ValueError: If no valid name column is found
            Exception: If there's an error during prediction
        """
        if name_column is None:
            name_column = self._identify_column_name(df.columns.tolist())
        
        if name_column not in df.columns:
            raise ValueError(f"Column '{name_column}' not found in DataFrame")
        
        # Preprocess the data
        df_processed = self._preprocess(df, name_column)
        
        # Make predictions
        df_result = self._predict(df_processed)
        
        # Clean up temporary columns
        df_result.drop(['clean_name_nlp', 'transform_name_nlp'], axis=1, inplace=True)
        
        return df_result
    
    def _identify_column_name(self, columns: List[str]) -> str:
        """
        Automatically identify the name column from available columns.
        
        Args:
            columns: List of column names in the DataFrame
        
        Returns:
            The identified name column
        
        Raises:
            ValueError: If no suitable name column is found
        """
        normalized_columns = [col.lower() for col in columns]
        
        # Look for common name column patterns
        name_patterns = ['name', 'nombre', 'first_name', 'firstname', 'primer_nombre']
        
        for pattern in name_patterns:
            if pattern in normalized_columns:
                index = normalized_columns.index(pattern)
                return columns[index]
        
        raise ValueError(
            "No name column found. Please specify the name_column parameter. "
            "Supported column names: name, nombre, first_name, firstname, primer_nombre"
        )
    
    def _load_model(self, model_path: str):
        """
        Load the TensorFlow model from the specified path.
        
        Args:
            model_path: Path to the model file
        
        Returns:
            Loaded TensorFlow model
        
        Raises:
            FileNotFoundError: If the model file doesn't exist
            Exception: If there's an error loading the model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            return load_model(model_path)
        except Exception as e:
            raise Exception(f"Error loading model from {model_path}: {str(e)}") from e
    
    def _preprocess(self, df: pd.DataFrame, name_column: str) -> pd.DataFrame:
        """
        Preprocess names for model input.
        
        Args:
            df: DataFrame containing the names
            name_column: Name of the column containing names
        
        Returns:
            DataFrame with additional preprocessing columns
        """
        # Create a copy to avoid modifying the original
        df_copy = df.copy()
        
        # Clean names: remove special characters and normalize
        clean_names = df_copy[name_column].apply(
            lambda x: re.sub(r"[^a-z0-9 ]+", "", unidecode.unidecode(str(x).lower()))
        )
        
        # Transform names to character arrays
        transformed_names = [list(name) for name in clean_names]
        
        # Pad names to fixed length
        name_length = 50
        padded_names = [
            (name + [' '] * name_length)[:name_length] 
            for name in transformed_names
        ]
        
        # Encode characters to numbers (a=1, b=2, ..., space=0)
        encoded_names = [
            [max(0.0, ord(char) - 96.0) for char in name] 
            for name in padded_names
        ]
        
        # Add preprocessing columns
        df_copy['clean_name_nlp'] = clean_names
        df_copy['transform_name_nlp'] = encoded_names
        
        return df_copy
    
    def _predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Make gender predictions using the loaded model.
        
        Args:
            df: DataFrame with preprocessed name data
        
        Returns:
            DataFrame with prediction results
        """
        # Convert to numpy array for model input
        input_data = np.asarray(df['transform_name_nlp'].values.tolist())
        
        # Make predictions
        predictions = self._model.predict(input_data).squeeze(axis=1)
        
        # Convert predictions to gender labels and probabilities
        gender_labels = ['M' if prob > 0.5 else 'F' for prob in predictions]
        gender_probs = [
            prob if prob > 0.5 else 1.0 - prob 
            for prob in predictions
        ]
        
        # Add results to DataFrame
        df_result = df.copy()
        df_result['gender_predicted'] = gender_labels
        df_result['gender_probability'] = [round(prob, 2) for prob in gender_probs]
        
        return df_result 