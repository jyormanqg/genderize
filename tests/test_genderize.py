"""
Tests for the LatamGenderize module.
"""

import os
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock

import numpy as np
import pandas as pd

from latam_genderize import LatamGenderize


class TestLatamGenderize(unittest.TestCase):
    """Test cases for the LatamGenderize class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a mock model for testing
        self.mock_model = Mock()
        self.mock_model.predict.return_value = np.array([[0.7], [0.3], [0.8]])
        
        # Sample test data
        self.test_df = pd.DataFrame({
            'name': ['Juan', 'Maria', 'Carlos'],
            'age': [25, 30, 35]
        })

    def test_init_with_default_model(self):
        """Test initialization with default model path."""
        with patch('latam_genderize.genderize.load_model') as mock_load:
            mock_load.return_value = self.mock_model
            
            genderizer = LatamGenderize()
            
            # Check that load_model was called with the correct path
            expected_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), 
                '..', 'latam_genderize', 'models', 'boyorgirl_CO_ES.h5'
            )
            mock_load.assert_called_once()
            self.assertEqual(genderizer._model, self.mock_model)

    def test_init_with_custom_model(self):
        """Test initialization with custom model path."""
        custom_path = "/path/to/custom/model.h5"
        
        with patch('latam_genderize.genderize.load_model') as mock_load:
            mock_load.return_value = self.mock_model
            
            genderizer = LatamGenderize(model_path=custom_path)
            
            mock_load.assert_called_once_with(custom_path)
            self.assertEqual(genderizer._model_path, custom_path)

    def test_init_model_not_found(self):
        """Test initialization when model file is not found."""
        with patch('latam_genderize.genderize.os.path.exists') as mock_exists:
            mock_exists.return_value = False
            
            with self.assertRaises(FileNotFoundError):
                LatamGenderize()

    def test_identify_column_name_name(self):
        """Test automatic column identification with 'name' column."""
        genderizer = LatamGenderize.__new__(LatamGenderize)
        columns = ['id', 'name', 'age']
        
        result = genderizer._identify_column_name(columns)
        self.assertEqual(result, 'name')

    def test_identify_column_name_nombre(self):
        """Test automatic column identification with 'nombre' column."""
        genderizer = LatamGenderize.__new__(LatamGenderize)
        columns = ['id', 'nombre', 'edad']
        
        result = genderizer._identify_column_name(columns)
        self.assertEqual(result, 'nombre')

    def test_identify_column_name_first_name(self):
        """Test automatic column identification with 'first_name' column."""
        genderizer = LatamGenderize.__new__(LatamGenderize)
        columns = ['id', 'first_name', 'last_name']
        
        result = genderizer._identify_column_name(columns)
        self.assertEqual(result, 'first_name')

    def test_identify_column_name_not_found(self):
        """Test automatic column identification when no name column is found."""
        genderizer = LatamGenderize.__new__(LatamGenderize)
        columns = ['id', 'age', 'city']
        
        with self.assertRaises(ValueError):
            genderizer._identify_column_name(columns)

    def test_preprocess_names(self):
        """Test name preprocessing functionality."""
        genderizer = LatamGenderize.__new__(LatamGenderize)
        
        df = pd.DataFrame({'name': ['Juan-Pérez', 'María José', 'Carlos123']})
        result = genderizer._preprocess(df, 'name')
        
        # Check that preprocessing columns were added
        self.assertIn('clean_name_nlp', result.columns)
        self.assertIn('transform_name_nlp', result.columns)
        
        # Check that names were cleaned
        expected_clean = ['juanperez', 'maria jose', 'carlos123']
        self.assertEqual(result['clean_name_nlp'].tolist(), expected_clean)
        
        # Check that names were transformed to character arrays
        self.assertEqual(len(result['transform_name_nlp'].iloc[0]), 50)

    def test_predict_gender(self):
        """Test gender prediction functionality."""
        genderizer = LatamGenderize.__new__(LatamGenderize)
        genderizer._model = self.mock_model
        
        # Create test data with preprocessing columns
        df = pd.DataFrame({
            'name': ['Juan', 'Maria', 'Carlos'],
            'transform_name_nlp': [
                [1, 2, 3] + [0] * 47,  # Mock encoded name
                [4, 5, 6] + [0] * 47,
                [7, 8, 9] + [0] * 47
            ]
        })
        
        result = genderizer._predict(df)
        
        # Check that prediction columns were added
        self.assertIn('gender_predicted', result.columns)
        self.assertIn('gender_probability', result.columns)
        
        # Check predictions based on mock model output
        expected_genders = ['M', 'F', 'M']  # Based on [0.7, 0.3, 0.8]
        self.assertEqual(result['gender_predicted'].tolist(), expected_genders)

    def test_genderize_complete_flow(self):
        """Test the complete genderize workflow."""
        with patch('latam_genderize.genderize.load_model') as mock_load:
            mock_load.return_value = self.mock_model
            
            genderizer = LatamGenderize()
            
            # Test with sample data
            result = genderizer.genderize(self.test_df)
            
            # Check that original columns are preserved
            self.assertIn('name', result.columns)
            self.assertIn('age', result.columns)
            
            # Check that prediction columns were added
            self.assertIn('gender_predicted', result.columns)
            self.assertIn('gender_probability', result.columns)
            
            # Check that temporary columns were removed
            self.assertNotIn('clean_name_nlp', result.columns)
            self.assertNotIn('transform_name_nlp', result.columns)

    def test_genderize_with_specified_column(self):
        """Test genderize with explicitly specified column name."""
        with patch('latam_genderize.genderize.load_model') as mock_load:
            mock_load.return_value = self.mock_model
            
            genderizer = LatamGenderize()
            
            # Test with explicit column name
            result = genderizer.genderize(self.test_df, name_column='name')
            
            self.assertIn('gender_predicted', result.columns)
            self.assertIn('gender_probability', result.columns)

    def test_genderize_column_not_found(self):
        """Test genderize when specified column is not found."""
        with patch('latam_genderize.genderize.load_model') as mock_load:
            mock_load.return_value = self.mock_model
            
            genderizer = LatamGenderize()
            
            with self.assertRaises(ValueError):
                genderizer.genderize(self.test_df, name_column='nonexistent_column')

    def test_load_model_error(self):
        """Test error handling when loading model fails."""
        with patch('latam_genderize.genderize.load_model') as mock_load:
            mock_load.side_effect = Exception("Model loading failed")
            
            with self.assertRaises(Exception):
                LatamGenderize()


if __name__ == '__main__':
    unittest.main() 