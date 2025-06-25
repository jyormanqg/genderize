"""
Basic usage example for the LatamGenderize package.

This example demonstrates how to use the package to predict gender
based on Latin American names.
"""

import pandas as pd
from latam_genderize import LatamGenderize


def main():
    """Demonstrate basic usage of the LatamGenderize package."""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'name': [
            'Juan Carlos',
            'María José',
            'Carlos Alberto',
            'Ana Sofía',
            'Luis Fernando',
            'Carmen Elena',
            'Roberto Carlos',
            'Isabella María'
        ],
        'age': [25, 30, 35, 28, 42, 29, 38, 26],
        'city': ['Bogotá', 'Medellín', 'Cali', 'Barranquilla', 'Cartagena', 'Bucaramanga', 'Pereira', 'Manizales']
    })
    
    print("Sample data:")
    print(sample_data)
    print("\n" + "="*50 + "\n")
    
    # Initialize the genderizer
    print("Initializing LatamGenderize...")
    genderizer = LatamGenderize()
    print("✓ LatamGenderize initialized successfully!")
    print("\n" + "="*50 + "\n")
    
    # Predict gender
    print("Predicting gender for names...")
    result = genderizer.genderize(sample_data)
    
    print("Results:")
    print(result[['name', 'gender_predicted', 'gender_probability']])
    print("\n" + "="*50 + "\n")
    
    # Show summary statistics
    print("Summary:")
    gender_counts = result['gender_predicted'].value_counts()
    print(f"Male predictions: {gender_counts.get('M', 0)}")
    print(f"Female predictions: {gender_counts.get('F', 0)}")
    
    avg_probability = result['gender_probability'].mean()
    print(f"Average confidence: {avg_probability:.2f}")
    
    # Show high confidence predictions
    high_confidence = result[result['gender_probability'] >= 0.8]
    print(f"\nHigh confidence predictions (≥80%): {len(high_confidence)}")
    if len(high_confidence) > 0:
        print(high_confidence[['name', 'gender_predicted', 'gender_probability']])


if __name__ == "__main__":
    main() 