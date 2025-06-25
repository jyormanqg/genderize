# Models Directory

This directory contains the pre-trained machine learning models for gender prediction.

## Model Files

Place your `.h5` model files in this directory. The default model expected by the package is:

- `boyorgirl_CO_ES.h5` - Default model for Colombian and Spanish names

## Adding Custom Models

To use a custom model, you can:

1. Place your `.h5` file in this directory
2. Pass the model path when initializing `LatamGenderize`:

```python
from latam_genderize import LatamGenderize

# Use custom model
genderizer = LatamGenderize(model_path="path/to/your/model.h5")
```

## Model Format

The models should be TensorFlow/Keras models saved in `.h5` format and trained to predict gender based on character-level name encoding. 