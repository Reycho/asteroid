# Enhanced Asteroid Classification System

## Overview

This system implements a state-of-the-art machine learning pipeline for classifying asteroids based on their spectral data. It uses visible and near-infrared (VISNIR) reflectance spectra to determine asteroid taxonomic types, which are crucial for understanding asteroid composition, origin, and potential resources.

## Table of Contents
- [How It Works](#how-it-works)
- [Features](#features)
- [Data Requirements](#data-requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output Files](#output-files)
- [Model Architecture](#model-architecture)
- [Performance Metrics](#performance-metrics)
- [Troubleshooting](#troubleshooting)

## How It Works

### 1. **Data Loading and Preprocessing**

The system begins by loading two critical data sources:

- **Catalog File** (`MITHNEOSCLEAN.csv`): Contains asteroid metadata including identification numbers and taxonomic classifications
- **Spectral Data** (`visnir_files.zip`): A compressed archive containing individual spectrum files for each asteroid

The preprocessing pipeline:
1. Matches asteroid IDs from the catalog to their corresponding spectrum files
2. Loads wavelength-reflectance pairs from spectrum files
3. Filters out invalid data points (NaN, negative values)
4. Normalizes reflectance values by dividing by the median reflectance

### 2. **Advanced Feature Extraction**

The system extracts 46 sophisticated features from each spectrum, grouped into several categories:

#### Statistical Features (10 features)
- Basic statistics: mean, standard deviation, min, max, median, range
- Distribution shape: skewness, kurtosis
- Percentiles: 25th and 75th percentiles

#### Spectral Slope Analysis (9 features)
- **Overall spectral slope**: Linear trend across entire wavelength range
- **Visible slope** (0.5-0.9 μm): Captures blue-to-red gradient
- **Near-IR slope** (0.9-1.5 μm): Measures near-infrared behavior
- **Extended NIR slope** (1.5-2.5 μm): Captures longer wavelength trends
- **Polynomial residuals**: Quantifies non-linearity of the spectrum

#### Absorption Band Analysis (21 features)
Analyzes 7 key absorption features at wavelengths critical for mineral identification:
- 0.7 μm: Hydrated minerals
- 0.9 μm: Olivine/pyroxene (Band I)
- 1.0 μm: Olivine/pyroxene (Band I center)
- 1.25 μm: Feldspar
- 1.9 μm: Water/hydroxyl
- 2.0 μm: Pyroxene (Band II)
- 2.3 μm: Hydroxyl compounds

For each band, the system calculates:
- **Depth**: How deep the absorption is relative to the continuum
- **Center**: Actual wavelength of minimum reflectance
- **Width**: Full width at half maximum (FWHM) of the absorption

#### Spectral Curvature Features (3 features)
- **Mean curvature**: Average second derivative
- **Maximum curvature**: Peak second derivative
- **Spectral roughness**: Variability in first derivative

#### Color Indices (3 features)
- **Blue-Red index** (0.44/0.70 μm ratio)
- **Green-Red index** (0.55/0.70 μm ratio)
- **NIR-Red ratio** (0.85/0.70 μm ratio)

### 3. **Class Balancing Techniques**

To handle the imbalanced nature of asteroid taxonomy data:

#### Data Augmentation
For minority classes with fewer than 10 samples:
- Adds Gaussian noise (σ = 2% of signal)
- Applies random scaling (95-105%)
- Creates 2 synthetic samples per real sample

#### SMOTE (Synthetic Minority Over-sampling Technique)
- Generates synthetic samples by interpolating between existing minority class samples
- Combined with Tomek links to remove borderline samples
- Creates a more balanced training dataset

### 4. **Ensemble Learning Architecture**

The system uses a weighted voting ensemble of four complementary algorithms:

#### Random Forest Classifier (Weight: 2.0)
- **Hyperparameters optimized**: n_estimators (200-500), max_depth (10-30), min_samples_split/leaf
- **Strengths**: Handles non-linear relationships, feature importance ranking
- **Role**: Primary classifier due to robustness

#### Gradient Boosting Classifier (Weight: 1.5)
- **Hyperparameters optimized**: n_estimators (100-300), learning_rate (0.01-0.1), max_depth (3-10)
- **Strengths**: Sequential error correction, high accuracy
- **Role**: Refined predictions through boosting

#### Support Vector Machine (Weight: 1.0)
- **Configuration**: RBF kernel, C=10, gamma='scale'
- **Strengths**: Effective in high-dimensional space
- **Role**: Non-linear boundary detection

#### Neural Network (MLP) (Weight: 1.0)
- **Architecture**: 3 layers (100, 50, 25 neurons)
- **Strengths**: Complex pattern recognition
- **Role**: Captures subtle spectral patterns

### 5. **Training Process**

1. **Data Splitting**: 80/20 train-test split with stratification
2. **Scaling**: RobustScaler (resistant to outliers) normalizes features
3. **Hyperparameter Optimization**: RandomizedSearchCV with 5-fold cross-validation
4. **Model Training**: Each model trained independently, then combined
5. **Ensemble Creation**: Soft voting aggregates probability predictions

## Features

### Key Capabilities
- ✅ Processes hundreds of spectra automatically
- ✅ Handles varying wavelength ranges and resolutions
- ✅ Robust to noise and outliers
- ✅ Provides probability estimates for each class
- ✅ Generates confidence scores for predictions
- ✅ Creates visualizations of results

### Advanced Techniques
- 🔬 46 scientifically-motivated spectral features
- 🎯 Multiple class balancing strategies
- 🤖 4-model ensemble for robust predictions
- 📊 Comprehensive evaluation metrics
- 🎨 Feature importance analysis

## Data Requirements

### Input Files

1. **Catalog File** (`MITHNEOSCLEAN.csv`)
   ```
   Columns required:
   - Number: Integer asteroid ID
   - Simplified Category: Taxonomic classification (e.g., 'S-Group', 'C-Group')
   ```

2. **Spectral Data Archive** (`visnir_files.zip`)
   ```
   File format: Text files with two columns
   - Column 1: Wavelength (μm)
   - Column 2: Reflectance (normalized or raw)
   
   Naming convention: 
   - {asteroid_id}.txt or {asteroid_id}.visnir.txt
   - Example: "433.txt" for asteroid 433 Eros
   ```

### Data Quality Requirements
- Minimum 10 wavelength points per spectrum
- Wavelength range should ideally cover 0.5-2.5 μm
- At least 3 samples per class for training

## Installation

### Dependencies
```python
# Core libraries
numpy >= 1.19.0
pandas >= 1.2.0
scipy >= 1.6.0

# Machine learning
scikit-learn >= 0.24.0
imbalanced-learn >= 0.8.0

# Visualization
matplotlib >= 3.3.0
seaborn >= 0.11.0

# System
pickle (standard library)
zipfile (standard library)
warnings (standard library)
```

### Installation Steps
```bash
# 1. Install required packages
pip install numpy pandas scipy scikit-learn imbalanced-learn matplotlib seaborn

# 2. Prepare your data
# Place MITHNEOSCLEAN.csv and visnir_files.zip in asteroid/ directory

# 3. Run the classifier
python enhanced_asteroid_classifier.py
```

## Usage

### Basic Usage
```python
# The script runs automatically when executed
python enhanced_asteroid_classifier.py
```

### Using the Trained Model
```python
import pickle
import numpy as np

# Load the trained model and preprocessors
with open('enhanced_classification_model.pkl', 'rb') as f:
    model = pickle.load(f)
    
with open('enhanced_classification_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
    
with open('enhanced_classification_label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
    
with open('feature_extractor.pkl', 'rb') as f:
    extract_features = pickle.load(f)

# Process a new spectrum
wavelength = np.array([...])  # Your wavelength data
reflectance = np.array([...])  # Your reflectance data

# Extract features
features_dict = extract_features(wavelength, reflectance)
features = np.array(list(features_dict.values())).reshape(1, -1)

# Scale features
features_scaled = scaler.transform(features)

# Make prediction
prediction = model.predict(features_scaled)[0]
probabilities = model.predict_proba(features_scaled)[0]

# Decode prediction
predicted_class = label_encoder.inverse_transform([prediction])[0]
confidence = np.max(probabilities)

print(f"Predicted class: {predicted_class} (confidence: {confidence:.2%})")
```

## Output Files

### Model Files
- `enhanced_classification_model.pkl`: Trained ensemble classifier
- `enhanced_classification_scaler.pkl`: Feature scaling transformer
- `enhanced_classification_label_encoder.pkl`: Class label encoder
- `feature_extractor.pkl`: Feature extraction function

### Results Files
- `enhanced_classification_predictions.csv`: Predictions for all samples
  ```csv
  Original_Taxonomy, Predicted_Taxonomy, Prob_C-Group, Prob_S-Group, ..., Confidence, Prediction_Correct
  ```

### Visualization Files
- `enhanced_confusion_matrix.png`: Confusion matrix heatmap
- `feature_importance.png`: Top 20 most important features

## Model Architecture

```
Input Spectrum (wavelength, reflectance)
           ↓
    Feature Extraction (46 features)
           ↓
     RobustScaler Normalization
           ↓
    ┌──────┴──────┬──────┬──────┐
    ↓             ↓      ↓      ↓
Random Forest  Gradient  SVM   MLP
(weight: 2.0)  Boosting       Neural
              (wt: 1.5)       Network
    ↓             ↓      ↓      ↓
    └──────┬──────┴──────┴──────┘
           ↓
    Soft Voting Ensemble
           ↓
    Class Probabilities
           ↓
    Final Prediction
```

## Performance Metrics

### Evaluation Metrics
- **Balanced Accuracy**: Accounts for class imbalance
- **F1 Scores**: Weighted and macro averages
- **Precision/Recall**: Per-class performance
- **Confusion Matrix**: Visual error analysis

### Expected Performance
- Overall balanced accuracy: >0.85
- S-Group (majority class): >0.90 F1 score
- C-Group: >0.75 F1 score
- Minority classes: >0.50 F1 score

### Confidence Scores
Each prediction includes a confidence score (0-1):
- >0.8: High confidence
- 0.6-0.8: Moderate confidence
- <0.6: Low confidence (consider manual review)

## Troubleshooting

### Common Issues and Solutions

#### ValueError: inhomogeneous shape
**Cause**: Inconsistent feature vector lengths
**Solution**: Ensures using the updated version with `get_feature_names()`

#### Low performance on minority classes
**Solutions**:
- Increase augmentation multiplier for rare classes
- Adjust class weights in classifiers
- Collect more samples of minority classes
- Consider merging similar rare classes

#### Memory issues with large datasets
**Solutions**:
- Process spectra in batches
- Reduce n_iter in RandomizedSearchCV
- Use fewer estimators in ensemble

#### No spectral files found
**Check**:
- Zip file path is correct
- File naming convention matches expectation
- Zip file isn't corrupted

### Performance Optimization

#### For Faster Training
- Reduce hyperparameter search iterations
- Use fewer models in ensemble
- Decrease n_estimators in forest models
- Use parallel processing (n_jobs=-1)

#### For Better Accuracy
- Increase training data through augmentation
- Add more hyperparameter options
- Include additional spectral features
- Use cross-validation for threshold tuning

## Scientific Background

### Asteroid Taxonomy
Asteroid classification is based on spectral characteristics that indicate surface composition:

- **S-Group**: Silicaceous, stony composition (most common in inner belt)
- **C-Group**: Carbonaceous, primitive composition (most common overall)
- **X-Group**: Metallic composition (includes M, E, P types)
- **V-type**: Basaltic, associated with Vesta family
- **D-type**: Very red, organic-rich (outer belt/Trojans)

### Why Spectral Features Matter
- **Absorption bands**: Direct indicators of mineral composition
- **Spectral slopes**: Related to space weathering and composition
- **Color indices**: Correlate with taxonomic types
- **Curvature**: Indicates mixture of materials

## Future Enhancements

### Planned Improvements
- [ ] Deep learning models (CNN on raw spectra)
- [ ] Transfer learning from lunar/meteorite data
- [ ] Active learning for uncertain samples
- [ ] Multi-wavelength data fusion (thermal IR)
- [ ] Hierarchical classification (complex types)

### Contributing
To improve the model:
1. Add more labeled training data
2. Implement new feature extraction methods
3. Test alternative ML algorithms
4. Optimize hyperparameters further

## Citation

If you use this classification system in your research, please cite:
```
Enhanced Asteroid Classification System
[Your Name/Organization]
Version 2.0, 2024
https://github.com/[your-repo]
```

## License

[Specify your license here]

## Contact

For questions, issues, or contributions:
- Email: [your-email]
- GitHub: [your-github]
- Issues: [repo-issues-link]

---

*Last updated: 2024*
