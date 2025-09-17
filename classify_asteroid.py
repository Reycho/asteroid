import os
import numpy as np
import pandas as pd
import pickle
import argparse
import sys

# ==============================================================================
# Configuration
# ==============================================================================
MODEL_PATH = 'stable_classification_model.pkl'
SCALER_PATH = 'stable_classification_scaler.pkl'
LABEL_ENCODER_PATH = 'stable_classification_label_encoder.pkl'

# ==============================================================================
# Feature Extraction
# ==============================================================================
def extract_spectral_features(wavelength, reflectance):
    """Extract comprehensive spectral features."""
    features = {}

    features['mean_reflectance'] = np.mean(reflectance)
    features['std_reflectance'] = np.std(reflectance)
    features['min_reflectance'] = np.min(reflectance)
    features['max_reflectance'] = np.max(reflectance)

    if len(wavelength) > 1:
        slope, intercept = np.polyfit(wavelength, reflectance, 1)
        features['spectral_slope'] = slope

        vis_mask = (wavelength >= 0.5) & (wavelength <= 0.9)
        if np.sum(vis_mask) > 5:
            vis_slope, _ = np.polyfit(wavelength[vis_mask], reflectance[vis_mask], 1)
            features['visible_slope'] = vis_slope
        else:
            features['visible_slope'] = slope

    for band_center in [1.0, 2.0]:
        feature_name = f'band_{band_center}um_depth'
        features[feature_name] = 0.0

        if np.min(wavelength) <= band_center <= np.max(wavelength):
            idx_center = np.argmin(np.abs(wavelength - band_center))

            continuum_left_wl = band_center - 0.1
            continuum_right_wl = band_center + 0.1

            if np.min(wavelength) <= continuum_left_wl and np.max(wavelength) >= continuum_right_wl:
                idx_left = np.argmin(np.abs(wavelength - continuum_left_wl))
                idx_right = np.argmin(np.abs(wavelength - continuum_right_wl))

                if idx_left < idx_center < idx_right:
                    continuum_value = np.interp(wavelength[idx_center], [wavelength[idx_left], wavelength[idx_right]], [reflectance[idx_left], reflectance[idx_right]])

                    if continuum_value > 0:
                        features[feature_name] = max(0, 1 - (reflectance[idx_center] / continuum_value))

    expected_features = ['mean_reflectance', 'std_reflectance', 'min_reflectance', 'max_reflectance', 'spectral_slope', 'visible_slope', 'band_1.0um_depth', 'band_2.0um_depth']
    for f in expected_features:
        if f not in features:
            features[f] = 0.0

    return features

# ==============================================================================
# Main Classification Function
# ==============================================================================
def classify_spectrum(file_path, model, scaler, label_encoder):
    """Loads a spectrum, preprocesses it, and classifies it."""
    try:
        # Load the spectrum data
        spectrum = np.loadtxt(file_path, comments='#', usecols=(0, 1))
        if spectrum.ndim != 2 or spectrum.shape[1] != 2 or spectrum.shape[0] < 10:
            print("Error: Invalid spectrum file format.", file=sys.stderr)
            return

        wavelength, reflectance = spectrum[:, 0], spectrum[:, 1]

        # Clean and normalize the data
        mask = np.isfinite(wavelength) & np.isfinite(reflectance) & (reflectance > 0)
        wavelength, reflectance = wavelength[mask], reflectance[mask]

        if len(wavelength) < 10:
            print("Error: Not enough valid data points in spectrum.", file=sys.stderr)
            return

        median_reflectance = np.median(reflectance)
        if median_reflectance > 0:
            reflectance /= median_reflectance

        # Extract features
        features = extract_spectral_features(wavelength, reflectance)
        features_array = np.array(list(features.values())).reshape(1, -1)

        # Scale features
        scaled_features = scaler.transform(features_array)

        # Predict
        prediction_proba = model.predict_proba(scaled_features)
        prediction_index = np.argmax(prediction_proba)
        prediction_label = label_encoder.inverse_transform([prediction_index])[0]

        # Print results
        print(f"File: {os.path.basename(file_path)}")
        print(f"Predicted Asteroid Class: {prediction_label}\n")
        print("Class Probabilities:")
        for i, class_name in enumerate(label_encoder.classes_):
            print(f"  - {class_name}: {prediction_proba[0][i]:.4f}")

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Classify an asteroid from its spectral data file.')
    parser.add_argument('file_path', type=str, help='The path to the spectral data file (.txt or .dat).')
    args = parser.parse_args()

    # Load the trained model and artifacts
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(LABEL_ENCODER_PATH, 'rb') as f:
            label_encoder = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error: Could not load a required artifact: {e.filename}", file=sys.stderr)
        print("Please ensure you have run the `stable_trainer.py` script first to generate the model files.", file=sys.stderr)
        sys.exit(1)

    classify_spectrum(args.file_path, model, scaler, label_encoder)
