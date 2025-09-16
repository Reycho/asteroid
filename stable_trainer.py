import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings
import pickle

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
CATALOG_PATH = 'MITHNEOSCLEAN.csv'
SPECTRA_ZIP_PATH = 'visnir_files.zip'

# Output paths
MODEL_PATH = 'stable_classification_model.pkl'
SCALER_PATH = 'stable_classification_scaler.pkl'
PREDICTIONS_PATH = 'stable_classification_predictions.csv'
CONFUSION_MATRIX_PATH = 'stable_confusion_matrix.png'

# ==============================================================================
# Feature Extraction and Data Preparation
# ==============================================================================
def create_filename_map(spectra_zip_path):
    """Creates a robust filename mapping with multiple format support."""
    file_map = {}
    print(f"Scanning zip archive: {spectra_zip_path} to build filename map...")

    if not os.path.exists(spectra_zip_path):
        print(f"ERROR: Zip archive {spectra_zip_path} does not exist!")
        return file_map

    with zipfile.ZipFile(spectra_zip_path, 'r') as zip_ref:
        for filename in zip_ref.namelist():
            if (filename.endswith('.txt') or filename.endswith('.dat')) and not filename.startswith('__MACOSX'):
                try:
                    base_filename = os.path.basename(filename)
                    cleaned_name = base_filename.lower().replace('.visnir.txt', '').replace('.txt', '').replace('.dat', '').replace('a', '')
                    asteroid_id = int(cleaned_name)
                    file_map[asteroid_id] = filename
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse asteroid ID from filename: {filename}")
                    continue
    print(f"Successfully created map with {len(file_map)} entries.")
    return file_map

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
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    print("Starting Stable Asteroid Classification Trainer...")

    # Load and process data
    print("--- Loading and Processing Data ---")
    try:
        catalog_df = pd.read_csv(CATALOG_PATH)
        catalog_df.dropna(subset=['Number', 'Simplified Category'], inplace=True)
        catalog_df['Number'] = catalog_df['Number'].astype(int)
        print(f"Loaded catalog with {len(catalog_df)} entries.")
    except FileNotFoundError:
        sys.exit(f"FATAL: Catalog file not found at {CATALOG_PATH}")

    file_map = create_filename_map(SPECTRA_ZIP_PATH)
    if not file_map:
        sys.exit("FATAL: No spectral files found or mapped. Exiting.")

    all_features = []
    all_labels = []

    with zipfile.ZipFile(SPECTRA_ZIP_PATH, 'r') as zip_f:
        for _, row in catalog_df.iterrows():
            asteroid_id = row['Number']
            if asteroid_id in file_map:
                filename = file_map[asteroid_id]
                try:
                    with zip_f.open(filename) as f:
                        spectrum = np.loadtxt(f, comments='#', usecols=(0, 1))

                    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or spectrum.shape[0] < 10:
                        continue

                    wavelength, reflectance = spectrum[:, 0], spectrum[:, 1]
                    mask = np.isfinite(wavelength) & np.isfinite(reflectance) & (reflectance > 0)
                    wavelength, reflectance = wavelength[mask], reflectance[mask]

                    if len(wavelength) < 10:
                        continue

                    median_reflectance = np.median(reflectance)
                    if median_reflectance > 0:
                        reflectance /= median_reflectance

                    features = extract_spectral_features(wavelength, reflectance)
                    all_features.append(list(features.values()))
                    all_labels.append(row['Simplified Category'])

                except Exception as e:
                    print(f"Warning: Error processing {filename} for asteroid {asteroid_id}: {e}")
                    continue

    print(f"Successfully processed {len(all_labels)} spectra.")
    if not all_labels:
        sys.exit("FATAL: No data was successfully processed. Exiting.")

    X = np.array(all_features)
    y_str = np.array(all_labels)

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    X_train, X_test, y_train, y_test, y_str_train, y_str_test = train_test_split(
        X, y, y_str, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter Tuning with GridSearchCV
    print("\n--- Tuning RandomForestClassifier with GridSearchCV ---")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    model = grid_search.best_estimator_
    print("--- Model training complete ---")

    # Evaluate the model
    print("\n--- Evaluating model ---")
    y_pred = model.predict(X_test_scaled)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"Confusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # Save the model, scaler, and label encoder
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"Trained model saved to {MODEL_PATH}")

    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Feature scaler saved to {SCALER_PATH}")

    with open('stable_classification_label_encoder.pkl', 'wb') as f:
        pickle.dump(le, f)
    print("Label encoder saved to stable_classification_label_encoder.pkl")

    # Generate predictions for the entire dataset
    print("\n--- Generating predictions for all data ---")
    X_scaled = scaler.transform(X)
    full_predictions = model.predict(X_scaled)
    full_probabilities = model.predict_proba(X_scaled)

    results_df = pd.DataFrame({
        'Original_Taxonomy': y_str,
        'Predicted_Taxonomy': le.inverse_transform(full_predictions)
    })

    for i, class_label in enumerate(le.classes_):
        results_df[f'Prob_{class_label}'] = full_probabilities[:, i]

    results_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"Predictions saved to {PREDICTIONS_PATH}")

    print("\nScript finished successfully.")
