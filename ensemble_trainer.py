import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
base_path = '.'
catalog_path = os.path.join(base_path, 'MITHNEOSCLEAN.csv')
catalog_path_new = os.path.join(base_path, 'catalog_for_training.csv')
spectra_zip_path = os.path.join(base_path, 'visnir_files.zip')
spectra_zip_path_new = os.path.join(base_path, 'final_spectra_for_training-20250912T220022Z-1-001.zip')
output_predictions_path = os.path.join(base_path, 'nea_classification_predictions_sklearn.csv')

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
                    if base_filename.startswith('a') and len(base_filename) > 7:
                        asteroid_id = int(base_filename[1:7])
                        file_map[asteroid_id] = filename
                    elif base_filename.startswith('a'):
                        parts = base_filename.replace('a', '').replace('.txt', '').replace('.dat', '')
                        asteroid_id = int(parts)
                        file_map[asteroid_id] = filename
                    else:
                        parts = base_filename.replace('.txt', '').replace('.dat', '').split('.')[0]
                        asteroid_id = int(parts)
                        file_map[asteroid_id] = filename
                except (ValueError, IndexError):
                    print(f"Warning: Could not parse asteroid ID from filename: {filename}")
                    continue
    print(f"Successfully created map with {len(file_map)} entries.")
    return file_map

def extract_spectral_features(wavelength, reflectance):
    """Extract comprehensive spectral features for resource assessment."""
    features = {}
    features['mean_reflectance'] = np.mean(reflectance)
    features['std_reflectance'] = np.std(reflectance)
    features['min_reflectance'] = np.min(reflectance)
    features['max_reflectance'] = np.max(reflectance)
    features['reflectance_range'] = np.ptp(reflectance)
    if len(wavelength) > 1:
        slope = np.polyfit(wavelength, reflectance, 1)[0]
        features['spectral_slope'] = slope
        vis_mask = (wavelength >= 0.5) & (wavelength <= 0.9)
        if np.sum(vis_mask) > 5:
            features['visible_slope'] = np.polyfit(wavelength[vis_mask], reflectance[vis_mask], 1)[0]
        else:
            features['visible_slope'] = slope
    if np.min(wavelength) <= 1.0 <= np.max(wavelength):
        idx_1um = np.argmin(np.abs(wavelength - 1.0))
        if 2 < idx_1um < len(reflectance) - 2:
            continuum = np.mean([reflectance[idx_1um-2], reflectance[idx_1um+2]])
            if continuum > 0:
                features['band_1um_depth'] = max(0, 1 - (reflectance[idx_1um] / continuum))
    if np.min(wavelength) <= 2.0 <= np.max(wavelength):
        idx_2um = np.argmin(np.abs(wavelength - 2.0))
        if 2 < idx_2um < len(reflectance) - 2:
            continuum = np.mean([reflectance[idx_2um-2], reflectance[idx_2um+2]])
            if continuum > 0:
                features['band_2um_depth'] = max(0, 1 - (reflectance[idx_2um] / continuum))
    default_features = ['band_1um_depth', 'band_2um_depth', 'visible_slope']
    for feature in default_features:
        if feature not in features:
            features[feature] = 0.0
    return features

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    print("Starting new ensemble trainer...")

    # Load and merge catalogs
    print("--- Loading and merging catalogs ---")
    catalog_old_df = pd.read_csv(catalog_path)
    catalog_new_df = pd.read_csv(catalog_path_new)
    catalog_old_df = catalog_old_df[['Number', 'Simplified Category']]
    catalog_new_df = catalog_new_df[['Number', 'Simplified_Taxon']]
    catalog_new_df.rename(columns={'Simplified_Taxon': 'Simplified Category'}, inplace=True)
    catalog_full = pd.concat([catalog_old_df, catalog_new_df], ignore_index=True)
    catalog_full.drop_duplicates(subset=['Number', 'Simplified Category'], inplace=True)
    catalog_full.dropna(subset=['Number', 'Simplified Category'], inplace=True)
    print(f"Loaded and merged catalogs with {len(catalog_full)} unique entries.")

    # Create file maps for both zip archives
    file_map_old = create_filename_map(spectra_zip_path)
    file_map_new = create_filename_map(spectra_zip_path_new)

    # Process data from both zip files
    all_features = []
    all_taxonomies = []

    def process_zip(zip_path, file_map, catalog, processed_ids):
        features_list = []
        labels = []
        with zipfile.ZipFile(zip_path, 'r') as zip_f:
            for _, row in catalog.iterrows():
                asteroid_id = int(row['Number'])
                if asteroid_id in file_map and asteroid_id not in processed_ids:
                    filename = file_map[asteroid_id]
                    try:
                        with zip_f.open(filename) as f:
                            spectrum = np.loadtxt(f, comments='#', usecols=(0, 1))
                        if spectrum.shape[0] < 10: continue
                        wavelength, reflectance = spectrum[:, 0], spectrum[:, 1]
                        mask = np.isfinite(wavelength) & np.isfinite(reflectance) & (reflectance > 0)
                        wavelength, reflectance = wavelength[mask], reflectance[mask]
                        if len(wavelength) < 10: continue
                        if np.median(reflectance) > 0:
                            reflectance /= np.median(reflectance)
                        features = extract_spectral_features(wavelength, reflectance)
                        features_list.append(list(features.values()))
                        labels.append(row['Simplified Category'])
                        processed_ids.add(asteroid_id)
                    except Exception as e:
                        print(f"Warning: Error processing {filename}: {str(e)}")
                        continue
        return features_list, labels

    processed_asteroid_ids = set()

    print("--- Processing new zip file ---")
    new_features, new_labels = process_zip(spectra_zip_path_new, file_map_new, catalog_full, processed_asteroid_ids)
    all_features.extend(new_features)
    all_taxonomies.extend(new_labels)

    print("--- Processing old zip file (for remaining files) ---")
    old_features, old_labels = process_zip(spectra_zip_path, file_map_old, catalog_full, processed_asteroid_ids)
    all_features.extend(old_features)
    all_taxonomies.extend(old_labels)

    print(f"Successfully processed {len(all_features)} total spectra.")

    if not all_features:
        sys.exit("No features were extracted. Exiting.")

    # Prepare data for scikit-learn
    X = np.array(all_features)
    y_str = np.array(all_taxonomies)

    le = LabelEncoder()
    y = le.fit_transform(y_str)

    # Train-test split
    X_train, X_test, y_train, y_test, y_str_train, y_str_test = train_test_split(
        X, y, y_str, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Hyperparameter Tuning with GridSearchCV
    print("--- Tuning RandomForestClassifier with GridSearchCV ---")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

    rf = RandomForestClassifier(random_state=42, class_weight='balanced')

    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')
    grid_search.fit(X_train_scaled, y_train)

    print("Best parameters found: ", grid_search.best_params_)

    # Use the best estimator for evaluation
    model = grid_search.best_estimator_
    print("--- Model training complete ---")

    # Evaluate the model
    print("--- Evaluating model ---")
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
    plt.savefig('confusion_matrix_sklearn.png')
    print("Confusion matrix saved to confusion_matrix_sklearn.png")

    # Generate predictions for the entire dataset
    print("--- Generating predictions for all data ---")
    X_scaled = scaler.transform(X)
    full_predictions = model.predict(X_scaled)
    full_probabilities = model.predict_proba(X_scaled)

    results_df = pd.DataFrame({
        'Original_Taxonomy': y_str,
        'Predicted_Taxonomy': le.inverse_transform(full_predictions)
    })

    for i, class_label in enumerate(le.classes_):
        results_df[f'Prob_{class_label}'] = full_probabilities[:, i]

    results_df.to_csv(output_predictions_path, index=False)
    print(f"Predictions saved to {output_predictions_path}")

    print("Script finished.")
