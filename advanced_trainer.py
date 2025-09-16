import os
import zipfile
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv1D, GlobalMaxPooling1D, concatenate, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# Configuration
# ==============================================================================
# Only use the specified data sources
CATALOG_PATH = 'MITHNEOSCLEAN.csv'
SPECTRA_ZIP_PATH = 'visnir_files.zip'

# Output paths
MODEL_PATH = 'advanced_classification_model.keras'
SCALER_PATH = 'advanced_classification_scaler.pkl'
LABEL_ENCODER_PATH = 'advanced_classification_label_encoder.pkl'
PREDICTIONS_PATH = 'advanced_classification_predictions.csv'
CONFUSION_MATRIX_PATH = 'advanced_confusion_matrix.png'

# Model parameters
RESAMPLE_POINTS = 500 # Number of points to resample spectra to for CNN

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
                    # Extract the base filename to handle nested directories within the zip
                    base_filename = os.path.basename(filename)

                    # Attempt to parse asteroid ID from filename
                    # Handles formats like: 'a123456.txt', '123456.dat', 'a000132.visnir.txt'
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

    # Basic stats
    features['mean_reflectance'] = np.mean(reflectance)
    features['std_reflectance'] = np.std(reflectance)
    features['min_reflectance'] = np.min(reflectance)
    features['max_reflectance'] = np.max(reflectance)

    # Spectral slope
    if len(wavelength) > 1:
        # Overall slope
        slope, intercept = np.polyfit(wavelength, reflectance, 1)
        features['spectral_slope'] = slope

        # Visible slope (e.g., 0.5 to 0.9 microns)
        vis_mask = (wavelength >= 0.5) & (wavelength <= 0.9)
        if np.sum(vis_mask) > 5:
            vis_slope, _ = np.polyfit(wavelength[vis_mask], reflectance[vis_mask], 1)
            features['visible_slope'] = vis_slope
        else:
            features['visible_slope'] = slope # Fallback to overall slope

    # Band depths (e.g., around 1um and 2um)
    # A simple way to calculate band depth is to find the ratio of the band center
    # to a continuum (a straight line across the absorption feature).
    for band_center in [1.0, 2.0]:
        feature_name = f'band_{band_center}um_depth'
        features[feature_name] = 0.0 # Default to 0

        # Find the closest point in the wavelength data to the band center
        if np.min(wavelength) <= band_center <= np.max(wavelength):
            idx_center = np.argmin(np.abs(wavelength - band_center))

            # Define a region around the band for the continuum
            # These can be tuned based on typical spectral feature widths
            continuum_left_wl = band_center - 0.1
            continuum_right_wl = band_center + 0.1

            if np.min(wavelength) <= continuum_left_wl and np.max(wavelength) >= continuum_right_wl:
                idx_left = np.argmin(np.abs(wavelength - continuum_left_wl))
                idx_right = np.argmin(np.abs(wavelength - continuum_right_wl))

                # Ensure indices are valid and distinct
                if idx_left < idx_center < idx_right:
                    # Linear continuum
                    continuum_value = np.interp(wavelength[idx_center], [wavelength[idx_left], wavelength[idx_right]], [reflectance[idx_left], reflectance[idx_right]])

                    if continuum_value > 0:
                        # Band depth = 1 - (Reflectance / Continuum)
                        features[feature_name] = max(0, 1 - (reflectance[idx_center] / continuum_value))

    # Fill any missing features with 0
    expected_features = ['mean_reflectance', 'std_reflectance', 'min_reflectance', 'max_reflectance', 'spectral_slope', 'visible_slope', 'band_1.0um_depth', 'band_2.0um_depth']
    for f in expected_features:
        if f not in features:
            features[f] = 0.0

    return features

# ==============================================================================
# Main Data Loading and Processing Function
# ==============================================================================
def load_and_process_data():
    """Loads data, processes spectra, and extracts features."""
    print("--- Loading and Processing Data ---")

    # Load the asteroid catalog
    try:
        catalog_df = pd.read_csv(CATALOG_PATH)
        catalog_df.dropna(subset=['Number', 'Simplified Category'], inplace=True)
        catalog_df['Number'] = catalog_df['Number'].astype(int)
        print(f"Loaded catalog with {len(catalog_df)} entries.")
    except FileNotFoundError:
        sys.exit(f"FATAL: Catalog file not found at {CATALOG_PATH}")

    # Create a map of asteroid numbers to spectral filenames
    file_map = create_filename_map(SPECTRA_ZIP_PATH)
    if not file_map:
        sys.exit("FATAL: No spectral files found or mapped. Exiting.")

    # Lists to store our processed data
    all_engineered_features = []
    all_resampled_spectra = []
    all_labels = []

    # Define the new, uniform wavelength range for resampling
    min_wl = 0.45 # Common minimum for VISNIR
    max_wl = 2.45 # Common maximum for VISNIR
    uniform_wavelengths = np.linspace(min_wl, max_wl, RESAMPLE_POINTS)

    with zipfile.ZipFile(SPECTRA_ZIP_PATH, 'r') as zip_f:
        for _, row in catalog_df.iterrows():
            asteroid_id = row['Number']
            if asteroid_id in file_map:
                filename = file_map[asteroid_id]
                try:
                    with zip_f.open(filename) as f:
                        # Load spectrum, skipping header comments
                        spectrum = np.loadtxt(f, comments='#', usecols=(0, 1))

                    # Basic validation
                    if spectrum.ndim != 2 or spectrum.shape[1] != 2 or spectrum.shape[0] < 10:
                        continue

                    wavelength, reflectance = spectrum[:, 0], spectrum[:, 1]

                    # Clean the data: remove NaNs/Infs and non-positive reflectance values
                    mask = np.isfinite(wavelength) & np.isfinite(reflectance) & (reflectance > 0)
                    wavelength, reflectance = wavelength[mask], reflectance[mask]

                    if len(wavelength) < 10:
                        continue

                    # Normalize reflectance by dividing by the median value
                    # This helps standardize the spectra
                    median_reflectance = np.median(reflectance)
                    if median_reflectance > 0:
                        reflectance /= median_reflectance

                    # Resample the spectrum to a uniform wavelength grid for the CNN
                    resampled_reflectance = np.interp(uniform_wavelengths, wavelength, reflectance)

                    # Extract engineered features from the original, cleaned spectrum
                    features = extract_spectral_features(wavelength, reflectance)

                    all_engineered_features.append(list(features.values()))
                    all_resampled_spectra.append(resampled_reflectance)
                    all_labels.append(row['Simplified Category'])

                except Exception as e:
                    print(f"Warning: Error processing {filename} for asteroid {asteroid_id}: {e}")
                    continue

    print(f"Successfully processed {len(all_labels)} spectra.")
    if not all_labels:
        sys.exit("FATAL: No data was successfully processed. Exiting.")

    # Convert lists to numpy arrays for machine learning
    X_engineered = np.array(all_engineered_features)
    X_spectra = np.array(all_resampled_spectra)
    y_labels = np.array(all_labels)

    # Add a channel dimension to the spectral data for the 1D CNN
    X_spectra = np.expand_dims(X_spectra, axis=-1)

    return X_engineered, X_spectra, y_labels


# ==============================================================================
# Feature Preprocessing Function
# ==============================================================================
def preprocess_features(X_engineered, y_labels):
    """Scales engineered features and encodes labels."""
    print("\n--- Preprocessing Features ---")

    # Scale the engineered features
    scaler = StandardScaler()
    X_engineered_scaled = scaler.fit_transform(X_engineered)
    print("Engineered features scaled.")

    # Encode the string labels to integers, then to categorical (one-hot)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_labels)
    y_categorical = to_categorical(y_encoded)
    print("Labels encoded and converted to categorical format.")

    num_classes = len(label_encoder.classes_)
    print(f"Number of classes: {num_classes}")

    return X_engineered_scaled, y_categorical, scaler, label_encoder, num_classes


# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    print("Starting Advanced Asteroid Classification Trainer...")
    X_engineered, X_spectra, y_labels = load_and_process_data()

    # Preprocess the features and labels
    X_engineered_scaled, y_categorical, scaler, label_encoder, num_classes = preprocess_features(X_engineered, y_labels)

    # Split data into training and testing sets
    print("\n--- Splitting Data into Training and Test Sets ---")
    X_eng_train, X_eng_test, X_spec_train, X_spec_test, y_train, y_test = train_test_split(
        X_engineered_scaled,
        X_spectra,
        y_categorical,
        test_size=0.2,
        random_state=42,
        stratify=y_categorical
    )
    print(f"Training set size: {len(y_train)}")
    print(f"Test set size: {len(y_test)}")

    # Handle class imbalance with SMOTE
    print("\n--- Handling Class Imbalance with SMOTE ---")

    # SMOTE works on 2D data, so we need to flatten the spectral data first.
    nsamples, nx, ny = X_spec_train.shape
    X_spec_train_reshaped = X_spec_train.reshape((nsamples, nx * ny))

    # Concatenate the two feature sets to apply SMOTE consistently
    X_train_combined = np.hstack((X_eng_train, X_spec_train_reshaped))

    # We use the original integer-encoded labels for SMOTE
    y_train_integers = np.argmax(y_train, axis=1)

    # SMOTE requires at least k_neighbors+1 samples in each class.
    # We check the smallest class size and adjust k_neighbors if necessary.
    min_class_size = np.min(np.bincount(y_train_integers))
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1

    print(f"Applying SMOTE with k_neighbors={k_neighbors}")
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_smote_combined, y_train_smote_cat = smote.fit_resample(X_train_combined, y_train)

    # Separate the engineered and spectral features again
    X_eng_train_smote = X_train_smote_combined[:, :X_eng_train.shape[1]]
    X_spec_train_smote_reshaped = X_train_smote_combined[:, X_eng_train.shape[1]:]

    # Reshape the spectral data back to its original 3D format for the CNN
    X_spec_train_smote = X_spec_train_smote_reshaped.reshape((len(X_spec_train_smote_reshaped), nx, ny))

    print(f"Original training samples: {len(y_train)}")
    print(f"SMOTE-resampled training samples: {len(y_train_smote_cat)}")

    # Rename variables for clarity in the training step
    X_eng_train_final = X_eng_train_smote
    X_spec_train_final = X_spec_train_smote
    y_train_final = y_train_smote_cat

    print("\nSMOTE application complete.")

# ==============================================================================
# Model Building Function
# ==============================================================================
def build_model(engineered_shape, spectra_shape, num_classes):
    """Builds the multi-input 1D CNN and MLP model."""
    print("\n--- Building the Model ---")

    # MLP branch for engineered features
    input_engineered = Input(shape=engineered_shape, name='engineered_input')
    x_eng = Dense(64, activation='relu')(input_engineered)
    x_eng = Dropout(0.5)(x_eng)
    x_eng = Dense(32, activation='relu')(x_eng)

    # CNN branch for spectral data
    input_spectra = Input(shape=spectra_shape, name='spectra_input')
    x_spec = Conv1D(filters=32, kernel_size=10, activation='relu')(input_spectra)
    x_spec = Conv1D(filters=64, kernel_size=10, activation='relu')(x_spec)
    x_spec = GlobalMaxPooling1D()(x_spec)
    x_spec = Dropout(0.5)(x_spec)

    # Concatenate branches
    combined = concatenate([x_eng, x_spec])

    # Final dense layers
    z = Dense(128, activation='relu')(combined)
    z = Dropout(0.5)(z)
    output = Dense(num_classes, activation='softmax', name='output')(z)

    # Create and compile model
    model = Model(inputs=[input_engineered, input_spectra], outputs=output)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    return model

# ==============================================================================
# Main Execution
# ==============================================================================
if __name__ == '__main__':
    print("Starting Advanced Asteroid Classification Trainer...")
    # ... (previous code remains the same)
    X_engineered, X_spectra, y_labels = load_and_process_data()
    X_engineered_scaled, y_categorical, scaler, label_encoder, num_classes = preprocess_features(X_engineered, y_labels)
    X_eng_train, X_eng_test, X_spec_train, X_spec_test, y_train, y_test = train_test_split(
        X_engineered_scaled, X_spectra, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )

    # ... (SMOTE logic remains the same)
    nsamples, nx, ny = X_spec_train.shape
    X_spec_train_reshaped = X_spec_train.reshape((nsamples, nx * ny))
    X_train_combined = np.hstack((X_eng_train, X_spec_train_reshaped))
    y_train_integers = np.argmax(y_train, axis=1)
    min_class_size = np.min(np.bincount(y_train_integers))
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_smote_combined, y_train_smote_cat = smote.fit_resample(X_train_combined, y_train)
    X_eng_train_smote = X_train_smote_combined[:, :X_eng_train.shape[1]]
    X_spec_train_smote_reshaped = X_train_smote_combined[:, X_eng_train.shape[1]:]
    X_spec_train_smote = X_spec_train_smote_reshaped.reshape((len(X_spec_train_smote_reshaped), nx, ny))

    X_eng_train_final = X_eng_train_smote
    X_spec_train_final = X_spec_train_smote
    y_train_final = y_train_smote_cat

    # Build the model
    model = build_model(
        engineered_shape=(X_eng_train_final.shape[1],),
        spectra_shape=(X_spec_train_final.shape[1], X_spec_train_final.shape[2]),
        num_classes=num_classes
    )

    # Train the model
    print("\n--- Training the Model ---")

    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)

    history = model.fit(
        [X_eng_train_final, X_spec_train_final],
        y_train_final,
        epochs=100,
        batch_size=32,
        validation_split=0.2, # Use 20% of training data for validation
        callbacks=[early_stopping, reduce_lr]
    )

    print("\nModel training complete.")

# ==============================================================================
# Evaluation and Artifact Saving
# ==============================================================================
def evaluate_and_save(model, history, X_eng_test, X_spec_test, y_test, scaler, label_encoder):
    """Evaluates the model and saves all artifacts."""
    print("\n--- Evaluating Model and Saving Artifacts ---")

    # Evaluate the model on the test data
    loss, accuracy = model.evaluate([X_eng_test, X_spec_test], y_test, verbose=0)
    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # Generate predictions
    y_pred_proba = model.predict([X_eng_test, X_spec_test])
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_test_true = np.argmax(y_test, axis=1)

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test_true, y_pred, target_names=label_encoder.classes_, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_test_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    plt.savefig(CONFUSION_MATRIX_PATH)
    print(f"\nConfusion matrix saved to {CONFUSION_MATRIX_PATH}")

    # Save the model, scaler, and label encoder
    model.save(MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH}")

    import pickle
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Feature scaler saved to {SCALER_PATH}")

    with open(LABEL_ENCODER_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"Label encoder saved to {LABEL_ENCODER_PATH}")

# Add the call to the main execution block
if __name__ == '__main__':
    # ... (all previous code from main block) ...
    print("Starting Advanced Asteroid Classification Trainer...")
    X_engineered, X_spectra, y_labels = load_and_process_data()
    X_engineered_scaled, y_categorical, scaler, label_encoder, num_classes = preprocess_features(X_engineered, y_labels)
    X_eng_train, X_eng_test, X_spec_train, X_spec_test, y_train, y_test = train_test_split(
        X_engineered_scaled, X_spectra, y_categorical, test_size=0.2, random_state=42, stratify=y_categorical
    )
    nsamples, nx, ny = X_spec_train.shape
    X_spec_train_reshaped = X_spec_train.reshape((nsamples, nx * ny))
    X_train_combined = np.hstack((X_eng_train, X_spec_train_reshaped))
    y_train_integers = np.argmax(y_train, axis=1)
    min_class_size = np.min(np.bincount(y_train_integers))
    k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_train_smote_combined, y_train_smote_cat = smote.fit_resample(X_train_combined, y_train)
    X_eng_train_smote = X_train_smote_combined[:, :X_eng_train.shape[1]]
    X_spec_train_smote_reshaped = X_train_smote_combined[:, X_eng_train.shape[1]:]
    X_spec_train_smote = X_spec_train_smote_reshaped.reshape((len(X_spec_train_smote_reshaped), nx, ny))
    X_eng_train_final, X_spec_train_final, y_train_final = X_eng_train_smote, X_spec_train_smote, y_train_smote_cat
    model = build_model(
        engineered_shape=(X_eng_train_final.shape[1],),
        spectra_shape=(X_spec_train_final.shape[1], X_spec_train_final.shape[2]),
        num_classes=num_classes
    )
    history = model.fit(
        [X_eng_train_final, X_spec_train_final], y_train_final,
        epochs=100, batch_size=32, validation_split=0.2,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
                   ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001, verbose=1)]
    )

    # Final step: Evaluate and save
    evaluate_and_save(model, history, X_eng_test, X_spec_test, y_test, scaler, label_encoder)

    print("\nScript finished successfully.")
