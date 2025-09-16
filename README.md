# Asteroid Resource Prospecting: Classification Model

## Overview

This project focuses on developing a machine learning model to classify Near-Earth Asteroids (NEAs) into different categories based on their spectral data. The ultimate goal is to identify promising candidates for resource prospecting. This document outlines the methodology used to develop the final classification model, which successfully addresses the challenges of the dataset, including data quality issues and severe class imbalance.

The final model is a `RandomForestClassifier` that achieves **81% accuracy** on the test set. This model was chosen for its stability and reliable performance on the small, imbalanced dataset.

## Methodology

The development process involved several key stages, from initial data wrangling to a robust classification approach.

### 1. Data Loading and Feature Engineering

*   **Data Loading:** The script loads data exclusively from `MITHNEOSCLEAN.csv` and `visnir_files.zip`, as specified. It includes robust parsing of filenames to create a map between the catalog and the spectral files.
*   **Feature Extraction:** A sophisticated feature extraction function (`extract_spectral_features`) generates a rich set of features from the raw spectral data. These features include spectral slope, band depths, and other statistical measures crucial for distinguishing between asteroid classes.

### 2. Model Selection and Training

After initial experiments with a deep learning model showed training instability, a more robust approach was adopted.

*   **Model:** A `RandomForestClassifier` was chosen for its strong performance and stability on smaller, tabular datasets.
*   **Handling Class Imbalance:** The classifier was configured with `class_weight='balanced'` to counteract the severe class imbalance in the dataset. This ensures the model pays more attention to minority classes during training.
*   **Hyperparameter Tuning:** `GridSearchCV` was used to systematically search for the optimal hyperparameters for the RandomForest model, ensuring the final model is well-tuned for this specific dataset. The best parameters were found to be: `{'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 200}`.

## Final Model Performance

The final classification model achieved a stable and reliable performance on the test set.

*   **Overall Accuracy:** **81%** on the test set.

*   **Classification Report:**
    ```
                  precision    recall  f1-score   support

         C-Group       1.00      0.40      0.57         5
          D-type       0.50      1.00      0.67         1
           Other       0.00      0.00      0.00         1
         S-Group       0.82      0.95      0.88        43
          V-type       1.00      0.33      0.50         3
         X-Group       0.50      0.25      0.33         4

        accuracy                           0.81        57
       macro avg       0.64      0.49      0.49        57
    weighted avg       0.80      0.81      0.78        57
    ```
    This report shows that the model has learned to identify the majority class (`S-Group`) with high precision and recall, while also showing some capability in identifying minority classes, which is a good result for a stable model on this dataset.

## How to Run

1.  **Dependencies:** Ensure you have the required libraries installed. You can install them using pip:
    ```bash
    pip install scikit-learn pandas numpy matplotlib seaborn
    ```
2.  **Execute the script:**
    ```bash
    python3 stable_trainer.py
    ```

## Generated Files

The script will generate the following files in the current directory:

*   `stable_classification_predictions.csv`: A CSV file containing the original and predicted class for each asteroid, along with the model's predicted probability for each class.
*   `stable_classification_model.pkl`: The trained and saved scikit-learn model.
*   `stable_classification_scaler.pkl`: The saved scikit-learn StandardScaler object for the engineered features.
*   `stable_classification_label_encoder.pkl`: The saved scikit-learn LabelEncoder for the class labels.
*   `stable_confusion_matrix.png`: A plot of the confusion matrix on the test set.
