import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression # NEW: Our Heart AI Model!
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix # NEW: For evaluation
import pickle
import os

print("--- Starting Heart AI Model Training ---")

# --- Path to saved preprocessing tools and data ---
models_folder_path = 'models/' # Relative to where this script runs
data_folder_path = 'data/'

# Load the processed heart data (we'll re-run preparation steps to get X_train_heart etc.)
# In a real pipeline, you might save and load these directly, but repeating preparation
# ensures consistency with what we just did.
heart_data_path = os.path.join(data_folder_path, 'heart.csv')
df_heart = pd.read_csv(heart_data_path)

X = df_heart.drop('target', axis=1)
y = df_heart['target']

# These lists MUST match what you finalized in heart_ai_preparation.py
categorical_cols = [
    'sex',
    'chest_pain_type',
    'fasting_blood_sugar',
    'rest_ecg',
    'exercise_induced_angina',
    'slope',
    'vessels_colored_by_flourosopy',
    'thalassemia'
]
numerical_cols = [
    'age',
    'resting_blood_pressure',
    'cholestoral',
    'Max_heart_rate',
    'oldpeak'
]

# One-hot encode categorical features
X_categorical = pd.get_dummies(X[categorical_cols], columns=categorical_cols, drop_first=True)

# Load the saved scaler for numerical features
scaler_save_path = os.path.join(models_folder_path, 'heart_scaler.pkl')
with open(scaler_save_path, 'rb') as file:
    scaler = pickle.load(file)

# Scale numerical features using the loaded scaler
X_numerical_scaled = scaler.transform(X[numerical_cols])
X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=X.index)

# Load the saved feature names to ensure correct column order
feature_names_save_path = os.path.join(models_folder_path, 'heart_feature_names.pkl')
with open(feature_names_save_path, 'rb') as file:
    all_feature_names = pickle.load(file)

# Reconstruct X_processed ensuring column order matches the saved feature names
X_processed = pd.concat([X_numerical_scaled, X_categorical], axis=1)
# Important: Reindex X_processed to match the order of all_feature_names
# This ensures consistency for the model
X_processed = X_processed.reindex(columns=all_feature_names, fill_value=0) # fill_value=0 for dummy columns not present in all data subsets

# Split the Data into Training and Testing Sets
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)
print("Data loaded and prepared for heart AI model training.")
print(f"Features shape (X_train_heart): {X_train_heart.shape}")
print(f"Target shape (y_train_heart): {y_train_heart.shape}")

# --- Build and Train the Heart AI Model ---
print("\n--- Building and Training Logistic Regression Model for Heart AI ---")

# Initialize the Logistic Regression model
# solver='liblinear' is a good choice for small datasets and binary classification
model_heart = LogisticRegression(solver='liblinear', random_state=42)

# Train the model
model_heart.fit(X_train_heart, y_train_heart)
print("Logistic Regression model training complete!")

# --- Evaluate the Model ---
print("\n--- Evaluating Heart AI Model Performance ---")

# Make predictions on the test data
y_pred_heart = model_heart.predict(X_test_heart)

# Calculate accuracy score
accuracy_heart = accuracy_score(y_test_heart, y_pred_heart)
print(f"Heart AI Model Accuracy on Test Data: {accuracy_heart * 100:.2f}%")

# Generate classification report
print("\nHeart AI Classification Report:")
# target_names are '0' and '1' (no disease / disease)
print(classification_report(y_test_heart, y_pred_heart, target_names=['No Heart Disease', 'Heart Disease']))

# Confusion Matrix (optional, but good to see how many correct/incorrect predictions)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test_heart, y_pred_heart))


# --- Save the Trained Heart AI Model ---
heart_model_save_path = 'models/heart_disease_model.pkl'
with open(heart_model_save_path, 'wb') as file:
    pickle.dump(model_heart, file)
print(f"\nTrained Heart AI model saved to: {heart_model_save_path}")

print("\n--- Heart AI Model Training, Evaluation, and Saving Complete ---")
print("Your dedicated Heart AI brain is ready!")