import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # NEW: For scaling numerical features
import pickle
import os

print("--- Starting Heart AI Data Preparation ---")

# --- Path to the heart disease dataset ---
# Make sure 'heart.csv' is in your 'data' folder
heart_data_path = 'data/heart.csv'

try:
    df_heart = pd.read_csv(heart_data_path)
    print(f"Heart disease data loaded successfully from: {heart_data_path}")
except FileNotFoundError:
    print(f"Error: heart.csv not found at {heart_data_path}. Make sure it's in the 'data' folder.")
    exit()

# --- Initial Data Inspection ---
print("\n--- First 5 rows of Heart Data ---")
print(df_heart.head())

print("\n--- Info about Heart Data ---")
df_heart.info()

print("\n--- Missing values in Heart Data ---")
print(df_heart.isnull().sum()) # Check for missing values

print("\n--- Basic statistics of Heart Data ---")
print(df_heart.describe())

# --- Feature Engineering & Preprocessing ---

# Separate features (X) and target (y)
# The target variable is usually named 'target' or 'output' in these datasets.
# Check your df_heart.head() to confirm the target column name.
# We'll assume 'target' for now. If yours is 'output' or something else, change it here.
X = df_heart.drop('target', axis=1) # Features are all columns except 'target'
y = df_heart['target']             # Target is the 'target' column
print("\nSeparated features (X) and target (y).")

# Identify categorical and numerical columns for preprocessing
# This list might need adjustment based on the exact column names in your downloaded heart.csv
# Common categorical features in heart disease datasets:
# Based on your df_heart.info() and df_heart.head() output:
categorical_cols = [
    'sex',
    'chest_pain_type', # This was missing or different
    'fasting_blood_sugar', # This was missing or different
    'rest_ecg',
    'exercise_induced_angina',
    'slope',
    'vessels_colored_by_flourosopy', # This was missing or different
    'thalassemia'
]
numerical_cols = [
    'age',
    'resting_blood_pressure', # This was missing or different
    'cholestoral', # This was missing or different
    'Max_heart_rate', # This was missing or different
    'oldpeak'
]

# Ensure all identified columns exist in X
categorical_cols = [col for col in categorical_cols if col in X.columns]
numerical_cols = [col for col in numerical_cols if col in X.columns]
print(f"Identified Categorical Columns: {categorical_cols}")
print(f"Identified Numerical Columns: {numerical_cols}")

# One-hot encode categorical features
# This converts categorical labels (like 'male'/'female' or 'type 1'/'type 2' chest pain)
# into numerical (0s and 1s) columns, which ML models can understand.
print("\nPerforming One-Hot Encoding for categorical features...")
X_categorical = pd.get_dummies(X[categorical_cols], columns=categorical_cols, drop_first=True)
print("One-Hot Encoding complete.")

# Scale numerical features
# Scaling ensures that features with larger values (e.g., cholesterol) don't
# unfairly dominate features with smaller values (e.g., age) during model training.
# StandardScaler makes the mean 0 and standard deviation 1.
print("Performing Standard Scaling for numerical features...")
scaler = StandardScaler()
X_numerical_scaled = scaler.fit_transform(X[numerical_cols])
# Convert scaled array back to DataFrame for easy concatenation and retaining column names
X_numerical_scaled = pd.DataFrame(X_numerical_scaled, columns=numerical_cols, index=X.index)
print("Standard Scaling complete.")

# Combine preprocessed features
X_processed = pd.concat([X_numerical_scaled, X_categorical], axis=1)
print("\nCombined scaled numerical and one-hot encoded categorical features.")
print(f"Shape of processed features (X_processed): {X_processed.shape}")
print("First 5 rows of processed features:")
print(X_processed.head())

# --- Split the Data into Training and Testing Sets ---
# This step is crucial for evaluating our model unbiasedly.
# test_size=0.2 means 20% of data for testing, 80% for training.
# random_state ensures reproducibility of the split.
print("\nSplitting data into training and testing sets...")
X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(
    X_processed, y, test_size=0.2, random_state=42
)
print("Data split complete.")
print(f"Shape of X_train_heart: {X_train_heart.shape}")
print(f"Shape of X_test_heart: {X_test_heart.shape}")

# --- Save Preprocessing Tools ---
# We need to save the scaler and the list of feature names (after encoding)
# so that when a new user provides input in the Flask app, we can apply the *same*
# transformations to their input before feeding it to the trained model.

# Save the scaler
scaler_save_path = 'models/heart_scaler.pkl'
with open(scaler_save_path, 'wb') as file:
    pickle.dump(scaler, file)
print(f"\nHeart scaler saved to: {scaler_save_path}")

# Save the list of final feature column names
# This is important because pd.get_dummies creates new columns, and their order must be consistent
feature_names_save_path = 'models/heart_feature_names.pkl'
with open(feature_names_save_path, 'wb') as file:
    pickle.dump(X_processed.columns.tolist(), file)
print(f"Heart feature names saved to: {feature_names_save_path}")

print("\n--- Heart AI Data Preparation Complete ---")
print("Data is now ready for Heart AI Model Training.")