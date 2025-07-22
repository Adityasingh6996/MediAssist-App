import pandas as pd
from sklearn.model_selection import train_test_split # New: For splitting data if needed
from sklearn.preprocessing import LabelEncoder # New: For converting disease names to numbers

# --- Step 1: Load the Datasets ---
print("--- Starting Data Loading ---")
training_data_path = 'data/training.csv'
testing_data_path = 'data/testing.csv'

try:
    df_train = pd.read_csv(training_data_path)
    print(f"Training data loaded successfully from: {training_data_path}")
except FileNotFoundError:
    print(f"Error: training.csv not found at {training_data_path}. Make sure it's in the 'data' folder.")
    exit()

try:
    df_test = pd.read_csv(testing_data_path)
    print(f"Testing data loaded successfully from: {testing_data_path}")
except FileNotFoundError:
    print(f"Error: testing.csv not found at {testing_data_path}. Make sure it's in the 'data' folder.")
    exit()
print("--- Data Loading Complete ---")


# --- Step 2: Data Cleaning and Preprocessing ---
print("\n--- Starting Data Cleaning and Preprocessing ---")

# 2.1 Drop the 'Unnamed: 133' column from the training data
# This column is completely empty and useless, as observed from df_train.isnull().sum()
if 'Unnamed: 133' in df_train.columns:
    df_train = df_train.drop('Unnamed: 133', axis=1) # axis=1 means drop a column
    print("Dropped 'Unnamed: 133' column from training data.")

# 2.2 Standardize column names (optional but good practice)
# We can clean up any leading/trailing spaces in column names if they exist
df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()
print("Trimmed whitespace from column names.")


# 2.3 Separate features (symptoms) and target (disease)
# Features (X): All columns except 'prognosis'
# Target (y): The 'prognosis' column

# X_train will contain all symptom columns from the training data
X_train = df_train.drop('prognosis', axis=1)
# y_train will contain the 'prognosis' column from the training data
y_train = df_train['prognosis']
print("Separated features (X_train) and target (y_train) for training data.")

# X_test will contain all symptom columns from the testing data
X_test = df_test.drop('prognosis', axis=1)
# y_test will contain the 'prognosis' column from the testing data
y_test = df_test['prognosis']
print("Separated features (X_test) and target (y_test) for testing data.")

# 2.4 Handle categorical target variable (prognosis)
# Machine Learning models work best with numbers. Our 'prognosis' column has text (disease names).
# We'll use LabelEncoder to convert each disease name into a unique number.
# For example, 'Fungal infection' might become 0, 'Allergy' might become 1, and so on.

label_encoder = LabelEncoder()
print("Initializing LabelEncoder for 'prognosis' column.")

# Fit and transform y_train: Learn the mapping from names to numbers, then apply it.
y_train_encoded = label_encoder.fit_transform(y_train)
print(f"Encoded training target (y_train_encoded). First 5 encoded: {y_train_encoded[:5]}")

# Transform y_test: Use the *same* mapping learned from y_train to convert y_test.
# It's crucial to use the *same* encoder for training and testing data to ensure consistency.
y_test_encoded = label_encoder.transform(y_test)
print(f"Encoded testing target (y_test_encoded). First 5 encoded: {y_test_encoded[:5]}")

# Store the classes (original disease names) that the encoder learned.
# This is important for converting numerical predictions back to human-readable disease names.
disease_names = label_encoder.classes_
print(f"Stored original disease names (classes): {disease_names[:5]}...") # print first 5 to confirm

print("--- Data Cleaning and Preprocessing Complete ---")
print("Data is now ready for Machine Learning Model Training (X_train, y_train_encoded, X_test, y_test_encoded).")

# You can add print statements here to confirm the shapes of the processed dataframes
print(f"\nShape of X_train (features for training): {X_train.shape}")
print(f"Shape of y_train_encoded (target for training): {y_train_encoded.shape}")
print(f"Shape of X_test (features for testing): {X_test.shape}")
print(f"Shape of y_test_encoded (target for testing): {y_test_encoded.shape}")

# This script only prepares the data. The actual model training will be in the next step.