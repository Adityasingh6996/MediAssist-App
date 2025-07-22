import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier # New: Our Machine Learning Model!
from sklearn.metrics import accuracy_score, classification_report # New: For evaluating the model
import pickle # New: For saving our model and encoder

# --- Re-load and Preprocess Data (as done in data_preparation.py) ---
# In a real project, you might have these steps in a function or separate file,
# but for clarity in this journey, we'll repeat the necessary loading and preprocessing
# to ensure X_train, y_train_encoded, etc., are available in this script.

print("--- Starting Data Loading and Preprocessing for Model Training ---")

training_data_path = 'data/training.csv'
testing_data_path = 'data/testing.csv'

df_train = pd.read_csv(training_data_path)
df_test = pd.read_csv(testing_data_path)

# Drop the 'Unnamed: 133' column from the training data if it exists
if 'Unnamed: 133' in df_train.columns:
    df_train = df_train.drop('Unnamed: 133', axis=1)

# Trim whitespace from column names
df_train.columns = df_train.columns.str.strip()
df_test.columns = df_test.columns.str.strip()


# Separate features (X) and target (y)
X_train = df_train.drop('prognosis', axis=1)
y_train = df_train['prognosis']

X_test = df_test.drop('prognosis', axis=1)
y_test = df_test['prognosis']

# Initialize and fit LabelEncoder for the target variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Store disease names (classes) for later use
disease_names = label_encoder.classes_

print("Data loaded and preprocessed for model training.")
print(f"Features shape (X_train): {X_train.shape}")
print(f"Target shape (y_train_encoded): {y_train_encoded.shape}")

# --- Step 2.3: Build and Train the Machine Learning Model ---
print("\n--- Building and Training Random Forest Classifier ---")

# Initialize the Random Forest Classifier model
# n_estimators is the number of trees in the forest. More trees generally means better performance but slower training.
# random_state ensures we get the same results every time we run the code.
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model using the training data (features and encoded target)
model.fit(X_train, y_train_encoded)
print("Random Forest Classifier training complete!")

# --- Step 2.4: Evaluate the Model ---
print("\n--- Evaluating Model Performance ---")

# Make predictions on the test data
y_pred_encoded = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test_encoded, y_pred_encoded)
print(f"Model Accuracy on Test Data: {accuracy * 100:.2f}%")

# Generate a detailed classification report
# This shows precision, recall, f1-score for each disease
# We use target_names to show actual disease names in the report
print("\nClassification Report:")
print(classification_report(y_test_encoded, y_pred_encoded, target_names=disease_names))

# --- Step 2.5: Save the Trained Model and LabelEncoder ---
# We need to save the model and the encoder so our web application can use them later to make predictions.
# They will be saved in the 'models/' folder.

model_save_path = 'models/disease_prediction_model.pkl'
encoder_save_path = 'models/label_encoder.pkl'

# Save the trained model
with open(model_save_path, 'wb') as file: # 'wb' means write in binary mode
    pickle.dump(model, file)
print(f"\nTrained model saved to: {model_save_path}")

# Save the LabelEncoder
with open(encoder_save_path, 'wb') as file:
    pickle.dump(label_encoder, file)
print(f"LabelEncoder saved to: {encoder_save_path}")

print("\n--- Model Training, Evaluation, and Saving Complete ---")
print("Your AI brain is ready!")