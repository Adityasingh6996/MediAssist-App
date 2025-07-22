# Import necessary libraries
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
import pickle
import os
import sys
# No requests_oauthlib or dotenv needed for non-authenticated version
from functools import wraps # Keep for general decorators if any, but login_required will be removed

# --- Load environment variables from .env file ---
# load_dotenv() # Not needed if no .env secrets are used for the main app

# --- VERY IMPORTANT: Adjust Python's search path ---
current_dir = os.path.abspath(os.path.dirname(__file__)) # Path to D:/MediAssist_Project/app
parent_dir = os.path.join(current_dir, '..')             # Path to D:/MediAssist_Project
sys.path.insert(0, parent_dir)                           # Insert at the beginning of search path

# Import DISEASE_INFORMATION and TRANSLATIONS
from data.disease_info import DISEASE_INFORMATION
from data.translations import TRANSLATIONS

# Create a Flask web application instance
app = Flask(__name__)

# --- Flask Configuration for Sessions (still needed for language switcher) ---
app.secret_key = 'your_default_secret_key_for_non_auth_app' # A simple default secret key
app.config['SESSION_COOKIE_NAME'] = 'mediassist-session' # A generic session cookie name
app.config['SESSION_PERMANENT'] = False # Sessions not permanent if no login

# --- Path Definitions ---
models_folder_path = os.path.join(parent_dir, 'models')
data_folder_path = os.path.join(parent_dir, 'data')

# --- Load the Trained General Disease Model and LabelEncoder ---
general_model_path = os.path.join(models_folder_path, 'disease_prediction_model.pkl')
general_encoder_path = os.path.join(models_folder_path, 'label_encoder.pkl')

general_model = None
general_label_encoder = None
try:
    with open(general_model_path, 'rb') as model_file:
        general_model = pickle.load(model_file)
    print("General Disease model loaded successfully.")
    with open(general_encoder_path, 'rb') as encoder_file:
        general_label_encoder = pickle.load(encoder_file)
    print("General LabelEncoder loaded successfully.")
except FileNotFoundError:
    print(f"Error: General model or encoder file not found. Check paths: {general_model_path}, {general_encoder_path}")
except Exception as e:
    print(f"An error occurred while loading general model/encoder: {e}")

# --- Get the list of all symptoms ---
all_symptoms = []
try:
    training_csv_path = os.path.join(data_folder_path, 'training.csv')
    df_train_temp = pd.read_csv(training_csv_path)
    if 'Unnamed: 133' in df_train_temp.columns:
        df_train_temp = df_train_temp.drop('Unnamed: 133', axis=1)
    df_train_temp.columns = df_train_temp.columns.str.strip()
    all_symptoms = df_train_temp.drop('prognosis', axis=1).columns.tolist()
    print(f"Loaded {len(all_symptoms)} symptoms from training data.")
except FileNotFoundError:
    print(f"Error: training.csv not found at {training_csv_path}. Cannot retrieve symptom list.")
except Exception as e:
    print(f"An error occurred while getting symptom list: {e}. Ensure training.csv is correctly formatted.")

# --- Load Clinic Data ---
clinics_csv_path = os.path.join(data_folder_path, 'clinics.csv')
all_clinics = []
try:
    df_clinics = pd.read_csv(clinics_csv_path)
    all_clinics = df_clinics.to_dict(orient='records')
    print(f"Loaded {len(all_clinics)} clinic entries from {clinics_csv_path}.")
except FileNotFoundError:
    print(f"Error: clinics.csv not found at {clinics_csv_path}. Clinic finder will not work.")
except Exception as e:
    print(f"An error occurred while loading clinic data: {e}")

# --- Load Heart Disease AI Model and Preprocessing Tools ---
heart_model_path = os.path.join(models_folder_path, 'heart_disease_model.pkl')
heart_scaler_path = os.path.join(models_folder_path, 'heart_scaler.pkl')
heart_feature_names_path = os.path.join(models_folder_path, 'heart_feature_names.pkl')

heart_model = None
heart_scaler = None
heart_feature_names = None
try:
    with open(heart_model_path, 'rb') as file:
        heart_model = pickle.load(file)
    print("Heart Disease AI model loaded successfully.")
    with open(heart_scaler_path, 'rb') as file:
        heart_scaler = pickle.load(file)
    print("Heart Disease Scaler loaded successfully.")
    with open(heart_feature_names_path, 'rb') as file:
        heart_feature_names = pickle.load(file)
    print("Heart Disease Feature Names loaded successfully.")
except FileNotFoundError:
    print(f"Error: Heart AI model or preprocessing tools not found. Check paths: {heart_model_path}, {heart_scaler_path}, {heart_feature_names_path}")
except Exception as e:
    print(f"An error occurred while loading Heart AI model/tools: {e}")

# --- Define Numerical and Categorical Columns for Heart AI ---
categorical_cols = [
    'sex', 'chest_pain_type', 'fasting_blood_sugar', 'rest_ecg', 'exercise_induced_angina',
    'slope', 'vessels_colored_by_flourosopy', 'thalassemia'
]
numerical_cols = [
    'age', 'resting_blood_pressure', 'cholestoral', 'Max_heart_rate', 'oldpeak'
]
print("Defined numerical_cols and categorical_cols for Heart AI within app.py.")

# --- Language Management Helper ---
def get_translation_text():
    lang = session.get('language', 'en')
    return TRANSLATIONS.get(lang, TRANSLATIONS['en'])

# --- Define Flask Routes (NO @login_required decorators here) ---

@app.route('/')
def home():
    if not all_symptoms:
        return TRANSLATIONS['en']['error_symptoms_data_not_available']
    current_lang_text = get_translation_text()
    # No user_name passed here as there's no login
    return render_template('index.html', symptoms=all_symptoms, clinics=all_clinics, text=current_lang_text, current_lang=session.get('language', 'en'))

@app.route('/predict', methods=['POST'])
def predict():
    current_lang_text = get_translation_text()

    if general_model is None or general_label_encoder is None or not all_symptoms:
        print("DEBUG: General model or encoder or symptoms not loaded correctly at start of predict.")
        return render_template('index.html', symptoms=all_symptoms,
                            prediction_text=current_lang_text['error_system_not_loaded'],
                            clinics=all_clinics,
                            text=current_lang_text,
                            current_lang=session.get('language', 'en'))

    print("\n--- DEBUGGING PREDICT ROUTE (General Symptom Checker) ---")
    user_symptoms_raw = request.form.to_dict()
    print(f"DEBUG: Raw user symptoms from form: {user_symptoms_raw}")
    input_data = np.zeros(len(all_symptoms))
    selected_symptoms_from_form = []

    for symptom_key in user_symptoms_raw.keys():
        cleaned_symptom_key = symptom_key.strip()
        if cleaned_symptom_key in all_symptoms:
            symptom_index = all_symptoms.index(cleaned_symptom_key)
            input_data[symptom_index] = 1
            selected_symptoms_from_form.append(cleaned_symptom_key)

    print(f"DEBUG: Symptoms recognized from form: {selected_symptoms_from_form}")
    print(f"DEBUG: Prepared input data for model (first 10 elements): {input_data[:10]}...")
    print(f"DEBUG: Sum of input_data (should match number of selected symptoms): {np.sum(input_data)}")

    input_data = input_data.reshape(1, -1)
    prediction_encoded = general_model.predict(input_data)
    predicted_disease = general_label_encoder.inverse_transform(prediction_encoded)[0]

    print(f"DEBUG: Predicted disease (encoded): {prediction_encoded}")
    print(f"DEBUG: Predicted disease (human-readable): {predicted_disease}")
    print("--- DEBUGGING PREDICT ROUTE END ---")

    predicted_disease_info = DISEASE_INFORMATION.get(predicted_disease)
    if predicted_disease_info:
        print(f"DEBUG: Found detailed info for '{predicted_disease}'.")
        predicted_disease_info['name'] = predicted_disease
    else:
        print(f"DEBUG: No detailed info found for '{predicted_disease}'.")
        predicted_disease_info = {
            "name": predicted_disease,
            "description": current_lang_text['no_description_available'],
            "symptoms_common": current_lang_text['no_symptoms_listed'],
            "precautions": [current_lang_text['no_precautions_available']],
            "treatments": [current_lang_text['no_treatments_available']],
            "more_info_link": "#"
        }

    # No user_name passed here
    return render_template('index.html',
                        symptoms=all_symptoms,
                        prediction_text=f"{current_lang_text['prediction_prefix']} {predicted_disease}",
                        clinics=all_clinics,
                        predicted_disease_details=predicted_disease_info,
                        text=current_lang_text,
                        current_lang=session.get('language', 'en'))

@app.route('/heart_info')
def heart_info_page():
    current_lang_text = get_translation_text()
    # No user_name passed here
    return render_template('heart_info.html', text=current_lang_text, current_lang=session.get('language', 'en'))

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart_disease():
    current_lang_text = get_translation_text()

    if heart_model is None or heart_scaler is None or heart_feature_names is None:
        print("DEBUG: Heart AI model or tools not loaded correctly.")
        return render_template('heart_info.html', text=current_lang_text, current_lang=session.get('language', 'en'),
                            heart_prediction_text=current_lang_text['error_system_not_loaded'])

    print("\n--- DEBUGGING PREDICT ROUTE (Heart Disease AI) ---")
    user_input = request.form.to_dict()
    print(f"DEBUG: Raw user input from heart form: {user_input}")

    input_df = pd.DataFrame(0, index=[0], columns=heart_feature_names)

    for col in numerical_cols:
        if col in user_input and user_input[col].replace('.', '', 1).isdigit():
            input_df[col] = float(user_input[col])
        else:
            print(f"WARNING: Missing or invalid numerical input for '{col}'. Defaulting to 0.")
            input_df[col] = 0.0

    if 'sex' in user_input and user_input['sex'] == 'Male' and 'sex_Male' in heart_feature_names:
        input_df['sex_Male'] = 1

    if 'chest_pain_type' in user_input:
        cp_type = user_input['chest_pain_type']
        encoded_col = f'chest_pain_type_{cp_type}'
        if encoded_col in heart_feature_names:
            input_df[encoded_col] = 1

    if 'fasting_blood_sugar' in user_input and user_input['fasting_blood_sugar'] == 'True' and 'fasting_blood_sugar_True' in heart_feature_names:
        input_df['fasting_blood_sugar_True'] = 1

    if 'rest_ecg' in user_input:
        ecg_type = user_input['rest_ecg']
        encoded_col = f'rest_ecg_{ecg_type}'
        if encoded_col in heart_feature_names:
            input_df[encoded_col] = 1

    if 'exercise_induced_angina' in user_input and user_input['exercise_induced_angina'] == 'Yes' and 'exercise_induced_angina_Yes' in heart_feature_names:
        input_df['exercise_induced_angina_Yes'] = 1

    if 'slope' in user_input:
        slope_type = user_input['slope']
        encoded_col = f'slope_{slope_type}'
        if encoded_col in heart_feature_names:
            input_df[encoded_col] = 1

    if 'vessels_colored_by_flourosopy' in user_input:
        vessels_val = user_input['vessels_colored_by_flourosopy']
        encoded_col = f'vessels_colored_by_flourosopy_{vessels_val}'
        if encoded_col in heart_feature_names:
            input_df[encoded_col] = 1

    if 'thalassemia' in user_input:
        thal_type = user_input['thalassemia']
        encoded_col = f'thalassemia_{thal_type}'
        if encoded_col in heart_feature_names:
            input_df[encoded_col] = 1

    input_for_prediction = input_df[heart_feature_names].fillna(0)
    input_for_prediction[numerical_cols] = heart_scaler.transform(input_for_prediction[numerical_cols])

    print(f"DEBUG: Final input for Heart AI model (first 5 features): {input_for_prediction.iloc[0,:5].values}...")

    prediction_heart_encoded = heart_model.predict(input_for_prediction)[0]
    prediction_proba_heart = heart_model.predict_proba(input_for_prediction)[0].tolist()

    risk_score = prediction_proba_heart[1] * 100
    heart_final_risk = ""
    if risk_score >= 65:
        heart_final_risk = current_lang_text['risk_high']
    elif risk_score >= 35:
        heart_final_risk = current_lang_text['risk_medium']
    else:
        heart_final_risk = current_lang_text['risk_low']

    heart_prediction_text_full = f"{current_lang_text['your_risk_is']} {heart_final_risk} ({risk_score:.2f}%)"

    print(f"DEBUG: Heart AI Predicted: {prediction_heart_encoded} ({heart_final_risk}, {risk_score:.2f}%)")
    print("--- DEBUGGING PREDICT ROUTE END ---")

    # No user_name passed here
    return render_template('heart_info.html',
                        text=current_lang_text,
                        current_lang=session.get('language', 'en'),
                        heart_prediction_text=heart_prediction_text_full,
                        show_heart_result=True)

@app.route('/set_language/<lang>')
def set_language(lang):
    if lang in TRANSLATIONS:
        session['language'] = lang
    else:
        session['language'] = 'en'

    # Determine where to redirect based on 'redirect_to' argument
    # This is for language switcher on login page, etc.
    redirect_to = request.args.get('redirect_to')
    if redirect_to == 'login':
        return redirect(url_for('login_page'))
    elif redirect_to == 'splash':
        return redirect(url_for('splash_screen'))
    else:
        return redirect(request.referrer or url_for('home'))


# --- Run the Flask Application ---
if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)