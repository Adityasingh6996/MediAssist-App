# MediAssist: Your Personal AI Health Helper

## **An AI-Powered Symptom Checker, Disease Information System, Clinic Finder, and Heart Disease Risk Assessor with Multilingual Support & Secure Authentication.**

![MediAssist Logo Placeholder](app/static/images/mediassist_logo.png)
*(Replace this with an actual screenshot of your login page, home page, or heart info page after deployment for better visual appeal on GitHub!)*

## Table of Contents

1.  [Introduction](#introduction)
2.  [Problem Statement](#problem-statement)
3.  [Features](#features)
4.  [Technologies Used](#technologies-used)
5.  [Dataset(s)](#datasets)
6.  [Machine Learning Models](#machine-learning-models)
7.  [Project Structure](#project-structure)
8.  [Local Installation & Setup](#local-installation--setup)
9.  [Usage Guide](#usage-guide)
10. [Deployment](#deployment)
11. [Future Enhancements](#future-enhancements)
12. [Contact](#contact)
13. [License](#license)

## Introduction

MediAssist is an intelligent web application designed to empower individuals, especially those in rural or underserved areas, with accessible and preliminary healthcare information. Leveraging machine learning and a user-friendly interface, it provides insights into potential health conditions based on symptoms, offers detailed information about diseases, helps locate nearby clinics, and assesses heart disease risk, all available in multiple languages.

## Problem Statement


In many regions, particularly in India, access to immediate medical consultation or clear health information can be challenging due to geographical distance, lack of awareness, or language barriers. This project aims to bridge this gap by offering:
* A preliminary tool for symptom understanding.
* Actionable information on common diseases.
* Easy access to local healthcare facility information.
* Early risk assessment for critical conditions like heart disease.
* Bilingual support (English & Hindi) to cater to a diverse user base.

## Features

* **Secure User Authentication:**
    * **Google OAuth 2.0 Login:** Users can securely sign in using their Google accounts.
    * **Animated Splash Screen:** A professional 3-second animated splash screen transitions after successful login.
    * **Session Management:** Maintains user login state across the application.
* **AI-Powered Symptom Checker:**
    * Users select a list of symptoms they are experiencing.
    * The system uses a trained Machine Learning model to suggest a preliminary potential disease.
* **Detailed Disease Information:**
    * For the predicted disease, the application provides:
        * A concise description.
        * Common associated symptoms.
        * General precautions.
        * Recommended treatments.
        * A link for more information (e.g., Wikipedia).
* **Dedicated Heart Disease Risk Assessment AI:**
    * A specialized section where users can input specific cardiovascular parameters (age, blood pressure, cholesterol, chest pain type, etc.).
    * A separate, trained Machine Learning model predicts the preliminary risk (Low, Medium, High) of heart disease.
    * Provides important disclaimers.
* **Local Clinic/Hospital Finder:**
    * Displays a list of nearby clinics and hospitals (using static dummy data for demonstration).
    * Includes a filtering option to search for facilities by locality (e.g., Meerut, Rubyapur).
* **Multilingual Support (English & Hindi):**
    * Users can seamlessly switch the entire user interface text between English and Hindi, enhancing accessibility.
* **User-Friendly Interface:** Built with Flask for a responsive and intuitive web experience.

## Technologies Used

* **Backend:**
    * Python 3.x
    * Flask (Web Framework)
    * Scikit-learn (Machine Learning)
    * Pandas & NumPy (Data Manipulation & Numerical Operations)
    * `requests-oauthlib` (Google OAuth 2.0 Integration)
    * `python-dotenv` (Environment Variable Management for secrets)
    * `pickle` (Model Serialization)
* **Frontend:**
    * HTML5
    * CSS3
    * JavaScript
    * Jinja2 (Flask Templating Engine)
* **Version Control:**
    * Git
    * GitHub
* **Deployment:**
    * Heroku (Planned/Ready for deployment)

## Dataset(s)

1.  **General Disease Symptom Data (`data/training.csv`, `data/testing.csv`):**
    * **Source:** Adapted from public symptom-disease datasets (e.g., Kaggle).
    * **Content:** Contains 132 binary symptom features and 41 distinct disease labels.
    * **Characteristics:** Clean, balanced dataset with 4920 training entries and 42 testing entries.
2.  **Heart Disease Data (`data/heart.csv`):**
    * **Source:** UCI Heart Disease Dataset (e.g., from Kaggle).
    * **Content:** Contains 13 cardiovascular features (e.g., age, sex, blood pressure, cholesterol, chest pain type) and a binary target variable (0 for no heart disease, 1 for heart disease).
    * **Characteristics:** Clean dataset with 1025 entries, requiring specific preprocessing for numerical and categorical features.
3.  **Clinic Data (`data/clinics.csv`):**
    * **Content:** A small, static CSV file with dummy clinic/hospital names, addresses, localities, specialties, and contact numbers for demonstration purposes.
4.  **Disease Information (`data/disease_info.py`):**
    * **Content:** A Python dictionary providing descriptions, common symptoms, precautions, and treatments for various diseases, accessed by the predicted disease name.
5.  **Translation Data (`data/translations.py`):**
    * **Content:** A Python dictionary mapping English UI strings to Hindi translations, used for multilingual support.

## Machine Learning Models

1.  **General Disease Prediction:**
    * **Model:** Random Forest Classifier
    * **Accuracy:** Achieved ~97.62% accuracy on the test set.
    * **Purpose:** Predicts one of 41 possible diseases based on user-selected symptoms.
2.  **Heart Disease Risk Assessment:**
    * **Model:** Logistic Regression Classifier
    * **Accuracy:** Achieved ~82.44% accuracy on the test set.
    * **Purpose:** Predicts the presence (High/Medium/Low Risk) of heart disease based on specific cardiovascular parameters.

## Project Structure

MediAssist_Project/
├── app/
│   ├── templates/          # HTML templates for web pages
│   │   ├── index.html      # Main symptom checker page
│   │   ├── heart_info.html # Heart health info & AI form page
│   │   ├── login.html      # User login page
│   │   └── splash.html     # Animated splash screen
│   └── static/             # Static files (CSS, images)
│       ├── style.css
│       └── images/         # Project logo, Google logo
├── data/                   # Datasets and static information
│   ├── training.csv        # Symptom checker training data
│   ├── testing.csv         # Symptom checker testing data
│   ├── clinics.csv         # Clinic finder data
│   │   # (Ensure init.py is here for Python to treat 'data' as a package)
│   ├── disease_info.py     # Detailed disease descriptions
│   └── translations.py     # Multilingual text strings
├── models/                 # Saved machine learning models and preprocessing tools
│   ├── disease_prediction_model.pkl
│   ├── label_encoder.pkl
│   ├── heart_disease_model.pkl
│   ├── heart_scaler.pkl
│   └── heart_feature_names.pkl
├── .env                    # Environment variables (Google Client ID/Secret - NEVER push to GitHub!)
├── .gitignore              # Specifies files/folders Git should ignore (.env, venv)
├── requirements.txt        # List of Python dependencies
├── Procfile                # Heroku deployment instructions
├── app.py                  # Main Flask application logic
├── data_preparation.py     # Script for general disease data preprocessing
├── model_training.py       # Script for general disease model training
├── heart_ai_preparation.py # Script for heart disease data preprocessing
└── heart_ai_training.py    # Script for heart disease model training


## Local Installation & Setup

Follow these steps to set up and run MediAssist on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/Adityasingh6996/MediAssist-App.git](https://github.com/Adityasingh6996/MediAssist-App.git)
    cd MediAssist-App
    ```

2.  **Create and Activate Virtual Environment:**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Google Cloud Project Setup for OAuth:**
    * Go to [Google Cloud Console](https://console.cloud.google.com/).
    * Create a new project (e.g., `MediAssist-Authentication`).
    * Go to **APIs & Services > OAuth consent screen**:
        * Choose `External` user type, fill in "App name" (e.g., `MediAssist Login`), "User support email", "Developer contact information".
        * Add **your own Google account email** as a `Test user`.
        * Save and go back to the dashboard.
    * Go to **APIs & Services > Credentials**:
        * Click `+ CREATE CREDENTIALS` -> `OAuth client ID`.
        * Select `Web application`.
        * **Name:** `MediAssist Web Client`.
        * **Authorized JavaScript origins:** Add `http://localhost:5000`.
        * **Authorized redirect URIs:** Add `http://localhost:5000/callback`.
        * Click `CREATE`.
        * **CRUCIALLY: Copy your `Client ID` and `Client Secret`. Save them securely!**

5.  **Configure Environment Variables:**
    * Create a file named `.env` in the root of your `MediAssist-App` directory.
    * Add your Client ID and Client Secret to it:
        ```
        GOOGLE_CLIENT_ID="YOUR_COPIED_CLIENT_ID"
        GOOGLE_CLIENT_SECRET="YOUR_COPIED_CLIENT_SECRET"
        FLASK_SECRET_KEY="A_VERY_LONG_RANDOM_STRING_HERE_FOR_FLASK_SESSION_SECURITY"
        ```
        *(For `FLASK_SECRET_KEY`, generate a random string, e.g., using `python -c 'import os; print(os.urandom(24))'` in your Python terminal).*
    * Ensure `.env` is listed in your `.gitignore` file.

6.  **Place Logo Images:**
    * Ensure `mediassist_logo.png` and `google_logo.png` are in `app/static/images/`.

7.  **Run Data Preparation and Model Training Scripts:**
    * These scripts preprocess the data and save the trained ML models and scalers.
    * Run them in order from your Command Prompt (with `(venv)` active):
        ```bash
        python data_preparation.py
        python model_training.py
        python heart_ai_preparation.py
        python heart_ai_training.py
        ```
    * Verify `models/` folder contains `.pkl` files and `data/` contains `__init__.py`.

8.  **Run the Flask Application:**
    ```bash
    python app.py
    ```
    The application will run on `http://127.0.0.1:5000/`.

## Usage Guide

1.  **Access:** Open your browser and go to `http://127.0.0.1:5000/`.
2.  **Login:**
    * You will be redirected to the login page.
    * Click "Sign in with Google".
    * Follow Google's prompts to select your account (remember, you must be a "Test user").
    * After successful login, an animated splash screen will appear.
    * You will then be redirected to the main MediAssist page.
3.  **Language Switcher:**
    * Click "English" or "हिंदी" buttons at the top right to switch the UI language.
4.  **Symptom Checker:**
    * Select symptoms from the checkboxes.
    * Click "Get Preliminary Suggestion" to see the predicted disease and detailed information.
5.  **Heart Health Information & AI:**
    * Click the "Special Information for Heart Patients" link.
    * Fill in the detailed form for cardiovascular parameters.
    * Click "Assess My Risk" to get a probabilistic heart disease risk assessment.
6.  **Clinic Finder:**
    * Scroll down the main page to see nearby dummy clinics.
    * Use the "Filter by Locality" input to search (e.g., "Meerut", "Rubyapur").
7.  **Logout:** Click the "Logout" button at the top right to end your session.

## Deployment

This project is configured for deployment on Heroku.

1.  **Heroku Account & CLI:** Ensure you have a [Heroku account](https://www.heroku.com/) and the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) installed and logged in (`heroku login`).
2.  **Git & GitHub:** Ensure your project is pushed to a GitHub repository (as outlined in Installation steps).
3.  **Procfile:** The `Procfile` at the root tells Heroku how to run your web app: `web: python app.py`.
4.  **Buildpacks:** You might need to manually add the `heroku/python` buildpack to your Heroku app settings.
5.  **Environment Variables:** Set `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, and `FLASK_SECRET_KEY` as Config Vars in your Heroku app's settings (under `Settings -> Reveal Config Vars`). Ensure `http://localhost:5000/callback` is replaced with `https://YOUR-HEROKU-APP-NAME.herokuapp.com/callback` in your Google Cloud Console "Authorized redirect URIs" for the deployed app.
6.  **Deploy:** From your project's root directory (with Git initialized and committed), deploy to Heroku:
    ```bash
    heroku create your-app-name-here
    git push heroku main
    heroku open
    ```

## Future Enhancements

* **Real-time Clinic Data:** Integrate with Google Maps Places API or similar services for live, comprehensive clinic/hospital data.
* **Medical Report Interpreter (OCR):** Implement OCR to allow users to upload images/PDFs of medical reports for interpretation (requires Tesseract OCR integration).
* **Personalized Dashboards:** Store user health data securely (with consent) and provide personalized health dashboards and trends.
* **Teleconsultation Integration:** Add features for booking virtual consultations with doctors.
* **AI Chatbot:** Implement a conversational AI for symptom pre-screening or health FAQs.
* **More ML Models:** Incorporate prediction models for other diseases (e.g., diabetes risk).
* **Mobile Application:** Develop a native mobile app (Android/iOS) for wider accessibility.

## Contact

For any questions or collaborations, feel free to contact:

**Aditya Singh**
* **GitHub:** [https://github.com/Adityasingh6996](https://github.com/Adityasingh6996)
* **Email:** your.email@example.com *(Replace with your actual email)*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. (You might need to create a LICENSE file manually, or choose a different license.)
