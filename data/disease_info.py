# D:/MediAssist_Project/data/disease_info.py

DISEASE_INFORMATION = {
    "Fungal infection": {
        "description": "Fungal infections are common skin conditions caused by fungi. They can appear on various parts of the body and often cause redness, itching, and scaling.",
        "symptoms_common": "Itching, skin rash, nodal skin eruptions, dischromic patches, scaling, blistering, red sore around nose, yellow crust ooze.",
        "precautions": [
            "Keep the affected area clean and dry.",
            "Wear loose-fitting, breathable clothing.",
            "Avoid sharing personal items like towels and clothing.",
            "Use antifungal powders or creams as advised by a doctor.",
            "Maintain good personal hygiene."
        ],
        "treatments": [
            "Antifungal creams, ointments, or sprays (e.g., Clotrimazole, Miconazole).",
            "Oral antifungal medications (e.g., Fluconazole) for severe or widespread infections, prescribed by a doctor."
        ],
        "more_info_link": "https://en.wikipedia.org/wiki/Fungal_infection" # Example link for more details
    },
    "Allergy": {
        "description": "Allergies occur when your immune system reacts to a foreign substance—like pollen, bee venom, or pet dander—that doesn't cause a reaction in most people. Symptoms can range from mild to life-threatening.",
        "symptoms_common": "Continuous sneezing, shivering, chills, watery eyes, runny nose, congestion, high fever, headache, rashes, fatigue.",
        "precautions": [
            "Identify and avoid allergens (substances causing allergy).",
            "Keep your home dust-free and clean.",
            "Use air purifiers if necessary.",
            "Wear masks to avoid pollen or dust.",
            "Carry prescribed allergy medication if prone to severe reactions."
        ],
        "treatments": [
            "Antihistamines (oral or nasal sprays).",
            "Decongestants.",
            "Nasal corticosteroids.",
            "Allergy shots (immunotherapy) for long-term management."
        ],
        "more_info_link": "https://en.wikipedia.org/wiki/Allergy"
    },
    "GERD": {
        "description": "Gastroesophageal Reflux Disease (GERD) is a chronic digestive disease where stomach acid or, occasionally, stomach content, flows back into your food pipe (esophagus). This backwash can irritate the lining of your esophagus.",
        "symptoms_common": "Acidity, vomiting, chest pain, stomach pain, nausea, loss of appetite, mild fever, burning micturition, abdominal pain, diarrhea.",
        "precautions": [
            "Avoid large meals; eat smaller, more frequent meals.",
            "Avoid trigger foods like fatty foods, spicy foods, acidic foods (citrus, tomatoes), chocolate, caffeine, and alcohol.",
            "Don't lie down immediately after eating (wait at least 2-3 hours).",
            "Elevate the head of your bed by 6-8 inches.",
            "Maintain a healthy weight."
        ],
        "treatments": [
            "Antacids for immediate relief.",
            "H2 receptor blockers (e.g., Ranitidine) to reduce acid production.",
            "Proton pump inhibitors (PPIs) (e.g., Omeprazole) for long-term acid reduction and healing.",
            "Lifestyle modifications."
        ],
        "more_info_link": "https://en.wikipedia.org/wiki/Gastroesophageal_reflux_disease"
    },
    "Dengue": {
        "description": "Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. Symptoms typically begin three to fourteen days after infection.",
        "symptoms_common": "High fever, headache, muscle_pain, joint_pain, rash, loss_of_appetite, vomiting, abdominal_pain, malaise, retro-orbital_pain, chills.",
        "precautions": [
            "Prevent mosquito bites: use repellents, wear long sleeves, and ensure windows/doors have screens.",
            "Eliminate mosquito breeding sites: drain stagnant water from pots, tires, and containers.",
            "Sleep under mosquito nets.",
            "Seek medical attention early if symptoms appear.",
        ],
        "treatments": [
            "No specific antiviral treatment.",
            "Symptomatic relief: pain relievers (acetaminophen/paracetamol, AVOID aspirin/ibuprofen).",
            "Rest and fluid intake.",
            "Hospitalization for severe dengue to manage fluid balance and blood pressure."
        ],
        "more_info_link": "https://en.wikipedia.org/wiki/Dengue_fever"
    },
    "Common Cold": {
        "description": "The common cold is a viral infectious disease of the upper respiratory tract that primarily affects the nose. Symptoms usually begin two days after infection.",
        "symptoms_common": "Continuous sneezing, shivering, chills, high fever, headache, congestion, runny nose, cough, malaise, sore throat.",
        "precautions": [
            "Wash hands frequently with soap and water.",
            "Avoid touching your face (eyes, nose, mouth).",
            "Avoid close contact with sick people.",
            "Get enough rest.",
            "Stay hydrated."
        ],
        "treatments": [
            "Rest and fluid intake.",
            "Over-the-counter pain relievers for fever and aches.",
            "Nasal decongestants.",
            "Cough syrups (use with caution and as advised).",
            "Sore throat lozenges."
        ],
        "more_info_link": "https://en.wikipedia.org/wiki/Common_cold"
    },
    # You can add more diseases here following the same structure.
    # For instance, add information for 'Acne', 'Typhoid', etc., from your model's unique disease list.
    # Example for a new one:
    # "Pneumonia": {
    #     "description": "Pneumonia is an inflammatory condition of the lung affecting primarily the small air sacs known as alveoli. It is usually caused by infection with viruses or bacteria.",
    #     "symptoms_common": "High fever, chills, cough, breathlessness, chest pain, fatigue, malaise.",
    #     "precautions": [
    #         "Get vaccinated (pneumococcal and flu vaccines).",
    #         "Wash hands regularly.",
    #         "Avoid smoking.",
    #         "Boost your immune system with healthy diet and exercise."
    #     ],
    #     "treatments": [
    #         "Antibiotics (for bacterial pneumonia).",
    #         "Antivirals (for viral pneumonia, if available).",
    #         "Rest and fluids.",
    #         "Oxygen therapy if needed."
    #     ],
    #     "more_info_link": "https://en.wikipedia.org/wiki/Pneumonia"
    # },
}