import streamlit as st
import pandas as pd
import joblib


# =========================
# LOAD MODEL AND SCALER
# =========================

model = joblib.load(
    "E:\ml assignment\diabetes_dataset_prediction\models/best_model_random_forest.pkl"
)

scaler = joblib.load(
    "E:\ml assignment\diabetes_dataset_prediction\models/scaler.pkl"
)


# =========================
# PAGE TITLE
# =========================

st.title("Diabetes Prediction System")

st.write(
    "Enter patient health details below"
)


# =========================
# USER INPUTS
# =========================

age = st.number_input(

    "Age",

    min_value=1,

    max_value=100,

    value=45

)

bmi = st.number_input(

    "BMI",

    min_value=10.0,

    max_value=60.0,

    value=25.0

)

HbA1c_level = st.number_input(

    "HbA1c Level",

    min_value=3.0,

    max_value=15.0,

    value=5.5

)

blood_glucose_level = st.number_input(

    "Blood Glucose Level",

    min_value=50,

    max_value=400,

    value=100

)

hypertension = st.selectbox(

    "Hypertension",

    [0, 1]

)

heart_disease = st.selectbox(

    "Heart Disease",

    [0, 1]

)

smoking_history = st.selectbox(

    "Smoking History",

    [0, 1, 2, 3, 4]

)


# PREDICTION BUTTON

if st.button("Predict"):


    # CREATE INPUT DATAFRAME


    new_patient = pd.DataFrame({

        'age': [age],

        'bmi': [bmi],

        'HbA1c_level': [HbA1c_level],

        'blood_glucose_level': [blood_glucose_level]

    })


    # =========================
    # SCALE FEATURES
    # =========================

    columns_to_scale = [

        'age',

        'bmi',

        'HbA1c_level',

        'blood_glucose_level'

    ]

    new_patient[columns_to_scale] = scaler.transform(

        new_patient[columns_to_scale]

    )


    # =========================
    # EXTRACT SCALED VALUES
    # =========================

    scaled_age = new_patient['age'][0]

    scaled_bmi = new_patient['bmi'][0]

    scaled_hba1c = new_patient['HbA1c_level'][0]

    scaled_glucose = new_patient['blood_glucose_level'][0]


    # =========================
    # FEATURE ENGINEERING
    # =========================

    # Risk Score

    risk_score = (

        hypertension +

        heart_disease +

        int(smoking_history >= 4) +

        int(scaled_bmi >= 0) +

        int(scaled_age >= 0)

    )


    # Blood Sugar Risk

    blood_sugar_risk = (

        int(scaled_hba1c >= 0.9) +

        int(scaled_glucose >= 0.5)

    )


    # Age-Hypertension

    age_hypertension = (

        scaled_age *

        hypertension

    )


    # BMI-Glucose

    bmi_glucose = (

        scaled_bmi *

        scaled_glucose

    )


    # =========================
    # FINAL MODEL INPUT
    # =========================

    final_input = pd.DataFrame({

        'blood_glucose_level': [scaled_glucose],

        'HbA1c_level': [scaled_hba1c],

        'blood_sugar_risk': [blood_sugar_risk],

        'risk_score': [risk_score],

        'age': [scaled_age],

        'bmi_glucose': [bmi_glucose],

        'bmi': [scaled_bmi],

        'hypertension': [hypertension],

        'age_hypertension': [age_hypertension],

        'heart_disease': [heart_disease],

        'smoking_history': [smoking_history]

    })


    # =========================
    # PREDICTION
    # =========================

    prediction = model.predict(
        final_input
    )[0]


    probability = model.predict_proba(
        final_input
    )[0][1]


    # =========================
    # OUTPUT
    # =========================

    if prediction == 1:

        st.error(
            "Patient is Diabetic"
        )

    else:

        st.success(
            "Patient is Non-Diabetic"
        )
