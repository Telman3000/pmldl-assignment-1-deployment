"""
Stage 3b — Streamlit web application
Talks to the FastAPI model service.
"""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")
st.title("Titanic Survival Predictor")
st.write("Enter passenger details and get a survival prediction from the model API.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        pclass = st.selectbox("Passenger class", options=[1, 2, 3], index=2)
        sex = st.selectbox("Sex", options=["male", "female"])
        age = st.number_input("Age", min_value=0.0, max_value=100.0, value=29.0, step=1.0)
        embarked = st.selectbox("Embarked", options=["S", "C", "Q"], index=0)

    with col2:
        sibsp = st.number_input("Siblings / spouses aboard", min_value=0, max_value=10, value=0)
        parch = st.number_input("Parents / children aboard", min_value=0, max_value=10, value=0)
        fare = st.number_input("Fare", min_value=0.0, value=32.0, step=1.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "Pclass": int(pclass),
        "Sex": sex,
        "Age": float(age),
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "Fare": float(fare),
        "Embarked": embarked,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()

        st.subheader("Prediction")
        st.metric("Outcome", result["survival_label"])
        st.write(f"Survival probability: **{result['survival_probability']:.2%}**")
        st.json(result)
    except requests.exceptions.RequestException as exc:
        st.error(f"Could not reach API at {API_URL}: {exc}")
