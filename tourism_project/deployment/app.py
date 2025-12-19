
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# Load trained model from HF MODEL repository
model_path = hf_hub_download(
    repo_id="Shalini94/tourism-model",
    filename="tourism_best_model.joblib"
)

model = joblib.load(model_path)

st.title("Visit With Us – Wellness Prediction")

st.write("""
This application predicts wellness outcomes based on user inputs.
""")

# User Inputs (match your training features)
age = st.number_input("Age", 18, 100, 30)
duration = st.number_input("Trip Duration (days)", 1, 30, 5)
monthly_income = st.number_input("Monthly Income", 1000, 100000, 30000)
stress_level = st.slider("Stress Level", 1, 10, 5)

# Create DataFrame
input_df = pd.DataFrame([{
    "Age": age,
    "DurationOfTrip": duration,
    "MonthlyIncome": monthly_income,
    "StressLevel": stress_level
}])

# Prediction
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted Wellness Outcome: **{prediction}**")
