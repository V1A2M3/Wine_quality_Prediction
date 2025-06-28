import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.title("🍷 Wine Quality Prediction")
st.markdown("Predict wine quality (Low / Medium / High) based on its chemical features.")

# Load model and scaler
model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")

features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
            'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
            'pH', 'sulphates', 'alcohol']

user_input = {}
for feature in features:
    user_input[feature] = st.slider(f"{feature}", float(0), float(20), float(5))

input_df = pd.DataFrame([user_input])
scaled = scaler.transform(input_df)

prediction = model.predict(scaled)[0]
quality_map = {0: 'Low', 1: 'Medium', 2: 'High'}

st.success(f"Predicted Wine Quality: *{quality_map[prediction]}*")
