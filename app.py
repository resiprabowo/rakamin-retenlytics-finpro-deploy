import streamlit as st
import lightgbm as lgb
import pickle
import pandas as pd

import os

model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

st.title("Employee Attrition Prediction")

# Input Features
age = st.number_input("Age", min_value=18, max_value=65, value=30)
salary = st.number_input("Salary", min_value=1000, max_value=20000, value=5000)
experience = st.number_input("Years of Experience", min_value=0, max_value=40, value=5)

# Convert input to DataFrame
input_data = pd.DataFrame([[age, salary, experience]], columns=["Age", "Salary", "Experience"])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"Prediction: {'Attrition' if prediction[0] == 1 else 'Stay'}")

