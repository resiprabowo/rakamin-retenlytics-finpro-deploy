import streamlit as st
import lightgbm as lgb
import pickle
import pandas as pd
import os

# Fungsi untuk load model
@st.cache_data
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    with open(model_path, "rb") as file:
        model = pickle.load(file)
    return model

# Load model
model = load_model()

# Judul Aplikasi
st.title("Employee Attrition Prediction")

# Input Features
st.sidebar.header("Masukkan Data Karyawan")
age = st.sidebar.number_input("Usia", min_value=18, max_value=65, value=30)
salary = st.sidebar.number_input("Gaji", min_value=1000, max_value=100000, value=5000)
years_at_company = st.sidebar.number_input("Tahun di Perusahaan", min_value=0, max_value=40, value=5)
satisfaction = st.sidebar.slider("Kepuasan Kerja (0-1)", 0.0, 1.0, 0.5)

# Convert input to DataFrame
input_data = pd.DataFrame(
    [[age, salary, years_at_company, satisfaction]],
    columns=["Age", "Salary", "YearsAtCompany", "Satisfaction"]
)

# Predict
if st.sidebar.button("Prediksi"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    # Hasil Prediksi
    if prediction == 1:
        st.error(f"Karyawan berisiko resign dengan probabilitas {probability:.2f}")
    else:
        st.success(f"Karyawan cenderung bertahan dengan probabilitas {1 - probability:.2f}")

