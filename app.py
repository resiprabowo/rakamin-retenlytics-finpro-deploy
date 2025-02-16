import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_10_features.pkl")

# Judul aplikasi
st.title("Prediksi Employee Attrition")

# Upload file CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    # Baca CSV
    df = pd.read_csv(uploaded_file)
    st.write("### Data yang Diupload:")
    st.dataframe(df)

    # Pastikan hanya memilih fitur yang sesuai dengan model
    selected_features = [
        "MonthlyIncome", "TotalWorkHours", "DistanceFromHome",
        "Age", "TotalWorkingYears", "YearsPerPromotion",
        "PercentSalaryHike", "YearsWithCurrManager",
        "PerformanceToSatisfactionRatio", "NumCompaniesWorked"
    ]

    # Filter data agar sesuai dengan fitur yang dibutuhkan model
    df_selected = df[selected_features]

    # Prediksi
    predictions = model.predict(df_selected)

    # Tambahkan kolom hasil prediksi ke dataframe
    df["Attrition_Prediction"] = predictions

    # Tampilkan hasil prediksi
    st.write("### Hasil Prediksi:")
    st.dataframe(df)

    # Download hasil
    csv = df.to_csv(index=False)
    st.download_button("Download Hasil", data=csv, file_name="prediksi_employee.csv", mime="text/csv")

