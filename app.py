import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_12_features.pkl")  # Ganti dengan nama model baru

# Judul aplikasi
st.title("Prediksi Employee Attrition")

# Upload file Excel
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca Excel
    df = pd.read_excel(uploaded_file)

    # Cek dan konversi tipe data untuk PerformanceToSatisfactionRatio
    if df['PerformanceToSatisfactionRatio'].dtype == 'object':
        df['PerformanceToSatisfactionRatio'] = pd.to_numeric(df['PerformanceToSatisfactionRatio'], errors='coerce')

    # Cek apakah ada nilai NaN setelah konversi
    if df['PerformanceToSatisfactionRatio'].isnull().any():
        st.error("Kolom PerformanceToSatisfactionRatio mengandung nilai yang tidak valid.")
    else:
        # Konversi MaritalStatus menjadi one-hot encoding
        df = pd.get_dummies(df, columns=["MaritalStatus"], drop_first=True)

        # Pastikan semua kolom yang diperlukan ada
        required_columns = [
            "EmployeeID", "TotalWorkHours", "DistanceFromHome", "Age",
            "TotalWorkingYears", "YearsPerPromotion", "YearsWithCurrManager",
            "PerformanceToSatisfactionRatio", "NumCompaniesWorked",
            "TrainingTimesLastYear", "MaritalStatus_Married", "MaritalStatus_Single"
        ]

        for col in required_columns:
            if col not in df.columns:
                st.error(f"Kolom {col} tidak ditemukan dalam data.")
                break
        else:
            # Pisahkan fitur dan hapus EmployeeID sebelum prediksi
            df_selected = df[required_columns].copy()
            df_selected = df_selected.drop(columns=["EmployeeID"])

            # Prediksi
            predictions = model.predict(df_selected)

            # Tambahkan hasil prediksi ke DataFrame
            df["Attrition_Prediction"] = predictions

            # Tampilkan hasil
            st.write("### Hasil Prediksi:")
            st.dataframe(df)

            # Download hasil prediksi
            excel_output = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil", data=excel_output, file_name="prediksi_employee.csv", mime="text/csv")









