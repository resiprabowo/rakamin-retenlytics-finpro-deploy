import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_13_features.pkl")

# Judul aplikasi
st.title("Prediksi Employee Attrition")

# Upload file Excel
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca Excel
    df = pd.read_excel(uploaded_file)
    st.write("### Data yang Diupload:")
    st.dataframe(df)

    # Cek kolom yang ada
    expected_columns = [
        "EmployeeID", "TotalWorkHours", "DistanceFromHome", "Age",
        "TotalWorkingYears", "YearsPerPromotion", "YearsWithCurrManager",
        "PerformanceToSatisfactionRatio", "NumCompaniesWorked",
        "TrainingTimesLastYear", "MaritalStatus"
    ]

    # Validasi kolom
    missing_cols = [col for col in expected_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Kolom berikut tidak ditemukan: {missing_cols}")
        st.stop()

    # Konversi MaritalStatus menjadi one-hot encoding
    df = pd.get_dummies(df, columns=["MaritalStatus"], drop_first=False)

    # Tambahkan kolom dummy jika tidak ada
    marital_statuses = ["MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"]
    for status in marital_statuses:
        if status not in df.columns:
            df[status] = 0  # Tambahkan kolom dengan nilai 0

    # Fitur akhir setelah encoding
    final_features = [
        "EmployeeID", "TotalWorkHours", "DistanceFromHome", "Age",
        "TotalWorkingYears", "YearsPerPromotion", "YearsWithCurrManager",
        "PerformanceToSatisfactionRatio", "NumCompaniesWorked", "TrainingTimesLastYear",
        "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"
    ]

    # Pisahkan fitur dan hapus EmployeeID sebelum prediksi
    df_selected = df[final_features].copy()
    df_selected = df_selected.drop(columns=["EmployeeID"])

    # Prediksi
    predictions = model.predict(df_selected)

    # Tambahkan hasil prediksi
    df["Attrition_Prediction"] = predictions

    # Tampilkan hasil
    st.write("### Hasil Prediksi:")
    st.dataframe(df)

    # Download hasil prediksi
    excel_output = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Hasil", data=excel_output, file_name="prediksi_employee.csv", mime="text/csv")














