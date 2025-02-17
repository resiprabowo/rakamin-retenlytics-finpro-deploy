import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model_11_features.pkl")

# Judul aplikasi
st.title("Prediksi Employee Attrition")

# Upload file Excel
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca Excel
    df = pd.read_excel(uploaded_file)
    st.write("### Data yang Diupload:")
    st.dataframe(df)

    # Pastikan hanya memilih fitur yang sesuai dengan model
    selected_features = [
        "EmployeeID", "TotalWorkHours", "DistanceFromHome",
        "Age", "TotalWorkingYears", "YearsPerPromotion",
        "MaritalStatus", "YearsWithCurrManager",
        "PerformanceToSatisfactionRatio", "NumCompaniesWorked",
        "TrainingTimesLastYear"
    ]

    # Filter data agar sesuai dengan fitur yang dibutuhkan model
    try:
        df_selected = df[selected_features]
        
        # Pastikan MaritalStatus diubah ke numerik jika perlu
        if df_selected["MaritalStatus"].dtype == "object":
            df_selected["MaritalStatus"] = df_selected["MaritalStatus"].astype("category").cat.codes
        
        # Prediksi
        predictions = model.predict(df_selected.drop(columns=["EmployeeID"]))  # Hapus EmployeeID sebelum prediksi

        # Tambahkan kolom hasil prediksi ke dataframe
        df["Attrition_Prediction"] = predictions

        # Tampilkan hasil prediksi
        st.write("### Hasil Prediksi:")
        st.dataframe(df)

        # Download hasil
        excel_output = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Hasil", data=excel_output, file_name="prediksi_employee.csv", mime="text/csv")

    except KeyError as e:
        st.error(f"Kolom yang diperlukan tidak ditemukan: {e}")

