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

    # Bersihkan dan konversi kolom PerformanceToSatisfactionRatio
    df['PerformanceToSatisfactionRatio'] = pd.to_numeric(df['PerformanceToSatisfactionRatio'], errors='coerce')
    df.dropna(subset=['PerformanceToSatisfactionRatio'], inplace=True)  # Hapus baris dengan nilai NaN
    df['PerformanceToSatisfactionRatio'] = df['PerformanceToSatisfactionRatio'].astype(float)  # Pastikan tipe data float

    # Tampilkan data yang diupload
    st.write("### Data yang Diupload:")
    st.dataframe(df)

    # Fitur yang diperlukan model
    selected_features = [
        "EmployeeID",
        "TotalWorkHours",
        "DistanceFromHome",
        "Age",
        "TotalWorkingYears",
        "YearsPerPromotion",
        "YearsWithCurrManager",
        "PerformanceToSatisfactionRatio",
        "NumCompaniesWorked",
        "TrainingTimesLastYear",
        "MaritalStatus_Divorced",
        "MaritalStatus_Married",
        "MaritalStatus_Single"
    ]

    # Cek apakah kolom MaritalStatus ada
    if "MaritalStatus" in df.columns:
        # Konversi MaritalStatus menjadi one-hot encoding
        df = pd.get_dummies(df, columns=["MaritalStatus"], drop_first=False)

        # Tambahkan kolom MaritalStatus yang hilang jika tidak ada
        for col in ["MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"]:
            if col not in df.columns:
                df[col] = 0  # Tambahkan kolom dengan nilai 0

    # Pastikan semua kolom yang diperlukan ada
    for col in selected_features:
        if col not in df.columns:
            df[col] = 0

    # Pisahkan fitur dan hapus EmployeeID sebelum prediksi
    df_selected = df[selected_features].copy()
    df_selected = df_selected.drop(columns=["EmployeeID"])

    # Debugging: Cek apakah kolom sudah sesuai
    st.write("Kolom di DataFrame Setelah Penyesuaian:", df_selected.columns.tolist())
    st.write("Jumlah Fitur di Input:", df_selected.shape[1])
    st.write("Jumlah Fitur di Model:", model.n_features_in_)

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










