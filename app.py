import streamlit as st
import pandas as pd
import joblib
import numpy as np
from io import BytesIO

# Load model
model = joblib.load("model_12_features.pkl")  # Pastikan model ini mendukung predict_proba()

# Optimal Threshold dari evaluasi sebelumnya
OPTIMAL_THRESHOLD = 0.8192

# Judul aplikasi
st.title("Prediksi Employee Attrition")

# Upload file Excel
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca Excel
    df = pd.read_excel(uploaded_file)

    # Konversi tipe data jika perlu
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except ValueError:
                st.error(f"Kolom {col} mengandung nilai yang tidak valid.")
                st.stop()

    # Isi nilai NaN
    df = df.fillna(df.mean(numeric_only=True))

    # One-hot encoding untuk MaritalStatus
    df = pd.get_dummies(df, columns=['MaritalStatus'], drop_first=False)

    # Pastikan semua kolom sesuai dengan model
    expected_columns = [
        'TotalWorkHours', 'DistanceFromHome', 'Age', 'TotalWorkingYears',
        'YearsPerPromotion', 'YearsWithCurrManager', 'PerformanceToSatisfactionRatio',
        'NumCompaniesWorked', 'TrainingTimesLastYear', 'MaritalStatus_Divorced',
        'MaritalStatus_Married', 'MaritalStatus_Single'
    ]
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0  # Tambahkan kolom jika tidak ada

    # Urutkan kolom sesuai urutan yang diharapkan model
    df_selected = df[expected_columns]

    # Lakukan prediksi probabilitas
    try:
        pred_proba = model.predict_proba(df_selected)[:, 1]  # Ambil probabilitas kelas positif
        predictions = (pred_proba >= OPTIMAL_THRESHOLD).astype(int)  # Terapkan threshold untuk prediksi akhir

        # Simpan hasil ke dataframe
        df['Attrition_Probability'] = pred_proba
        df['Attrition_Prediction'] = predictions

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")
        st.stop()

    # Tampilkan hasil prediksi
    st.write("### Hasil Prediksi:")
    st.dataframe(df)

    # Tambahkan grafik distribusi probabilitas (opsional)
    st.write("### Distribusi Probabilitas Resign")
    st.bar_chart(df['Attrition_Probability'])

    # Tombol download
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    processed_data = output.getvalue()
    st.download_button(
        label="Download Hasil",
        data=processed_data,
        file_name="hasil_prediksi.xlsx",
        mime="application/vnd.ms-excel"
    )









