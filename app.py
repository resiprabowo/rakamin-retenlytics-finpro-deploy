import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

# Load model
model = joblib.load("model_12_features.pkl")

# Judul aplikasi
st.title("Prediksi Employee Attrition")

# Upload file Excel
uploaded_file = st.file_uploader("Upload Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca Excel
    df = pd.read_excel(uploaded_file)

    # Tampilkan data yang diunggah
    st.write("### Data yang Diunggah:")
    st.dataframe(df)

    # Konversi tipe data kolom PerformanceToSatisfactionRatio
    try:
        df['PerformanceToSatisfactionRatio'] = pd.to_numeric(df['PerformanceToSatisfactionRatio'], errors='raise')
    except ValueError:
        st.error("Kolom PerformanceToSatisfactionRatio mengandung nilai yang tidak valid.")
        st.stop()

    # One-hot encoding untuk MaritalStatus
    df = pd.get_dummies(df, columns=['MaritalStatus'], drop_first=True)

    # Pastikan semua kolom yang dibutuhkan ada
    required_columns = ['TotalWorkHours', 'DistanceFromHome', 'Age', 'TotalWorkingYears',
                        'YearsPerPromotion', 'YearsWithCurrManager', 'PerformanceToSatisfactionRatio',
                        'NumCompaniesWorked', 'TrainingTimesLastYear', 'MaritalStatus_Married',
                        'MaritalStatus_Single']  # Kolom one-hot encoding

    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Kolom berikut tidak ditemukan dalam data Anda: {', '.join(missing_columns)}")
        st.stop()

    # Pilih kolom yang sesuai untuk prediksi
    df_selected = df[required_columns]

    # Lakukan prediksi
    try:
        predictions = model.predict(df_selected)
        df['Attrition_Prediction'] = predictions
        st.write("### Hasil Prediksi:")
        st.dataframe(df)

        # Tombol Download
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, sheet_name='predictions', index=False)
        processed_data = output.getvalue()

        st.download_button(
            label="Download Hasil",
            data=processed_data,
            file_name='predictions.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")








