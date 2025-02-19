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

    # Fitur yang diperlukan model
    selected_features = [
        "EmployeeID", "TotalWorkHours", "DistanceFromHome",
        "Age", "TotalWorkingYears", "YearsPerPromotion",
        "YearsWithCurrManager", "PerformanceToSatisfactionRatio",
        "NumCompaniesWorked", "TrainingTimesLastYear",
        "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"
    ]

    try:
        # Cek apakah semua kolom yang dibutuhkan tersedia
        missing_cols = [col for col in selected_features if col not in df.columns]

        # Tangkap jika kolom marital status belum di-encode
        if "MaritalStatus" in df.columns and any(col not in df.columns for col in ["MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"]):
            df = pd.get_dummies(df, columns=["MaritalStatus"], drop_first=False)

        # Pastikan semua kategori hasil encoding tersedia
        for col in ["MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"]:
            if col not in df.columns:
                df[col] = 0

        # Validasi ulang setelah encoding
        missing_cols = [col for col in selected_features if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan dalam file: {missing_cols}")
            st.stop()

        # Pilih kolom sesuai model (TANPA menghapus EmployeeID)
        df_selected = df[selected_features].copy()

        # Prediksi (gunakan semua fitur kecuali EmployeeID)
        X_for_prediction = df_selected.drop(columns=["EmployeeID"])
        predictions = model.predict(X_for_prediction)

        # Tambahkan hasil prediksi
        df["Attrition_Prediction"] = predictions

        # Tampilkan hasil prediksi
        st.write("### Hasil Prediksi:")
        st.dataframe(df)

        # Siapkan hasil prediksi untuk diunduh dalam format Excel
        excel_output = pd.ExcelWriter("hasil_prediksi.xlsx", engine="xlsxwriter")
        df.to_excel(excel_output, index=False, sheet_name="Prediksi")
        excel_output.close()

        with open("hasil_prediksi.xlsx", "rb") as f:
            st.download_button("Download Hasil", f, file_name="prediksi_employee.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

