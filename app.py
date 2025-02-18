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
        # Cek apakah semua kolom tersedia dalam file yang di-upload
        missing_cols = [col for col in selected_features if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom berikut tidak ditemukan dalam file: {missing_cols}")
        else:
            # Pilih hanya kolom yang dibutuhkan
            df_selected = df[selected_features].copy()

            # Pastikan MaritalStatus dikonversi menjadi one-hot encoding jika masih dalam bentuk teks
            if "MaritalStatus" in df.columns and df["MaritalStatus"].dtype == "object":
                df = pd.get_dummies(df, columns=["MaritalStatus"], drop_first=False)

                # Pastikan semua kategori tersedia setelah encoding
                for col in ["MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"]:
                    if col not in df:
                        df[col] = 0  # Tambahkan kolom yang hilang dengan nilai 0

                # Update df_selected setelah encoding
                df_selected = df[selected_features]

            # Hapus EmployeeID sebelum prediksi
            df_selected = df_selected.drop(columns=["EmployeeID"])

            # Prediksi
            predictions = model.predict(df_selected)

            # Tambahkan hasil prediksi ke dataframe asli
            df["Attrition_Prediction"] = predictions

            # Tampilkan hasil prediksi
            st.write("### Hasil Prediksi:")
            st.dataframe(df)

            # Download hasil prediksi
            excel_output = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Hasil", data=excel_output, file_name="prediksi_employee.csv", mime="text/csv")

    except KeyError as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")

