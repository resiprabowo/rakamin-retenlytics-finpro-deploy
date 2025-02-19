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
        # Validasi Kolom
        missing_cols = [col for col in selected_features if col not in df.columns]

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

        # Periksa lagi setelah penyesuaian
        missing_cols = [col for col in selected_features if col not in df.columns]
        if missing_cols:
            st.error(f"Kolom berikut masih belum ditemukan: {missing_cols}")
            st.stop()

        # Pisahkan fitur dan hapus EmployeeID sebelum prediksi
        df_selected = df[selected_features].copy()
        df_selected = df_selected.drop(columns=["EmployeeID"])

        # Pastikan urutan dan nama fitur konsisten
        expected_columns = [
            "EmployeeID", "TotalWorkHours", "DistanceFromHome",
        "Age", "TotalWorkingYears", "YearsPerPromotion",
        "YearsWithCurrManager", "PerformanceToSatisfactionRatio",
        "NumCompaniesWorked", "TrainingTimesLastYear",
        "MaritalStatus_Divorced", "MaritalStatus_Married", "MaritalStatus_Single"
        ]

        # Tambahkan kolom yang hilang dengan nilai 0
        for col in expected_columns:
            if col not in df_selected.columns:
                df_selected[col] = 0

        # Susun ulang sesuai urutan yang diharapkan
        df_selected = df_selected[expected_columns]

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

    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")




