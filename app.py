
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# =========================
# KONFIGURASI APLIKASI
# =========================
st.set_page_config(
    page_title="Academic Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "oulad_models.joblib"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

if not MODEL_PATH.exists():
    st.error("File oulad_models.joblib tidak ditemukan.")
    st.stop()

bundle = load_bundle()

models = bundle["models"]
categorical_cols = bundle["categorical_cols"]
numeric_cols = bundle["numeric_cols"]
category_options = bundle["category_options"]
metrics = bundle.get("metrics", [])

# =========================
# TAMPILAN
# =========================
st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.2);
    padding: 18px;
    border-radius: 14px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("🎓 Academic Analytics")
st.sidebar.caption("Student Performance Prediction")

page = st.sidebar.radio(
    "MENU UTAMA",
    ["Dashboard", "Prediksi Akademik", "Evaluasi Model"]
)

st.sidebar.divider()
st.sidebar.caption("Dataset: OULAD")

# =========================
# DASHBOARD
# =========================
if page == "Dashboard":
    st.title("Academic Performance Prediction")
    st.write(
        "Sistem prediksi performa akademik mahasiswa "
        "berbasis Machine Learning menggunakan dataset OULAD."
    )

    st.info(
        "Sistem menggunakan atribut mahasiswa dan aktivitas belajar "
        "pada 28 hari pertama sesuai fitur pelatihan."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Model tersedia", len(models))
    col2.metric("Jumlah fitur", len(categorical_cols) + len(numeric_cols))
    col3.metric("Kategori target", len(bundle["target_labels"]))

    st.subheader("Algoritma yang digunakan")

    st.markdown("""
    - **Support Vector Machine (SVM)** sebagai model utama.
    - **Gaussian Naive Bayes** sebagai model pembanding.
    """)

    st.subheader("Kategori prediksi")
    st.write(", ".join(bundle["target_labels"]))

    st.warning(
        "Prediksi merupakan estimasi berdasarkan pola data historis, "
        "bukan jaminan hasil akademik mahasiswa."
    )

# =========================
# PREDIKSI AKADEMIK
# =========================
elif page == "Prediksi Akademik":
    st.title("Prediksi Performa Akademik")
    st.write("Masukkan data sesuai atribut yang digunakan saat pelatihan.")

    with st.form("prediction_form"):
        input_data = {}

        st.subheader("Informasi mahasiswa")
        left, right = st.columns(2)

        for i, feature in enumerate(categorical_cols):
            options = category_options.get(feature, [])

            if not options:
                st.error(f"Pilihan kategori {feature} tidak tersedia.")
                st.stop()

            with (left if i % 2 == 0 else right):
                input_data[feature] = st.selectbox(
                    feature.replace("_", " ").title(),
                    options=options
                )

        st.subheader("Aktivitas dan data numerik")
        left, right = st.columns(2)

        for i, feature in enumerate(numeric_cols):
            with (left if i % 2 == 0 else right):
                input_data[feature] = st.number_input(
                    feature.replace("_", " ").title(),
                    value=0.0,
                    step=1.0
                )

        selected_model = st.selectbox(
            "Pilih algoritma",
            list(models.keys())
        )

        submitted = st.form_submit_button(
            "🔍 Jalankan Prediksi",
            use_container_width=True
        )

    if submitted:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[categorical_cols + numeric_cols]

        try:
            prediction = models[selected_model].predict(input_df)[0]

            st.success("Prediksi berhasil dilakukan.")
            st.subheader("Hasil Prediksi")

            st.metric("Kategori hasil prediksi", str(prediction))
            st.caption(f"Model yang digunakan: {selected_model}")

        except Exception as e:
            st.error(f"Prediksi gagal: {e}")

# =========================
# EVALUASI MODEL
# =========================
elif page == "Evaluasi Model":
    st.title("Evaluasi Model")
    st.write("Perbandingan berdasarkan hasil pengujian yang tersimpan.")

    if metrics:
        metrics_df = pd.DataFrame(metrics)

        st.dataframe(
            metrics_df,
            use_container_width=True,
            hide_index=True
        )

        chart_cols = [
            col for col in
            ["Accuracy", "Precision_macro", "Recall_macro", "F1_macro"]
            if col in metrics_df.columns
        ]

        if chart_cols:
            st.subheader("Perbandingan metrik")
            st.bar_chart(
                metrics_df.set_index("Model")[chart_cols]
            )

    else:
        st.warning(
            "Metrik belum tersimpan. Periksa hasil evaluasi di Colab."
        )

    st.caption(
        "Metrik menunjukkan kinerja pada data pengujian, "
        "bukan kepastian prediksi untuk seorang mahasiswa."
    )