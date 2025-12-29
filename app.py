import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Judul dan Konfigurasi
st.set_page_config(page_title="Deteksi Penyakit Daun Karet", layout="centered")
st.title("🍂 Sistem Deteksi Keparahan Penyakit Gugur Daun Karet")
st.write("Implementasi Arsitektur SE-ResNet-50")
st.markdown("---")

# 2. Fungsi Load Model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("my_best_model.h5")

model = load_model()

# 3. Daftar Kelas (Sesuaikan dengan urutan tingkat keparahan di skripsi Anda)
# Contoh: Sehat, Ringan, Sedang, Parah
class_names = ['Sehat', 'Tingkat Ringan', 'Tingkat Sedang', 'Tingkat Parah'] 

# 4. Informasi Performa (Sesuai Metrik Anda)
st.sidebar.header("📊 Hasil Evaluasi Model")
st.sidebar.info("""
**Metrik Performa Utama:**
- **Loss:** Cross Entropy Loss (0.2826)
- **Akurasi:** Macro F1-Score (0.9130)
""")

# 5. Fitur Deteksi
uploaded_file = st.file_uploader("Unggah foto daun karet...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Daun Karet yang Diuji', use_container_width=True)
    
    # Preprocessing
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    with st.spinner('Menganalisis tingkat keparahan...'):
        preds = model.predict(img_array)
        result_idx = np.argmax(preds[0])
        confidence = np.max(preds[0]) * 100

    # Tampilan Hasil
    st.subheader(f"Hasil Prediksi: {class_names[result_idx]}")
    st.write(f"**Confidence Score:** {confidence:.2f}%")
    
    if result_idx > 0:
        st.warning("Rekomendasi: Lakukan tindakan pengendalian sesuai tingkat keparahan.")
