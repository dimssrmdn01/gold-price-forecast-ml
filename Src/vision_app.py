import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Candlestick Vision Scanner", layout="centered")

st.title("👁️ Computer Vision Candlestick Scanner")
st.write("Upload screenshot chart untuk mendeteksi pola geometris pasar.")

@st.cache_resource
def load_vision_model():
    return tf.keras.models.load_model('Models/candlestick_cv_model.h5')

try:
    model = load_vision_model()
    class_names = ['Double_Bottom', 'Double_Top', 'Head_and_Shoulders', 'Rounding_Bottom', 'V_Shape_Recovery']
    
    uploaded_file = st.file_uploader("Pilih gambar chart...", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file).convert('RGB')
        st.image(img, caption="Chart yang di-upload", use_container_width=True)
        
        # Preprocessing gambar agar sesuai standar model
        img_resized = img.resize((150, 150))
        img_array = np.array(img_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
        
        with st.spinner("Memindai struktur matriks piksel..."):
            predictions = model.predict(img_preprocessed)
            score = tf.nn.softmax(predictions[0])
            pred_class = class_names[np.argmax(score)]
            confidence = 100 * np.max(score)
            
        st.success(f"**Hasil Analisis:** Terdeteksi pola **{pred_class}** (Akurasi Spekulatif: {confidence:.2f}%)")
        st.warning("⚠️ Catatan: Model saat ini dalam mode purwarupa (dataset terbatas).")

except Exception as e:
    st.error(f"Gagal memuat mesin vision: {e}")