import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(
    page_title="Deteksi Mata Uang ASEAN",
    page_icon="💰",
    layout="centered"
)

st.title("💰 Deteksi Mata Uang ASEAN")
st.markdown("Deteksi uang kertas dari negara-negara Asia Tenggara")

# Load model
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    st.success("✅ Model siap digunakan!")
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# Pilihan input
option = st.radio("Pilih sumber gambar:", ["📁 Upload Gambar", "📷 Kamera"])

image = None

if option == "📁 Upload Gambar":
    uploaded = st.file_uploader("Pilih gambar uang...", type=['jpg', 'jpeg', 'png'])
    if uploaded:
        image = Image.open(uploaded)

else:  # Kamera
    image = st.camera_input("Ambil foto uang")

# Tombol deteksi
if st.button("🔍 DETEKSI SEKARANG", use_container_width=True) and image is not None:
    # Konversi ke numpy array
    img_array = np.array(image)
    
    with st.spinner("AI sedang menganalisis..."):
        # Jalankan deteksi
        results = model(img_array, conf=0.25)
        
        # Ambil hasil
        boxes = results[0].boxes
        
        if boxes is not None and len(boxes) > 0:
            st.success(f"✅ Terdeteksi {len(boxes)} mata uang!")
            
            # Map nama kelas ke tampilan yang rapi
            currency_names = {
                'malaysia': '🇲🇾 Ringgit Malaysia',
                'indonesia': '🇮🇩 Rupiah Indonesia',
                'thailand': '🇹🇭 Baht Thailand',
                'vietnam': '🇻🇳 Dong Vietnam',
                'philippines': '🇵🇭 Peso Filipina',
                'laos': '🇱🇦 Kip Laos',
                'myanmar': '🇲🇲 Kyat Myanmar',
                'cambodia': '🇰🇭 Riel Kamboja',
                'singapore': '🇸🇬 Dollar Singapore',
                'brunei': '🇧🇳 Dollar Brunei'
            }
            
            # Tampilkan daftar deteksi
            st.subheader("📋 Hasil Deteksi:")
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = model.names[class_id].lower()
                confidence = float(box.conf[0]) * 100
                
                display_name = currency_names.get(class_name, class_name.title())
                st.markdown(f"- {display_name}: **{confidence:.1f}%**")
            
            # Tampilkan gambar dengan bounding box
            result_img = results[0].plot()
            st.image(result_img, caption="Hasil Deteksi", use_container_width=True)
            
        else:
            st.warning("⚠️ Tidak ada mata uang yang terdeteksi. Coba dengan:")
            st.markdown("""
            - Pencahayaan yang lebih baik
            - Uang dalam keadaan rata (tidak terlipat)
            - Latar belakang polos
            """)

elif st.button("🔍 DETEKSI SEKARANG", use_container_width=True):
    st.error("❌ Upload atau ambil foto dulu!")

# Footer
st.markdown("---")
st.caption("Model YOLOv8 | Training dengan dataset Roboflow | Deteksi Mata Uang ASEAN")