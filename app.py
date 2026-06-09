import streamlit as st
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import av
import cv2
import numpy as np

# ==================================================
# KONFIGURASI HALAMAN
# ==================================================
st.set_page_config(
    page_title="Real-time Deteksi Mata Uang ASEAN",
    page_icon="🎥",
    layout="centered"
)

st.title("🎥 REAL-TIME DETEKSI MATA UANG ASEAN")
st.markdown("**Arahkan kamera ke uang kertas, deteksi LANGSUNG muncul!**")

# ==================================================
# LOAD MODEL
# ==================================================
@st.cache_resource
def load_model():
    return YOLO('best.pt')

try:
    model = load_model()
    st.success("✅ Model AI siap! Aktifkan kamera untuk mulai deteksi.")
except Exception as e:
    st.error(f"❌ Gagal load model: {e}")
    st.stop()

# ==================================================
# VIDEO PROCESSOR UNTUK REAL-TIME
# ==================================================
class VideoProcessor(VideoTransformerBase):
    def __init__(self):
        self.model = model
        self.confidence_threshold = st.session_state.get('confidence', 0.25)
    
    def recv(self, frame):
        # Konversi frame ke format OpenCV
        img = frame.to_ndarray(format="bgr24")
        
        # Jalankan deteksi YOLO di frame ini
        results = self.model(img, conf=self.confidence_threshold)
        
        # Gambar bounding box di frame
        annotated_frame = results[0].plot()
        
        # Balikin ke bentuk video
        return av.VideoFrame.from_ndarray(annotated_frame, format="bgr24")

# ==================================================
# SLIDER UNTUK MENGATUR SENSITIVITAS
# ==================================================
st.sidebar.header("⚙️ Pengaturan")

confidence = st.sidebar.slider(
    "🎯 Confidence Threshold (Sensitivity)",
    min_value=0.0,
    max_value=1.0,
    value=0.25,
    step=0.05,
    help="Semakin rendah nilainya, semakin sensitif (tapi bisa lebih banyak false positive)"
)

# Simpan ke session state
if 'confidence' not in st.session_state:
    st.session_state.confidence = confidence
else:
    st.session_state.confidence = confidence

# Tampilkan info di sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
### 📊 Info:
- **Current Threshold:** {confidence:.2f}
- **Mode:** {'Sensitif' if confidence < 0.3 else 'Normal' if confidence < 0.6 else 'Ketat'}
""")

# ==================================================
# TAMPILAN WEBCAM REAL-TIME
# ==================================================
st.markdown("---")

# Tombol start/stop webcam
webrtc_ctx = webrtc_streamer(
    key="deteksi-uang-realtime",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": {
            "width": {"ideal": 640},
            "height": {"ideal": 480},
            "facingMode": "environment"  # pakai kamera belakang di HP
        },
        "audio": False
    },
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    }
)

# ==================================================
# STATUS & PETUNJUK
# ==================================================
if webrtc_ctx.state.playing:
    st.success("🎥 **KAMERA AKTIF!** Arahkan ke uang kertas, bounding box akan muncul otomatis.")
    
    # Tampilkan daftar mata uang yang bisa dideteksi
    with st.expander("📋 Mata uang yang bisa dideteksi", expanded=False):
        st.markdown("""
        - 🇲🇾 Ringgit Malaysia
        - 🇮🇩 Rupiah Indonesia
        - 🇹🇭 Baht Thailand
        - 🇻🇳 Dong Vietnam
        - 🇵🇭 Peso Filipina
        - 🇱🇦 Kip Laos
        - 🇲🇲 Kyat Myanmar
        - 🇰🇭 Riel Kamboja
        - 🇸🇬 Dollar Singapore
        """)
else:
    st.info("⏸️ **Kamera belum aktif.** Klik 'START' di atas untuk memulai deteksi real-time.")
    st.markdown("""
    ### 📌 Cara Penggunaan:
    1. Klik tombol **START** (di atas, frame video)
    2. Izinkan akses kamera
    3. Arahkan kamera ke uang kertas
    4. **Bounding box hijau akan muncul langsung** di layar!
    5. Gunakan slider di samping kiri untuk mengatur sensitivitas
    """)

# ==================================================
# FOOTER
# ==================================================
st.markdown("---")
st.caption("🎯 Real-time detection dengan YOLOv8 | Streamlit WebRTC | Deteksi Mata Uang ASEAN")
