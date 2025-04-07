import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import cv2
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import numpy as np
import random

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Ghost Detector", layout="wide", page_icon="👻")

# --- DAFTAR HANTU PALSU ---
ghost_classes = ['Pocong', 'Kuntilanak', 'Tuyul', 'Wewe Gombel', 'Genderuwo', 'Setan Gepeng']

# --- FUNGSI DETEKSI HANTU PALSU ---
def fake_ghost_detection(image):
    detections = []
    h, w = image.shape[:2]
    jumlah = random.choices([0, 1, 2], weights=[2, 5, 3])[0]

    for _ in range(jumlah):
        label = random.choice(ghost_classes)

        if label == "Tuyul":
            width = height = random.randint(50, 80)
            x1 = random.randint(0, w - width)
            y1 = random.randint(int(h * 0.75), h - height)
        elif label == "Kuntilanak":
            width = height = random.randint(100, 180)
            x1 = random.randint(0, w - width)
            y1 = random.randint(0, int(h * 0.3))
        elif label == "Genderuwo":
            width = height = random.randint(150, 250)
            x1 = random.randint(0, w - width)
            y1 = random.randint(int(h * 0.5), h - height)
        elif label == "Wewe Gombel":
            width = height = random.randint(180, 250)
            x1 = random.randint(0, w - width)
            y1 = random.randint(0, int(h * 0.4))
        elif label == "Setan Gepeng":
            width = random.randint(40, 60)
            height = random.randint(150, 220)
            x1 = random.randint(0, w - width)
            y1 = random.randint(0, h - height)
        else:  # Pocong
            width = height = random.randint(80, 120)
            x1 = random.randint(0, w - width)
            y1 = random.randint(0, h - height)

        x2 = x1 + width
        y2 = y1 + height
        detections.append((label, (x1, y1, x2, y2)))

    return detections

def draw_fake_detections(image, detections):
    for label, (x1, y1, x2, y2) in detections:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(image, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return image

# --- LOAD MODEL YOLOv8 ---
@st.cache_resource
def load_model():
    try:
        return YOLO('best.pt')  # Ganti path sesuai model kamu
    except Exception as e:
        st.error(f"❌ Gagal memuat model YOLOv8: {e}")
        return None

model = load_model()

# --- JUDUL HALAMAN ---
st.markdown("<h1 style='color:#ff0000;'>Ghost Detector 👻</h1>", unsafe_allow_html=True)
st.markdown("Upload gambar, video, atau gunakan kamera untuk mendeteksi objek asli dan hantu palsu.")
st.markdown("---")

# --- SESSION STATE ---
if 'deteksi_selesai' not in st.session_state:
    st.session_state['deteksi_selesai'] = False

# --- PILIH JENIS FILE ---
file_type = st.radio("Pilih jenis file yang ingin digunakan:", ["Gambar", "Video", "Kamera"])

# --- GAMBAR ---
if file_type == "Gambar":
    uploaded_file = st.file_uploader("📷 Upload gambar (jpg/jpeg/png):", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        image_np = np.array(image)
        st.image(image, caption="📷 Gambar yang diupload", use_container_width=True)

        with st.spinner("🔍 Mendeteksi..."):
            result_image = image_np.copy()
            if model:
                results = model.predict(result_image)
                result_image = results[0].plot()
            fake_detections = fake_ghost_detection(result_image)
            result_image = draw_fake_detections(result_image, fake_detections)
            result_pil = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))

        st.image(result_pil, caption="🎯 Hasil Deteksi", use_container_width=True)
        st.session_state['deteksi_selesai'] = True

# --- VIDEO ---
elif file_type == "Video":
    video_file = st.file_uploader("🎥 Upload video (mp4/mov/avi):", type=["mp4", "mov", "avi"])
    if video_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        with st.spinner("🔍 Mendeteksi pada video..."):
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated_frame = frame.copy()
                if model:
                    results = model.predict(annotated_frame)
                    annotated_frame = results[0].plot()
                fake_detections = fake_ghost_detection(annotated_frame)
                annotated_frame = draw_fake_detections(annotated_frame, fake_detections)
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
            cap.release()
            st.session_state['deteksi_selesai'] = True

# --- KAMERA (LIVE STREAM) ---
elif file_type == "Kamera":
    class GhostDetector(VideoProcessorBase):
        def __init__(self):
            self.model = model

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            result_img = img.copy()

            if self.model:
                results = self.model.predict(result_img)
                result_img = results[0].plot()

            fake_detections = fake_ghost_detection(result_img)
            result_img = draw_fake_detections(result_img, fake_detections)

            return av.VideoFrame.from_ndarray(result_img, format="bgr24")

    st.markdown("## 🎥 Live Ghost Detection")
    webrtc_streamer(
        key="ghost-live",
        video_processor_factory=GhostDetector,
        media_stream_constraints={"video": True, "audio": False}
    )
