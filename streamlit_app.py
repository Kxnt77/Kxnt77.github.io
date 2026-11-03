import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

st.set_page_config(page_title="YOLO Fruit Detection", layout="wide")
st.title("🍎 YOLOv8 Fruit Detection & Segmentation")
st.write("อัปโหลดรูปภาพผลไม้เพื่อทดสอบการทำนายด้วยโมเดล YOLO (Rec vs Polygon)")

# โหลดโมเดล
@st.cache_resource
def load_models():
    rec_model = YOLO("weights/best_rec.pt")
    poly_model = YOLO("weights/best_poly.pt")
    return rec_model, poly_model

model_rec, model_poly = load_models()

# Sidebar ตัวเลือก
model_type = st.sidebar.radio("เลือกประเภทโมเดล", ["Polygon (Segmentation)", "Rectangle (Bounding Box)"])
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

uploaded_file = st.file_uploader("เลือกรูปภาพผลไม้...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("รูปภาพต้นฉบับ")
        st.image(image, use_container_width=True)
        
    with col2:
        st.subheader("ผลการทำนาย")
        model = model_poly if model_type == "Polygon (Segmentation)" else model_rec
        
        # ทำนายผล
        results = model.predict(source=image, conf=conf_threshold)
        res_plotted = results[0].plot() # คืนค่า BGR numpy array
        
        # แปลง BGR เป็น RGB สำหรับ Streamlit
        res_rgb = res_plotted[:, :, ::-1]
        st.image(res_rgb, use_container_width=True)