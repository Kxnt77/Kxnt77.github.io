import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
from pathlib import Path

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="YOLO Fruit Detection", layout="wide")

# กำหนด Path ของโปรเจกต์
BASE_DIR = Path(__file__).resolve().parent

# โหลดโมเดลพร้อมทำ Cache
@st.cache_resource
def load_models():
    rec_model = YOLO(str(BASE_DIR / "weights" / "best_rec.pt"))
    poly_model = YOLO(str(BASE_DIR / "weights" / "best_poly.pt"))
    return rec_model, poly_model

model_rec, model_poly = load_models()

# Header หน้าเว็บ
st.title("🍎 YOLOv8 Fruit Detection & Segmentation")
st.write("อัปโหลดรูปภาพผลไม้เพื่อทดสอบการทำนายด้วยโมเดล YOLO (Rec vs Polygon)")

# --- ส่วนแสดงรายชื่อคลาสผลทำนายทั้งหมดจากโมเดล ---
class_names = list(model_poly.names.values())
st.markdown("**🏷️ คลาสทั้งหมดที่โมเดลสามารถทำนายได้ (Supported Classes):**")

# แสดงผลคลาสเป็น Badge สีสวยงาม
class_badges = " ".join([f"`{name}`" for name in class_names])
st.markdown(class_badges)

# --- ส่วนตั้งค่าโมเดล ---
st.divider()
col_set1, col_set2 = st.columns(2)

with col_set1:
    model_type = st.radio(
        "เลือกประเภทโมเดล", 
        ["Polygon (Segmentation)", "Rectangle (Bounding Box)"],
        horizontal=True
    )

with col_set2:
    conf_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

st.divider()

# ปุ่มอัปโหลดรูปภาพ
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

# --- ส่วนแสดงกราฟ/ผลการเทรนโมเดล ---
st.write("")
st.write("")
st.subheader("📊 ผลการเทรนโมเดล (Training Results)")

col_train1, col_train2 = st.columns(2)

with col_train1:
    st.markdown("**1. ผลการเทรน Rectangle (Bounding Box)**")
    train_rec_path = BASE_DIR / "images" / "rec_results.png"
    if train_rec_path.exists():
        st.image(str(train_rec_path), use_container_width=True)
    else:
        st.info("นำรูปกราฟผลการเทรน Rec มาใส่ที่ images/rec_results.png")

with col_train2:
    st.markdown("**2. ผลการเทรน Polygon (Segmentation)**")
    train_poly_path = BASE_DIR / "images" / "poly_results.png"
    if train_poly_path.exists():
        st.image(str(train_poly_path), use_container_width=True)
    else:
        st.info("นำรูปกราฟผลการเทรน Polygon มาใส่ที่ images/poly_results.png")

# --- Footer เครดิตผู้จัดทำ ---
st.divider()
st.caption("created by 6806021411236 Kanthida Hasut ITI sec B")
