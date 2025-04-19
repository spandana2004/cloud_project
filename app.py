import streamlit as st
import numpy as np
from PIL import Image
import os
import gdown
from ultralytics import YOLO

# -------------------------
# Download model from Google Drive
# -------------------------

MODEL_URL = "https://drive.google.com/uc?id=1Y_uW_GrpJthpJwHcW_0nk8eszy-a_lBN"
MODEL_PATH = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    return YOLO(MODEL_PATH)

model = load_model()

# -------------------------
# Streamlit App UI
# -------------------------

st.set_page_config(page_title="Garbage Detection App", layout="centered")
st.title("🗑️ Garbage Detection and Classification")
st.write("Upload a trash image to detect and count garbage categories like plastic, batteries, cans, etc.")

uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)

    with st.spinner("Running YOLOv8 inference..."):
        results = model(image_np)[0]
        detections = results.boxes.cls.cpu().numpy()
        class_names = model.names

        # Count detected objects
        counts = {}
        for cls in detections:
            label = class_names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

        # Annotated image
        annotated = results.plot()

    st.image(annotated, caption="🧾 Detection Result", use_column_width=True)

    st.subheader("📊 Detected Object Counts")
    if counts:
        for label, count in counts.items():
            st.markdown(f"- **{label}**: {count}")
    else:
        st.warning("No garbage items detected.")
