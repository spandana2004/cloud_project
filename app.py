import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# Load YOLOv8 model (replace with your actual model path)
MODEL_PATH = 'yolov8n.pt'  # Make sure this model is in the same directory
model = YOLO(MODEL_PATH)

# App title
st.set_page_config(page_title="Garbage Detection App", layout="centered")
st.title("🗑️ Garbage Detection and Classification")
st.write("Upload a trash image and get counts of detected garbage categories like plastic bottles, batteries, etc.")

# File uploader
uploaded_file = st.file_uploader("📷 Upload an image", type=["jpg", "jpeg", "png"])

# Process image
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to NumPy array
    image_np = np.array(image)

    # Run inference
    with st.spinner("Detecting objects..."):
        results = model(image_np)[0]

        # Count objects
        detections = results.boxes.cls.cpu().numpy()
        class_names = model.names

        counts = {}
        for cls in detections:
            label = class_names[int(cls)]
            counts[label] = counts.get(label, 0) + 1

        # Draw boxes
        annotated_image = results.plot()

    # Display annotated image
    st.image(annotated_image, caption="🧾 Detection Result", use_column_width=True)

    # Display counts
    st.subheader("📊 Detected Objects Count")
    if counts:
        for label, count in counts.items():
            st.markdown(f"- **{label}**: {count}")
    else:
        st.warning("No garbage objects detected.")
