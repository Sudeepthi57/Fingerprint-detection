import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
MODEL_PATH = "best_fingerprint_model.keras"
model = load_model(r'C:\Users\sande\OneDrive\drive\OneDrive\Desktop\Fingerprint-detection-main (2)\Fingerprint-detection-main\best_fingerprint_model.keras')

def predict(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_resized = cv2.resize(img_gray, (96, 96))
    img_norm = img_resized.reshape(1, 96, 96, 1) / 255.0
    raw = model.predict(img_norm)[0][0]
    if raw > 0.5:
        return "ALTERED", raw * 100
    else:
        return "REAL", (1 - raw) * 100

st.title("🔍 Fingerprint Alteration Detection")

file = st.file_uploader("Upload fingerprint image", type=["png","jpg","jpeg", "bmp"])

if file:
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Fingerprint")

    label, conf = predict(img)

    st.subheader("Result")
    st.write("Type:", label)
    st.write("Confidence:", f"{conf:.2f}%")