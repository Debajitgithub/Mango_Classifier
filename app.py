import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# Load models once
@st.cache_resource
def load_models():
    ripeness_model = load_model("ripeness_model.h5")
    species_model = load_model("species_model.h5")
    return ripeness_model, species_model

# Class labels (UPDATE these as per training)
ripeness_classes = ['Partially Ripe', 'Ripe', 'Rotten', 'Unripe']
species_classes = ['Amrapali', 'Fazlee', 'Harivanga', 'Mallika', 'Nilambari']  # 🛠️ Modify if needed

IMG_SIZE = 380  # EfficientNetB4

def preprocess_image(img):
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

st.title("🥭 Mango Classifier")
st.caption("EfficientNetB4-based mango ripeness and species detector")

uploaded_file = st.file_uploader("Upload a mango image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Mango Image", use_column_width=True)

    ripeness_model, species_model = load_models()
    img_processed = preprocess_image(img)

    # Predict
    ripeness_pred = ripeness_model.predict(img_processed)[0]
    species_pred = species_model.predict(img_processed)[0]

    ripeness_top2 = ripeness_pred.argsort()[-2:][::-1]
    species_top2 = species_pred.argsort()[-2:][::-1]

    st.markdown("### 🥭 Ripeness Prediction:")
    for i in ripeness_top2:
        st.write(f"**{ripeness_classes[i]}** — {ripeness_pred[i]*100:.2f}%")

    st.markdown("### 🌱 Species Prediction:")
    for i in species_top2:
        st.write(f"**{species_classes[i]}** — {species_pred[i]*100:.2f}%")
