import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import pandas as pd
import os, zipfile, io, time, h5py, keras
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json

st.set_page_config(page_title="Awounang-Donfack-Kabawa", layout="wide", initial_sidebar_state="collapsed")


@st.cache_resource
def load_model():
    base = tf.keras.applications.VGG16(weights=None, include_top=False, input_shape=(224, 224, 3))
    m = keras.Sequential([base, keras.layers.Flatten(), keras.layers.Dense(500, activation='relu'), keras.layers.Dropout(0.5), keras.layers.Dense(9, activation='softmax')])
    with h5py.File("model.h5", "r") as f:
        for layer in m.layers:
            if layer.name in f:
                w = [np.array(f[layer.name][k]) for k in f[layer.name].keys()]
                if w: layer.set_weights(w)
    return m

nom_classe = ["actinic keratosis", "basal cell carcinoma", "dermatofibroma",
              "melanoma", "nevus", "pigmented benign keratosis", "seborrheic keratosis",
              "squamous cell carcinoma", "vascular lesion"]
classes_malignes = [0, 1, 3, 7]
classes_benignes = [2, 4, 5, 6, 8]

with st.spinner('Chargement du modèle...'):
    model = load_model()

st.title("Diagnostic du cancer de la peau")
st.write("Chargez une image dermatoscopique pour obtenir un diagnostic.")
file = st.file_uploader("Chargez une image à classifier", type=["jpg", "png", "jpeg"])

def upload_predict(image, model):
    image = image.convert("RGB")  # ✅ force 3 canaux
    image = np.asarray(image).astype('float32')
    image = cv2.resize(image, (224, 224))
    image /= 255
    image = image.reshape([-1, 224, 224, 3])
    prediction = model.predict(image)
    return prediction

if file is None:
    st.info("Veuillez charger une image de lésion cutanée.")
else:
    image = Image.open(file)

    # ✅ Image réduite et centrée
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, width=300)

    prediction = upload_predict(image, model)
    proba = prediction[0]

    confiance_malin = float(np.sum(proba[classes_malignes]))
    confiance_benin = float(np.sum(proba[classes_benignes]))

    st.subheader("🔍 Diagnostic")
    if confiance_malin < confiance_benin:
        st.error(f"⚠️ CANCER DÉTECTÉ avec {confiance_malin*100:.1f}% de confiance")
        st.subheader("🧬 Type de cancer détecté")
        probas_malignes = {nom_classe[i]: proba[i] for i in classes_malignes}
        total = sum(probas_malignes.values())
        probas_triees = sorted(probas_malignes.items(), key=lambda x: x[1], reverse=True)
        for classe, p in probas_triees:
            st.write(f"→ **{classe}** : {p/total*100:.1f}%")
    else:
        st.success(f"✅ LÉSION BÉNIGNE avec {confiance_benin*100:.1f}% de confiance")
        st.subheader("📋 Type de lésion")
        probas_benignes = {nom_classe[i]: proba[i] for i in classes_benignes}
        total = sum(probas_benignes.values())
        probas_triees = sorted(probas_benignes.items(), key=lambda x: x[1], reverse=True)
        for classe, p in probas_triees:
            st.write(f"→ **{classe}** : {p/total*100:.1f}%")
