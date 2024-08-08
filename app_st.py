from fastcore.all import *
from fastai.vision.all import *
import streamlit as st

## LOAD MODEl
learn_inf = load_learner("aedesModel_2.pkl")

## CLASSIFIER
def classify_img(data):
    pred, pred_idx, probs = learn_inf.predict(data)
    return pred, probs[pred_idx]

## STREAMLIT
st.title("Aedes Classifier! 🦟")
bytes_data = None
uploaded_image = st.file_uploader("Choose your Mosquito:")
if uploaded_image:
    bytes_data = uploaded_image.getvalue()
    st.image(bytes_data, caption="Uploaded image")   
if bytes_data:
    classify = st.button("CLASSIFY!")
    if classify:
        label, confidence = classify_img(bytes_data)
        st.write(f"It is a {label}! ({confidence:.04f})")