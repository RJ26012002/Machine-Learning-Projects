import streamlit as st 
from keras.models import load_model
import numpy as np

model = load_model("model.h5")
labels = np.load("labels.npy")

st.title("Welcome to the Prediction App")

a  = st.number_input("Label 1")
b  = st.number_input("Label 2")
c  = st.number_input("Label 3")
d  = st.number_input("Label 4")
btn = st.button("Predict")

if btn:
    pred=model.predict(np.array([a,b,c,d]).reshape(1,-1))
    print(pred)
    print(np.argmax(pred))
    pred= labels[np.argmax(pred)]
    print(pred)

    st.subheader(pred)