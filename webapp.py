from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2

app = FastAPI()

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image = await file.read()
    # Process image and perform prediction...
    return {"prediction": "output_image_path"}

import streamlit as st
import requests

st.title("Brain MRI Metastasis Segmentation")

uploaded_file = st.file_uploader("Choose a brain MRI image...", type="jpg")
if uploaded_file is not None:
    response = requests.post("http://127.0.0.1:8000/predict/", files={"file": uploaded_file})
    st.image(response.json()["prediction"], caption='Predicted Segmentation')
