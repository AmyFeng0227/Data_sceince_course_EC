import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from joblib import load
from streamlit_drawable_canvas import st_canvas

trained_model = load("voting_clf.joblib")
scaler = load("standard_scaler.joblib")

st.title("Handwritten digits detection")
st.write("Please write a digit below")

#creating a drawable canvas
canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 1)",  # Filler color or painting color
    stroke_width=9,
    stroke_color="#ffffff",
    background_color="#000000",
    #background_image=None,
    update_streamlit=True,
    height=150,
    width=150,
    drawing_mode="freedraw",
    key="canvas",
)

# Process the drawing
if canvas_result.image_data is not None and np.sum(canvas_result.image_data) > 0:
    # Convert the canvas image
    img_data = canvas_result.image_data.astype("uint8")
    img_gray = cv2.cvtColor(img_data, cv2.COLOR_BGRA2GRAY)
    resized_image = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
    #inverted_image = cv2.bitwise_not(resized_image)
    flattened_image = resized_image.flatten().reshape(1, -1)
    flattened_image_scaled = scaler.transform(flattened_image)
    
    # Make a prediction
    prediction = trained_model.predict(flattened_image_scaled)
    st.write("the value is", prediction)