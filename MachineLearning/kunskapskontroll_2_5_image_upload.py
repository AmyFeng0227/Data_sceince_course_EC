import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
from joblib import load
from streamlit_drawable_canvas import st_canvas

def process_image_for_mnist(image):
   

    # Resize the image to 28x28 pixels
    resized_image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    # Invert the image colors to match MNIST (white on black background)
    inverted_image = cv2.bitwise_not(resized_image)
   
    # Reshape the image to fit the model input requirements
    flattened_image = inverted_image.flatten().reshape(1, -1)
    return flattened_image

trained_model = load("voting_clf.joblib")
scaler = load("standard_scaler.joblib")

st.title('Digit prediction')
uploaded_file = st.file_uploader("Upload a handwritten image for prediction", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert("L")
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Process the image
    image_np = np.array(image)
    flattened_image = process_image_for_mnist(image = image_np)
    flattened_image_scaled = scaler.transform(flattened_image)
   # Show the processed image (need to convert it back to a displayable format)
    display_image = flattened_image.reshape(28, 28)
    st.image(display_image, caption='Processed Image', use_column_width=True)


    #prediction
    prediction = trained_model.predict(flattened_image_scaled) 
    st.write(prediction)

