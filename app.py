import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

model = tf.keras.models.load_model('kidney_tumor_classifier.h5')

# Set the page configuration (title and layout)
st.set_page_config(page_title="Kidney Tumor Detection", layout="centered")

# Custom CSS Styling
st.markdown("""
    <style>
        body {
            background-color: #f5f5f5;
        }
        .header {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #4B8C39;
            margin-bottom: 20px;
        }
        .subheader {
            text-align: center;
            font-size: 20px;
            color: #7a7a7a;
            margin-bottom: 20px;
        }
        .result {
            text-align: center;
            font-size: 24px;
            font-weight: bold;
        }
        .malignant {
            color: #ff4c4c;
        }
        .benign {
            color: #28a745;
        }
        .confidence {
            font-size: 18px;
            font-weight: normal;
            color: #555;
        }
        .button {
            display: block;
            width: 250px;
            margin: auto;
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 18px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        .button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# Page Title and Icon
st.markdown('<div class="header">🩺Chronic Kidney Disease Detection</div>', unsafe_allow_html=True)

# Description of the app with styling
st.markdown('<div class="subheader">Upload an image of a kidney tumor and let the model predict whether it is malignant or benign.</div>', unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Upload a Kidney Disease Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction (resize, normalize, etc.)
    img = img.resize((64, 64))  # Resize to 64x64 to match the model's expected input size
    img_array = np.array(img) / 255.0  # Normalize the image

    # Ensure the image has the correct shape (add batch dimension)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Make the prediction with progress spinner
    with st.spinner('Making the prediction...'):
        prediction = model.predict(img_array)
        
        # Get the confidence score (model output is a probability)
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]

        # Display confidence with a percentage format
        confidence_score = f"{confidence * 100:.2f}%"

        # Display the result with confidence
        if prediction[0] > 0.5:
            st.markdown(f'<div class="result malignant">The tumor is **Malignant**!</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">Confidence: {confidence_score}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result benign">The tumor is **Benign**!</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="confidence">Confidence: {confidence_score}</div>', unsafe_allow_html=True)

    # Attractive Button for the prediction
    st.markdown('<button class="button">Re-Upload Image</button>', unsafe_allow_html=True)

# Footer Text (helpful information or instructions)
st.markdown("""
    <div style="text-align: center; font-size: 14px; color: #999;">
        Created with ❤️ by Kidney Disease Detection Team | <a href="https://www.kidneykindness.com">Visit our site</a>
    </div>
""", unsafe_allow_html=True)
