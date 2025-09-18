import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import os

# Page config
st.set_page_config(
    page_title="Kannada Digit Classifier",
    page_icon="🔢",
    layout="wide"
)

# Load model with caching
@st.cache_resource
def load_model_cached():
    try:
        # For deployment, use relative path or environment variable
        model_path = r"best_densenet121_kannada_digits_128x128.keras"  # Change this for deployment
        model = load_model(model_path, compile=False)
        return model, True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

KANNADA_DIGITS = ["೦", "೧", "೨", "೩", "೪", "೫", "೬", "೭", "೮", "೯"]

def predict_digit(img, model):
    try:
        img = img.convert("L")
        img = np.array(img)
        img_resized = cv2.resize(img, (128, 128))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2RGB)
        img_norm = img_rgb.astype("float32") / 255.0
        img_input = np.expand_dims(img_norm, axis=0)
        
        preds = model.predict(img_input, verbose=0)
        predicted_class = np.argmax(preds, axis=1)[0]
        confidence = float(np.max(preds))
        
        return predicted_class, confidence, Image.fromarray(img_rgb)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return None, None, None

# Main app
st.title("🔢 Kannada Digit Classifier")
st.markdown("Upload an image of a Kannada digit (೦-೯) to get predictions")

# Load model
model, model_loaded = load_model_cached()

if not model_loaded:
    st.error("⚠️ Model could not be loaded.")
    st.stop()

st.success("✅ Model loaded successfully!")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image file", 
    type=['png', 'jpg', 'jpeg', 'bmp']
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Image")
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner('🔄 Processing...'):
        predicted_class, confidence, processed_image = predict_digit(image, model)
    
    if predicted_class is not None:
        with col2:
            st.subheader("📥 Processed Image")
            st.image(processed_image, caption="Processed (128x128)", use_column_width=True)
        
        st.subheader("🎯 Results")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            st.metric("Predicted Digit", predicted_class)
        with col4:
            st.metric("Kannada Character", KANNADA_DIGITS[predicted_class])
        with col5:
            st.metric("Confidence", f"{confidence:.2%}")
        
        st.progress(confidence)

# Instructions
with st.expander("📋 Instructions"):
    st.markdown("""
    1. Upload a clear image of a Kannada digit
    2. Supported formats: PNG, JPG, JPEG, BMP
    3. Best results with dark digits on light backgrounds
    4. Supported digits: ೦, ೧, ೨, ೩, ೪, ೫, ೬, ೭, ೮, ೯
    """)