# app/streamlit_app.py

import streamlit as st
import numpy as np
from PIL import Image
import os
import sys # <-- NEW IMPORT

# --- Critical Path Fix ---
# Add the project root directory to the Python path so modules in 'src' can be found.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# -------------------------

# Import the necessary functions and variables from your finalized inference script
try:
    # Now, this import should work because the system path is updated:
    from src.inference_keras import predict_image, load_keras_model, MODEL_PATH, CLASS_NAMES
    from src.inference_keras import IMG_HEIGHT, IMG_WIDTH
except ImportError:
    st.error("Fatal Error: Could not import prediction logic. Check sys.path setup.")
    st.stop()


# --- 1. Streamlit UI and Configuration ---

st.set_page_config(
    page_title="AI Plant Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.title("🌾 AI-Powered Crop Disease Detection System")
st.markdown("Upload a photo of a plant leaf to get an instant disease diagnosis using our trained CNN model.")
st.markdown("---")


# --- 2. Load Model (Using Streamlit Caching) ---

# Load the Keras model using Streamlit's cache decorator for efficiency
# This ensures the model loads only once when the app starts.
@st.cache_resource
def get_model():
    return load_keras_model(MODEL_PATH)

model = get_model()

if model is None:
    st.error("Model Loading Failed. Please check the path and ensure training was complete.")
    st.markdown(f"**Attempted Path:** `{MODEL_PATH}`")
    st.stop()


# --- 3. File Uploader and Prediction Logic ---

st.header("Upload Image")
uploaded_file = st.file_uploader(
    "Choose a Leaf Image (JPG or PNG):", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image and prediction side-by-side
    col1, col2 = st.columns([1, 1])
    
    # Process the image
    image = Image.open(uploaded_file)
    
    with col1:
        st.subheader("Uploaded Leaf")
        st.image(image, caption='Image for Analysis', use_column_width=True)
        
    # --- Run Prediction ---
    with st.spinner('Analyzing image and predicting...'):
        predicted_class, confidence = predict_image(image)

    with col2:
        st.subheader("🔬 Diagnosis Result")
        
        # Display final result
        if confidence > 0.0:
            display_name = predicted_class.replace('_', ' ')
            
            # --- Prediction Metrics ---
            st.success(f"**Top Diagnosis:** {display_name}")
            st.metric(label="Confidence Score", value=f"{confidence*100:.2f}%")
            
            st.markdown("---")
            
            # --- Disease Advice Panel ---
            if 'Healthy' in predicted_class or 'healthy' in predicted_class:
                st.balloons()
                st.info("🌱 **Assessment:** The leaf appears healthy! Continue regular monitoring and care.")
            else:
                st.warning(f"🚨 **Action Required:** High confidence of **{display_name}** detected.")
                st.markdown("""
                **Recommended Next Steps:**
                1. **Isolate** the affected plant immediately.
                2. **Consult** a local agricultural extension expert for targeted treatment.
                3. **Monitor** environmental factors (humidity, drainage).
                """)
        
        else:
            st.warning("Could not determine a clear prediction. Please upload a clearer leaf image.")


# --- 4. Sidebar/Footer Information ---

st.sidebar.header("Project Overview")
st.sidebar.markdown(f"""
This application uses a Convolutional Neural Network (CNN) 
trained on **{len(CLASS_NAMES)}** distinct plant disease classes.

**Model Details:**
* **Architecture:** Custom Keras CNN
* **Input Size:** {IMG_HEIGHT}x{IMG_WIDTH} pixels
* **Data Source:** Cleaned PlantVillage Dataset
* **Model File:** `{MODEL_PATH}`
""")

st.sidebar.markdown("---")
st.sidebar.success("Project Cycle Complete: Data Preprocessing -> Training -> Deployment.")