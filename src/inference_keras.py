# src/inference_keras.py

import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- 1. Configuration (CRITICAL: Must match your training/data structure) ---
# FIX: The path is set relative to the project root for reliable Streamlit loading.
MODEL_PATH = 'models/plant_disease_model.h5' 
IMG_HEIGHT, IMG_WIDTH = 128, 128 # The size your model was trained on

# Get class names from your processed data
BASE_DIR = "processed_data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")

# Retrieve actual class names from the data folders (used for labeling predictions)
try:
    # Get class names sorted alphabetically as Keras uses this order
    CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
except FileNotFoundError:
    # Fallback if the path still doesn't resolve 
    CLASS_NAMES = ["Bacterial_Spot", "Early_blight", "Late_Blight", "Leaf_Mold", 
                   "Mosaic", "Septoria_Leaf_Spot", "Spidermite_TW", 
                   "Target_Spot", "Yellow_Leaf", "healthy"] 


# --- 2. Load Model (Load only once using Streamlit's cache) ---
def load_keras_model(path=MODEL_PATH):
    """Initializes and loads the Keras model."""
    try:
        model = load_model(path)
        return model
    except Exception as e:
        print(f"🚨 Error loading model at {path}: {e}")
        # Note: If this fails during Streamlit run, the app will show the error message.
        return None

# Load the model outside the prediction function for quick access
MODEL = load_keras_model()

# --- 3. Prediction Function ---
def predict_image(image_path_or_object):
    """
    Loads, preprocesses, and predicts the class of a leaf image.
    Accepts a file path (str) or a PIL Image object (from Streamlit).
    """
    if MODEL is None:
        return "Model_Error", 0.0
    
    # Handle both PIL Image objects (from Streamlit) and file paths (for testing)
    if isinstance(image_path_or_object, str):
        try:
            img = Image.open(image_path_or_object).convert('RGB')
        except FileNotFoundError:
            return "File_Not_Found", 0.0
    else:
        # Image is already a PIL object from Streamlit upload
        img = image_path_or_object.convert('RGB')

    # Resize image to match training input shape 
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    
    # Convert image to array and normalize (0-1)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) # Add batch dimension
    x /= 255.0  # Normalize as done in training DataGenerator
    
    # Run prediction
    prediction_raw = MODEL.predict(x, verbose=0)
    
    # Process results
    predicted_class_index = np.argmax(prediction_raw[0])
    confidence = prediction_raw[0][predicted_class_index]
    
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    return predicted_class_name, confidence

# --- 4. Standalone Test Block ---
if __name__ == "__main__":
    print("\n--- Testing Inference Script ---")
    
    # This path must be updated to an actual image file in your project
    example_path = 'data/raw/healthy/0a0d6a11-ddd6-4dac-8469-d5f65af5afca___RS_HL 0555.JPG' 
    
    if MODEL:
        if os.path.exists(example_path):
            predicted_name, confidence = predict_image(example_path)
            print(f"Input image: {example_path}")
            print(f"Prediction: {predicted_name.replace('_', ' ')}")
            print(f"Confidence: {confidence:.2f}")
        else:
            print(f"⚠️ Test failed: Example image not found at {example_path}. Please update path.")
    else:
        print("Model failed to load. Check 'models/plant_disease_model.h5' path.")