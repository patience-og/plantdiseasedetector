# src/train_keras.py (TensorFlow/Keras Implementation)

import os
import numpy as np
# Ensure you have tensorflow installed: pip install tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# train_model.py (Keras Implementation)

# ... [imports and definitions] ...

# --- 1. Paths and Configuration (CRITICAL UPDATE) ---
# The script is run from the root, so these paths are correct:
BASE_DIR = "processed_data" 
train_dir = os.path.join(BASE_DIR, "train")
val_dir = os.path.join(BASE_DIR, "val")
# Save the model one level up in the 'models' folder:
model_save_path = "models/plant_disease_model.h5" 

# ... [rest of the script] ...
# Parameters
img_height, img_width = 128, 128
batch_size = 32
epochs = 20 # Increased epochs for better results

# --- 2. Data Generators ---
train_datagen = ImageDataGenerator(
    rescale=1./255, # Normalize pixel values
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255) # No augmentation for validation

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = val_datagen.flow_from_directory(
    val_dir, # Use the validation split for validation
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# --- 3. Build Model ---
num_classes = len(train_generator.class_indices)
print(f"Building model with {num_classes} output classes.")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Optional: Save only the best model based on validation accuracy
checkpoint = ModelCheckpoint(
    model_save_path, 
    monitor='val_accuracy', 
    save_best_only=True, 
    mode='max', 
    verbose=1
)

# --- 4. Train ---
print("\n--- Starting Training ---")
history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=validation_generator,
    callbacks=[checkpoint] # Use the callback to save the best model
)

print(f"\nTraining complete! Best Model saved to {model_save_path}")