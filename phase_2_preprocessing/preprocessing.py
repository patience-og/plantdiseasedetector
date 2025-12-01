import os
from PIL import Image

# Paths
RAW_DIR = '../data/raw'         # Path to raw images
PROCESSED_DIR = '../processed'  # Path to save processed images

# Create processed folder if it doesn't exist
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Image preprocessing function
def preprocess_image(img_path, output_dir, target_size=(224, 224)):
    try:
        with Image.open(img_path) as img:
            # Convert to RGB if not
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Resize image
            img = img.resize(target_size)
            # Prepare output path
            class_name = os.path.basename(os.path.dirname(img_path))
            class_dir = os.path.join(output_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            out_path = os.path.join(class_dir, os.path.basename(img_path))
            # Save processed image
            img.save(out_path)
    except Exception as e:
        print(f"Failed to process {img_path}: {e}")

# Loop through all images
def preprocess_dataset(raw_dir, processed_dir):
    for root, dirs, files in os.walk(raw_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file)
                preprocess_image(img_path, processed_dir)

if __name__ == "__main__":
    preprocess_dataset(RAW_DIR, PROCESSED_DIR)
    print("Preprocessing complete!")
