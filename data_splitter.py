import os
import shutil
from sklearn.model_selection import train_test_split

# -----------------------------
# CONFIG
# -----------------------------
RAW_DATASET_DIR = "data/raw" 
OUTPUT_DIR = "processed_data"      # Folder that will be created
TEST_SIZE = 0.10                      # 10% test
VAL_SIZE = 0.10                       # 10% validation
SEED = 42

# -----------------------------
# CREATE OUTPUT FOLDERS
# -----------------------------
train_dir = os.path.join(OUTPUT_DIR, "train")
val_dir = os.path.join(OUTPUT_DIR, "val")
test_dir = os.path.join(OUTPUT_DIR, "test")

for folder in [train_dir, val_dir, test_dir]:
    os.makedirs(folder, exist_ok=True)

# -----------------------------
# PROCESS EACH CLASS FOLDER
# -----------------------------
for class_name in os.listdir(RAW_DATASET_DIR):
    class_path = os.path.join(RAW_DATASET_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Processing: {class_name}")

    images = os.listdir(class_path)

    # Split train + temp
    train_imgs, temp_imgs = train_test_split(
        images, test_size=TEST_SIZE + VAL_SIZE, random_state=SEED
    )

    # Split temp to val + test
    val_imgs, test_imgs = train_test_split(
        temp_imgs, test_size=0.5, random_state=SEED
    )

    # Create class folders inside train/val/test
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

    # Copy train images
    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_dir, class_name, img)
        )

    # Copy validation images
    for img in val_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(val_dir, class_name, img)
        )

    # Copy test images
    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_dir, class_name, img)
        )

print("\n✅ Preprocessing complete!")
print("Folders created:")
print(f"- {train_dir}")
print(f"- {val_dir}")
print(f"- {test_dir}")
