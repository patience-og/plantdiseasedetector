import os
import sys

# Define the root path to your raw data
DATA_ROOT = 'data/raw'

print("--- 🌿 Data Structure Diagnostic ---")
print(f"Checking Path: {os.path.abspath(DATA_ROOT)}")
print("-" * 35)

if not os.path.exists(DATA_ROOT):
    print(f"🚨 ERROR: The folder '{DATA_ROOT}' was not found.")
    sys.exit()

total_files = 0
found_classes = []

# Walk through the directories and files
for root, dirs, files in os.walk(DATA_ROOT):
    # Only process directories that are considered class folders (i.e., not the root or nested junk)
    if root != DATA_ROOT:
        class_name = os.path.basename(root)
        image_count = 0
        
        # Count only common image extensions
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_count += 1
        
        # A folder is considered a class if it has images
        if image_count > 0:
            found_classes.append((class_name, image_count, root))
            total_files += image_count
        elif not dirs and not image_count:
            # Report empty folders that aren't parents
            print(f"⚠️ Empty/Junk Folder Found: {root}")

# Print the organized results
print("\n--- ✅ Class Inventory (Images Found) ---")
if found_classes:
    for name, count, path in sorted(found_classes, key=lambda x: x[0]):
        # Report the name and the path so we can see if it's nested
        # We only care about the class folder name and the count
        print(f"  - **{name}**: {count} images (Path: {os.path.relpath(path, DATA_ROOT)})")
    
    print("-" * 35)
    print(f"**TOTAL IMAGES FOUND: {total_files}**")
else:
    print("🚨 NO IMAGE FILES FOUND IN ANY SUBFOLDERS. (Did you move your files?)")

print("\n--- 🛠️ Conclusion ---")
print("If the list above contains duplicate class names or nested paths, a manual correction is required.")