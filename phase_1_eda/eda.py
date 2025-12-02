# src/eda.py

import os
import random
from PIL import Image, UnidentifiedImageError
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


DATA_DIR = "data/raw"
SAMPLE_IMAGES_PER_CLASS = 5
CSV_OUTPUT = "data/metadata.csv"


def get_categories(data_dir):
    """List top-level categories in dataset."""
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    return categories

def collect_image_paths(data_dir):
    """Collect all image paths and labels."""
    data = []
    for category in get_categories(data_dir):
        category_path = os.path.join(data_dir, category)
        for root, _, files in os.walk(category_path):
            for f in files:
                img_path = os.path.join(root, f)
                data.append({"path": img_path, "category": category})
    return data

def analyze_image_sizes(data_dir):
    """Analyze image dimensions safely (skip unreadable files)."""
    records = []
    data = collect_image_paths(data_dir)
    for item in data:
        img_path = item["path"]
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                records.append({
                    "path": img_path,
                    "category": item["category"],
                    "width": width,
                    "height": height,
                    "pixels": width * height,
                    "aspect_ratio": width / height if height > 0 else 0
                })
        except (UnidentifiedImageError, OSError):
            print(f"Skipped unreadable image: {img_path}")
    df = pd.DataFrame(records)
    return df

def plot_sample_images(df, per_class=SAMPLE_IMAGES_PER_CLASS):
    """Display a few sample images per category."""
    categories = df["category"].unique()
    for cat in categories:
        sample = df[df["category"] == cat].sample(min(per_class, len(df[df["category"] == cat])))
        plt.figure(figsize=(15, 3))
        plt.suptitle(f"Sample Images - {cat}", fontsize=16)
        for i, row in enumerate(sample.itertuples()):
            try:
                with Image.open(row.path) as img:
                    plt.subplot(1, per_class, i+1)
                    plt.imshow(img)
                    plt.axis("off")
            except (UnidentifiedImageError, OSError):
                continue
        plt.show()

def plot_distributions(df):
    """Plot image size distributions."""
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    sns.histplot(df['width'], bins=30, kde=True)
    plt.title("Width Distribution")

    plt.subplot(1,3,2)
    sns.histplot(df['height'], bins=30, kde=True)
    plt.title("Height Distribution")

    plt.subplot(1,3,3)
    sns.histplot(df['pixels'], bins=30, kde=True)
    plt.title("Pixels Distribution")

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,4))
    sns.histplot(df['aspect_ratio'], bins=30, kde=True)
    plt.title("Aspect Ratio Distribution")
    plt.show()

def class_summary(df):
    """Compute summary statistics per category."""
    summary = df.groupby("category").agg({
        "width": ["count", "mean", "min", "max"],
        "height": ["mean", "min", "max"],
        "pixels": ["mean", "min", "max"],
        "aspect_ratio": ["mean", "min", "max"]
    }).reset_index()
    summary.columns = ["category", "count", "mean_width", "min_width", "max_width",
                       "mean_height", "min_height", "max_height",
                       "mean_pixels", "min_pixels", "max_pixels",
                       "mean_aspect_ratio", "min_aspect_ratio", "max_aspect_ratio"]
    print("\n=== Class-wise Summary Statistics ===")
    print(summary)
    return summary

def save_metadata(df, csv_path=CSV_OUTPUT):
    """Save image metadata to CSV for next step."""
    df.to_csv(csv_path, index=False)
    print(f"\nMetadata saved to: {csv_path}")


def run_eda():
    print("=== Dataset Categories ===")
    categories = get_categories(DATA_DIR)
    for cat in categories:
        print(f"- {cat}")

    print("\n=== Analyzing Image Sizes ===")
    df = analyze_image_sizes(DATA_DIR)
    print(f"Total readable images: {len(df)}")
    print(df.head())

    print("\n=== Sample Images ===")
    plot_sample_images(df)

    print("\n=== Image Size Distributions ===")
    plot_distributions(df)

    print("\n=== Class-wise Summary Statistics ===")
    class_summary(df)

    print("\n=== Saving Metadata CSV ===")
    save_metadata(df)


if __name__ == "__main__":
    run_eda()
