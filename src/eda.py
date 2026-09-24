"""
SentriVision AI - Data Analyst EDA Module

Performs exploratory data analysis on the processed CIFAR-10 dataset.

Data Analyst responsibilities:
- Dataset overview
- Class distribution analysis
- Image statistics
- RGB channel analysis
- Brightness analysis
- Contrast analysis
- Distribution and skewness analysis
"""

from pathlib import Path
import json
import pickle
import numpy as np
import pandas as pd


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = PROJECT_ROOT.parent / "cifar-10-data" / "cifar-10-batches-py"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# CIFAR-10 class names
CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck"
]


# ---------------------------------------------------------
# LOAD CIFAR-10 BATCH
# ---------------------------------------------------------

def load_cifar_batch(file_path):
    """
    Load a CIFAR-10 Python batch file.
    """

    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = np.array(batch[b"labels"])

    # Convert flattened images:
    # (N, 3072) -> (N, 32, 32, 3)
    images = images.reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)

    return images, labels


# ---------------------------------------------------------
# LOAD TRAINING DATA
# ---------------------------------------------------------

def load_training_data():
    """
    Load all five CIFAR-10 training batches.
    """

    images = []
    labels = []

    for i in range(1, 6):
        file_path = DATASET_DIR / f"data_batch_{i}"

        batch_images, batch_labels = load_cifar_batch(file_path)

        images.append(batch_images)
        labels.append(batch_labels)

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    return images, labels


# ---------------------------------------------------------
# DATASET OVERVIEW
# ---------------------------------------------------------

def dataset_overview(images, labels):
    """
    Generate basic dataset statistics.
    """

    overview = {
        "number_of_images": len(images),
        "image_height": images.shape[1],
        "image_width": images.shape[2],
        "channels": images.shape[3],
        "number_of_classes": len(np.unique(labels)),
        "pixel_minimum": int(images.min()),
        "pixel_maximum": int(images.max()),
        "pixel_mean": float(images.mean()),
        "pixel_std": float(images.std())
    }

    return overview


# ---------------------------------------------------------
# CLASS DISTRIBUTION
# ---------------------------------------------------------

def class_distribution(labels):
    """
    Calculate number and percentage of images in each class.
    """

    counts = np.bincount(labels, minlength=10)

    distribution = pd.DataFrame({
        "class_id": range(10),
        "class_name": CLASS_NAMES,
        "image_count": counts
    })

    distribution["percentage"] = (
        distribution["image_count"]
        / distribution["image_count"].sum()
        * 100
    )

    return distribution


# ---------------------------------------------------------
# IMAGE STATISTICS
# ---------------------------------------------------------

def calculate_image_statistics(images, labels):
    """
    Calculate image-level brightness and contrast statistics.
    """

    # Convert to float for statistical calculations
    images_float = images.astype(np.float32)

    brightness = images_float.mean(axis=(1, 2, 3))
    contrast = images_float.std(axis=(1, 2, 3))

    red_mean = images_float[:, :, :, 0].mean(axis=(1, 2))
    green_mean = images_float[:, :, :, 1].mean(axis=(1, 2))
    blue_mean = images_float[:, :, :, 2].mean(axis=(1, 2))

    statistics = pd.DataFrame({
        "label": labels,
        "class_name": [CLASS_NAMES[label] for label in labels],
        "brightness": brightness,
        "contrast": contrast,
        "red_mean": red_mean,
        "green_mean": green_mean,
        "blue_mean": blue_mean
    })

    return statistics


# ---------------------------------------------------------
# CLASS-WISE IMAGE STATISTICS
# ---------------------------------------------------------

def class_statistics(statistics):
    """
    Calculate average image statistics for every class.
    """

    result = (
        statistics
        .groupby("class_name")
        [
            [
                "brightness",
                "contrast",
                "red_mean",
                "green_mean",
                "blue_mean"
            ]
        ]
        .agg(["mean", "median", "std"])
    )

    return result


# ---------------------------------------------------------
# SUMMARY STATISTICS
# ---------------------------------------------------------

def summary_statistics(statistics):
    """
    Generate descriptive statistics for numerical image features.
    """

    return statistics[
        [
            "brightness",
            "contrast",
            "red_mean",
            "green_mean",
            "blue_mean"
        ]
    ].describe().T


# ---------------------------------------------------------
# SKEWNESS
# ---------------------------------------------------------

def calculate_skewness(statistics):
    """
    Calculate skewness of image-level numerical features.
    """

    columns = [
        "brightness",
        "contrast",
        "red_mean",
        "green_mean",
        "blue_mean"
    ]

    skewness = statistics[columns].skew()

    return skewness.to_frame(name="skewness")


# ---------------------------------------------------------
# SAVE ANALYSIS RESULTS
# ---------------------------------------------------------

def save_statistics(statistics):
    """
    Save image-level statistics for later analysis.
    """

    output_path = PROCESSED_DIR / "image_statistics.csv"

    statistics.to_csv(output_path, index=False)

    return output_path


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("SentriVision AI - Data Analyst EDA")
    print("=" * 60)

    print("\nLoading CIFAR-10 dataset...")

    images, labels = load_training_data()

    print(f"Images loaded: {len(images):,}")
    print(f"Image shape: {images.shape}")

    print("\nDataset Overview")

    overview = dataset_overview(images, labels)

    for key, value in overview.items():
        print(f"{key}: {value}")

    print("\nClass Distribution")

    distribution = class_distribution(labels)

    print(distribution.to_string(index=False))

    print("\nCalculating image statistics...")

    statistics = calculate_image_statistics(images, labels)

    print("\nSummary Statistics")

    print(summary_statistics(statistics))

    print("\nSkewness")

    print(calculate_skewness(statistics))

    print("\nSaving image statistics...")

    output = save_statistics(statistics)

    print(f"Saved to: {output}")

    print("\nEDA data preparation completed successfully.")