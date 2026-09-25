"""
Secure Vision AI - Data Analyst EDA Module

Role:
    Data Analyst

Purpose:
    Exploratory and statistical analysis of the CIFAR-10 dataset.

Analysis performed:
    1. Dataset overview
    2. Class distribution
    3. Representative image inspection
    4. Image-level statistics
    5. RGB channel analysis
    6. Brightness analysis
    7. Contrast analysis
    8. Distribution and skewness
    9. IQR-based anomaly detection
    10. Extreme image inspection
    11. Class-wise anomaly analysis

Outputs:
    data/processed/image_statistics.csv
    data/processed/class_distribution.csv
    data/processed/anomaly_summary.csv

The Data Engineer's preprocessing pipeline is not modified.
"""

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT.parent
    / "cifar-10-data"
    / "cifar-10-batches-py"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CIFAR-10 CLASS NAMES
# ============================================================

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


# ============================================================
# 1. LOAD CIFAR-10 BATCH
# ============================================================

def load_cifar_batch(file_path):
    """
    Load a single CIFAR-10 batch.

    Parameters
    ----------
    file_path : Path
        Path to CIFAR-10 batch file.

    Returns
    -------
    images : np.ndarray
        Images in shape (N, 32, 32, 3).

    labels : np.ndarray
        Integer class labels.
    """

    with open(file_path, "rb") as file:

        batch = pickle.load(
            file,
            encoding="bytes"
        )

    images = batch[b"data"]

    labels = np.array(
        batch[b"labels"]
    )

    # CIFAR-10 original format:
    # (N, 3072)
    #
    # Convert to:
    # (N, 3, 32, 32)

    images = images.reshape(
        -1,
        3,
        32,
        32
    )

    # Convert to:
    # (N, 32, 32, 3)

    images = images.transpose(
        0,
        2,
        3,
        1
    )

    return images, labels


# ============================================================
# 2. LOAD TRAINING DATA
# ============================================================

def load_training_data():
    """
    Load all five CIFAR-10 training batches.

    Returns
    -------
    images : np.ndarray
        50,000 training images.

    labels : np.ndarray
        50,000 labels.
    """

    images = []
    labels = []

    for batch_number in range(1, 6):

        file_path = (
            DATASET_DIR
            / f"data_batch_{batch_number}"
        )

        if not file_path.exists():

            raise FileNotFoundError(
                f"CIFAR-10 batch not found: {file_path}"
            )

        batch_images, batch_labels = (
            load_cifar_batch(file_path)
        )

        images.append(
            batch_images
        )

        labels.append(
            batch_labels
        )

    images = np.concatenate(
        images,
        axis=0
    )

    labels = np.concatenate(
        labels,
        axis=0
    )

    return images, labels


# ============================================================
# 3. DATASET OVERVIEW
# ============================================================

def dataset_overview(images, labels):
    """
    Generate basic dataset statistics.
    """

    overview = {

        "number_of_images": len(images),

        "image_height": images.shape[1],

        "image_width": images.shape[2],

        "channels": images.shape[3],

        "number_of_classes": len(
            np.unique(labels)
        ),

        "pixel_minimum": int(
            images.min()
        ),

        "pixel_maximum": int(
            images.max()
        ),

        "pixel_mean": float(
            images.mean()
        ),

        "pixel_std": float(
            images.std()
        )
    }

    return overview


# ============================================================
# 4. CLASS DISTRIBUTION
# ============================================================

def class_distribution(labels):
    """
    Calculate image count and percentage for each class.
    """

    counts = np.bincount(
        labels,
        minlength=10
    )

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


# ============================================================
# 5. IMAGE-LEVEL STATISTICS
# ============================================================

def calculate_image_statistics(
    images,
    labels
):
    """
    Calculate image-level brightness,
    contrast and RGB statistics.
    """

    images_float = images.astype(
        np.float32
    )

    # --------------------------------------------
    # Brightness
    # --------------------------------------------

    brightness = (
        images_float.mean(
            axis=(1, 2, 3)
        )
    )

    # --------------------------------------------
    # Contrast
    # --------------------------------------------

    contrast = (
        images_float.std(
            axis=(1, 2, 3)
        )
    )

    # --------------------------------------------
    # RGB channel means
    # --------------------------------------------

    red_mean = (
        images_float[:, :, :, 0]
        .mean(axis=(1, 2))
    )

    green_mean = (
        images_float[:, :, :, 1]
        .mean(axis=(1, 2))
    )

    blue_mean = (
        images_float[:, :, :, 2]
        .mean(axis=(1, 2))
    )

    statistics = pd.DataFrame({

        "label": labels,

        "class_name": [
            CLASS_NAMES[label]
            for label in labels
        ],

        "brightness": brightness,

        "contrast": contrast,

        "red_mean": red_mean,

        "green_mean": green_mean,

        "blue_mean": blue_mean
    })

    return statistics


# ============================================================
# 6. CLASS-WISE STATISTICS
# ============================================================

def class_statistics(statistics):
    """
    Calculate mean, median and standard deviation
    of image-level features by class.
    """

    features = [
        "brightness",
        "contrast",
        "red_mean",
        "green_mean",
        "blue_mean"
    ]

    result = (
        statistics
        .groupby("class_name")[features]
        .agg(
            [
                "mean",
                "median",
                "std"
            ]
        )
    )

    return result


# ============================================================
# 7. SUMMARY STATISTICS
# ============================================================

def summary_statistics(statistics):
    """
    Generate descriptive statistics.
    """

    features = [
        "brightness",
        "contrast",
        "red_mean",
        "green_mean",
        "blue_mean"
    ]

    return (
        statistics[features]
        .describe()
        .T
    )


# ============================================================
# 8. SKEWNESS
# ============================================================

def calculate_skewness(statistics):
    """
    Calculate skewness for numerical image features.
    """

    features = [
        "brightness",
        "contrast",
        "red_mean",
        "green_mean",
        "blue_mean"
    ]

    skewness = (
        statistics[features]
        .skew()
    )

    return skewness.to_frame(
        name="skewness"
    )


# ============================================================
# 9. IQR OUTLIER DETECTION
# ============================================================

def detect_iqr_outliers(
    statistics,
    column
):
    """
    Detect statistical outliers using IQR.

    IQR = Q3 - Q1

    Lower bound = Q1 - 1.5 * IQR

    Upper bound = Q3 + 1.5 * IQR
    """

    q1 = statistics[column].quantile(
        0.25
    )

    q3 = statistics[column].quantile(
        0.75
    )

    iqr = q3 - q1

    lower_bound = (
        q1 - 1.5 * iqr
    )

    upper_bound = (
        q3 + 1.5 * iqr
    )

    outliers = statistics[
        (statistics[column] < lower_bound)
        |
        (statistics[column] > upper_bound)
    ].copy()

    return {

        "q1": q1,

        "q3": q3,

        "iqr": iqr,

        "lower_bound": lower_bound,

        "upper_bound": upper_bound,

        "outliers": outliers
    }


# ============================================================
# 10. ANOMALY SUMMARY
# ============================================================

def anomaly_summary(statistics):
    """
    Generate anomaly summary for
    brightness, contrast and RGB features.
    """

    features = [
        "brightness",
        "contrast",
        "red_mean",
        "green_mean",
        "blue_mean"
    ]

    results = []

    for feature in features:

        result = detect_iqr_outliers(
            statistics,
            feature
        )

        outlier_count = len(
            result["outliers"]
        )

        outlier_percentage = (
            outlier_count
            / len(statistics)
            * 100
        )

        results.append({

            "feature": feature,

            "q1": result["q1"],

            "q3": result["q3"],

            "iqr": result["iqr"],

            "lower_bound":
                result["lower_bound"],

            "upper_bound":
                result["upper_bound"],

            "outlier_count":
                outlier_count,

            "outlier_percentage":
                outlier_percentage
        })

    return pd.DataFrame(
        results
    )


# ============================================================
# 11. EXTREME IMAGES
# ============================================================

def get_extreme_images(
    statistics,
    column,
    n=10
):
    """
    Get lowest and highest observations
    for a selected feature.
    """

    lowest = (
        statistics
        .nsmallest(
            n,
            column
        )
    )

    highest = (
        statistics
        .nlargest(
            n,
            column
        )
    )

    return lowest, highest


# ============================================================
# 12. SAVE ANALYSIS RESULTS
# ============================================================

def save_analysis_results(
    image_stats,
    distribution,
    anomalies
):
    """
    Save Data Analyst outputs.
    """

    image_stats_path = (
        PROCESSED_DIR
        / "image_statistics.csv"
    )

    distribution_path = (
        PROCESSED_DIR
        / "class_distribution.csv"
    )

    anomaly_path = (
        PROCESSED_DIR
        / "anomaly_summary.csv"
    )

    image_stats.to_csv(
        image_stats_path,
        index=False
    )

    distribution.to_csv(
        distribution_path,
        index=False
    )

    anomalies.to_csv(
        anomaly_path,
        index=False
    )

    return (
        image_stats_path,
        distribution_path,
        anomaly_path
    )


# ============================================================
# 13. PRINT TOP EDA INSIGHTS
# ============================================================

def generate_eda_insights(
    distribution,
    statistics,
    anomalies
):
    """
    Generate data-driven summary values
    for the final EDA report.
    """

    print()
    print("=" * 70)
    print("TOP EDA INSIGHTS")
    print("=" * 70)

    # --------------------------------------------------------
    # Insight 1 - Class Distribution
    # --------------------------------------------------------

    min_count = distribution[
        "image_count"
    ].min()

    max_count = distribution[
        "image_count"
    ].max()

    min_class = distribution.loc[
        distribution["image_count"].idxmin(),
        "class_name"
    ]

    max_class = distribution.loc[
        distribution["image_count"].idxmax(),
        "class_name"
    ]

    print("\nInsight 1 - Class Distribution")

    print(
        f"Class counts range from "
        f"{min_count:,} to {max_count:,} images."
    )

    print(
        f"Lowest-count class: {min_class}"
    )

    print(
        f"Highest-count class: {max_class}"
    )

    # --------------------------------------------------------
    # Insight 2 - Image Characteristics
    # --------------------------------------------------------

    brightness_mean = (
        statistics["brightness"].mean()
    )

    contrast_mean = (
        statistics["contrast"].mean()
    )

    print("\nInsight 2 - Image Characteristics")

    print(
        f"Average image brightness: "
        f"{brightness_mean:.2f}"
    )

    print(
        f"Average image contrast: "
        f"{contrast_mean:.2f}"
    )

    # --------------------------------------------------------
    # Insight 3 - Anomalies
    # --------------------------------------------------------

    highest_anomaly = anomalies.loc[
        anomalies["outlier_percentage"].idxmax()
    ]

    print("\nInsight 3 - Statistical Anomalies")

    print(
        f"Feature with the highest proportion "
        f"of statistical outliers: "
        f"{highest_anomaly['feature']}"
    )

    print(
        f"Outliers: "
        f"{int(highest_anomaly['outlier_count']):,} "
        f"({highest_anomaly['outlier_percentage']:.2f}%)"
    )

    print(
        "\nNote: Statistical outliers are not "
        "automatically considered corrupted images."
    )


# ============================================================
# 14. MAIN DATA ANALYST PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print("SentriVision AI - Data Analyst EDA")
    print("=" * 70)

    # --------------------------------------------------------
    # Dataset validation
    # --------------------------------------------------------

    print("\n[1/9] Loading CIFAR-10 training data...")

    images, labels = (
        load_training_data()
    )

    print(
        f"Images loaded: {len(images):,}"
    )

    print(
        f"Image shape: {images.shape}"
    )

    print(
        f"Labels shape: {labels.shape}"
    )

    # --------------------------------------------------------
    # Dataset overview
    # --------------------------------------------------------

    print(
        "\n[2/9] Dataset overview..."
    )

    overview = dataset_overview(
        images,
        labels
    )

    for key, value in overview.items():

        print(
            f"{key}: {value}"
        )

    # --------------------------------------------------------
    # Class distribution
    # --------------------------------------------------------

    print(
        "\n[3/9] Class distribution..."
    )

    distribution = class_distribution(
        labels
    )

    print(
        distribution.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Image statistics
    # --------------------------------------------------------

    print(
        "\n[4/9] Calculating image statistics..."
    )

    statistics = (
        calculate_image_statistics(
            images,
            labels
        )
    )

    print(
        "\nDescriptive Statistics:"
    )

    print(
        summary_statistics(
            statistics
        )
    )

    # --------------------------------------------------------
    # Class-wise statistics
    # --------------------------------------------------------

    print(
        "\n[5/9] Class-wise statistics..."
    )

    print(
        class_statistics(
            statistics
        )
    )

    # --------------------------------------------------------
    # Skewness
    # --------------------------------------------------------

    print(
        "\n[6/9] Calculating skewness..."
    )

    print(
        calculate_skewness(
            statistics
        )
    )

    # --------------------------------------------------------
    # Anomaly detection
    # --------------------------------------------------------

    print(
        "\n[7/9] Detecting statistical anomalies..."
    )

    anomalies = anomaly_summary(
        statistics
    )

    print(
        anomalies.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Extreme observations
    # --------------------------------------------------------

    print(
        "\n[8/9] Inspecting extreme observations..."
    )

    darkest, brightest = (
        get_extreme_images(
            statistics,
            "brightness",
            n=5
        )
    )

    lowest_contrast, highest_contrast = (
        get_extreme_images(
            statistics,
            "contrast",
            n=5
        )
    )

    print(
        "\nDarkest images:"
    )

    print(
        darkest[
            [
                "class_name",
                "brightness"
            ]
        ].to_string()
    )

    print(
        "\nBrightest images:"
    )

    print(
        brightest[
            [
                "class_name",
                "brightness"
            ]
        ].to_string()
    )

    print(
        "\nLowest contrast images:"
    )

    print(
        lowest_contrast[
            [
                "class_name",
                "contrast"
            ]
        ].to_string()
    )

    print(
        "\nHighest contrast images:"
    )

    print(
        highest_contrast[
            [
                "class_name",
                "contrast"
            ]
        ].to_string()
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    print(
        "\n[9/9] Saving Data Analyst outputs..."
    )

    output_paths = save_analysis_results(
        statistics,
        distribution,
        anomalies
    )

    for path in output_paths:

        print(
            f"Saved: {path}"
        )

    # --------------------------------------------------------
    # Insights
    # --------------------------------------------------------

    generate_eda_insights(
        distribution,
        statistics,
        anomalies
    )

    print()
    print("=" * 70)
    print("DATA ANALYST EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()