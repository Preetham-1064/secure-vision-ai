"""
SentriVision AI - Data Preprocessing Pipeline

Data Engineer responsibilities:
- Load CIFAR-10 data
- Validate dataset structure
- Validate image shape and pixel values
- Validate labels
- Check missing/invalid values
- Check class distribution
- Detect duplicate images
- Check train/test data leakage
- Generate data quality audit report
- Prepare train/test data
"""

from pathlib import Path
import pickle
import hashlib

import numpy as np
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = (
    PROJECT_ROOT.parent
    / "cifar-10-data"
    / "cifar-10-batches-py"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

REPORT_PATH = (
    PROCESSED_DIR / "data_quality_report.csv"
)


# ==========================================================
# CIFAR-10 CONFIGURATION
# ==========================================================

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
    "truck",
]

NUM_CLASSES = 10

IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_CHANNELS = 3


# ==========================================================
# LOAD ONE CIFAR-10 BATCH
# ==========================================================

def load_batch(file_path):
    """Load a single CIFAR-10 batch."""

    with open(file_path, "rb") as file:
        batch = pickle.load(file, encoding="bytes")

    images = batch[b"data"]
    labels = np.array(batch[b"labels"])

    return images, labels


# ==========================================================
# LOAD TRAINING DATA
# ==========================================================

def load_training_data():
    """Load all five CIFAR-10 training batches."""

    images = []
    labels = []

    for batch_number in range(1, 6):

        file_path = DATASET_DIR / f"data_batch_{batch_number}"

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing CIFAR-10 batch: {file_path}"
            )

        batch_images, batch_labels = load_batch(file_path)

        images.append(batch_images)
        labels.append(batch_labels)

    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)

    return images, labels


# ==========================================================
# LOAD TEST DATA
# ==========================================================

def load_test_data():
    """Load the CIFAR-10 test batch."""

    file_path = DATASET_DIR / "test_batch"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing CIFAR-10 test batch: {file_path}"
        )

    return load_batch(file_path)


# ==========================================================
# BASIC DATA VALIDATION
# ==========================================================

def validate_data(images, labels, dataset_name):
    """Run basic data-quality checks."""

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} DATA QUALITY CHECK")
    print(f"{'=' * 60}")

    print(f"Number of samples : {len(images)}")
    print(f"Image shape       : {images.shape}")
    print(f"Label shape       : {labels.shape}")
    print(f"Data type         : {images.dtype}")

    # Missing values
    missing_values = np.isnan(images).sum()

    print(f"Missing values    : {missing_values}")

    # Pixel range
    min_pixel = images.min()
    max_pixel = images.max()

    print(f"Minimum pixel     : {min_pixel}")
    print(f"Maximum pixel     : {max_pixel}")

    # Labels
    unique_labels = np.unique(labels)

    print(f"Unique labels     : {unique_labels}")

    invalid_labels = [
        label
        for label in unique_labels
        if label < 0 or label >= NUM_CLASSES
    ]

    print(f"Invalid labels    : {len(invalid_labels)}")

    # Expected sample count
    expected_samples = (
        50000
        if dataset_name == "TRAIN"
        else 10000
    )

    print(f"Expected samples  : {expected_samples}")

    if len(images) != expected_samples:
        raise ValueError(
            f"{dataset_name} dataset has an unexpected "
            f"number of samples."
        )

    if invalid_labels:
        raise ValueError(
            f"Invalid labels found: {invalid_labels}"
        )

    if min_pixel < 0 or max_pixel > 255:
        raise ValueError(
            "Pixel values outside the valid CIFAR-10 "
            "range were found."
        )

    print("Validation status : PASSED")

    return {
        "sample_count": len(images),
        "image_shape": str(images.shape),
        "data_type": str(images.dtype),
        "missing_values": int(missing_values),
        "min_pixel": int(min_pixel),
        "max_pixel": int(max_pixel),
        "unique_labels": len(unique_labels),
        "invalid_labels": len(invalid_labels),
        "expected_samples": expected_samples,
    }


# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

def check_class_distribution(labels, dataset_name):
    """Display number and percentage of samples per class."""

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} CLASS DISTRIBUTION")
    print(f"{'=' * 60}")

    unique, counts = np.unique(
        labels,
        return_counts=True
    )

    total = len(labels)

    distribution = {}

    for label, count in zip(unique, counts):

        percentage = (count / total) * 100

        distribution[int(label)] = {
            "count": int(count),
            "percentage": float(percentage),
        }

        print(
            f"{label}: "
            f"{CLASS_NAMES[label]:12s} "
            f"{count:5d} samples "
            f"({percentage:.2f}%)"
        )

    return distribution


# ==========================================================
# IMAGE HASHING
# ==========================================================

def calculate_image_hash(image):
    """Generate an MD5 hash for an image."""

    return hashlib.md5(
        image.tobytes()
    ).hexdigest()


# ==========================================================
# DUPLICATE CHECK
# ==========================================================

def check_duplicates(images, dataset_name):
    """Detect exact duplicate images."""

    print(f"\n{'=' * 60}")
    print(f"{dataset_name} DUPLICATE CHECK")
    print(f"{'=' * 60}")

    hashes = set()
    duplicate_count = 0

    for image in images:

        image_hash = calculate_image_hash(image)

        if image_hash in hashes:
            duplicate_count += 1
        else:
            hashes.add(image_hash)

    unique_count = len(hashes)

    print(f"Total images     : {len(images)}")
    print(f"Unique images    : {unique_count}")
    print(f"Duplicate images : {duplicate_count}")

    if duplicate_count == 0:
        print("Duplicate status : PASSED")
    else:
        print("Duplicate status : REVIEW REQUIRED")

    return hashes, duplicate_count


# ==========================================================
# TRAIN / TEST LEAKAGE CHECK
# ==========================================================

def check_train_test_leakage(
    train_hashes,
    test_images
):
    """Check for exact train/test image overlap."""

    print(f"\n{'=' * 60}")
    print("TRAIN / TEST LEAKAGE CHECK")
    print(f"{'=' * 60}")

    overlap_count = 0

    for image in test_images:

        image_hash = calculate_image_hash(image)

        if image_hash in train_hashes:
            overlap_count += 1

    print(f"Training images : {len(train_hashes)}")
    print(f"Test images     : {len(test_images)}")
    print(f"Overlap images  : {overlap_count}")

    if overlap_count == 0:
        print("Leakage status  : PASSED")
    else:
        print("Leakage status  : REVIEW REQUIRED")

    return overlap_count


# ==========================================================
# RESHAPE IMAGES
# ==========================================================

def reshape_images(images):
    """
    Convert CIFAR-10 flattened arrays.

    From:
        (N, 3072)

    To:
        (N, 32, 32, 3)
    """

    images = images.reshape(
        -1,
        IMAGE_CHANNELS,
        IMAGE_HEIGHT,
        IMAGE_WIDTH
    )

    images = images.transpose(
        0,
        2,
        3,
        1
    )

    return images


# ==========================================================
# GENERATE DATA QUALITY REPORT
# ==========================================================

def generate_quality_report(
    train_info,
    test_info,
    train_distribution,
    test_distribution,
    train_duplicates,
    test_duplicates,
    leakage_count
):
    """
    Generate a CSV data-quality audit report.
    """

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_rows = []

    # ------------------------------------------------------
    # Dataset-level checks
    # ------------------------------------------------------

    report_rows.extend([
        {
            "dataset": "TRAIN",
            "check": "Sample Count",
            "value": train_info["sample_count"],
            "status": (
                "PASS"
                if train_info["sample_count"] == 50000
                else "FAIL"
            ),
        },
        {
            "dataset": "TEST",
            "check": "Sample Count",
            "value": test_info["sample_count"],
            "status": (
                "PASS"
                if test_info["sample_count"] == 10000
                else "FAIL"
            ),
        },
        {
            "dataset": "TRAIN",
            "check": "Missing Values",
            "value": train_info["missing_values"],
            "status": (
                "PASS"
                if train_info["missing_values"] == 0
                else "FAIL"
            ),
        },
        {
            "dataset": "TEST",
            "check": "Missing Values",
            "value": test_info["missing_values"],
            "status": (
                "PASS"
                if test_info["missing_values"] == 0
                else "FAIL"
            ),
        },
        {
            "dataset": "TRAIN",
            "check": "Pixel Range",
            "value": (
                f"{train_info['min_pixel']}-"
                f"{train_info['max_pixel']}"
            ),
            "status": (
                "PASS"
                if (
                    train_info["min_pixel"] >= 0
                    and train_info["max_pixel"] <= 255
                )
                else "FAIL"
            ),
        },
        {
            "dataset": "TEST",
            "check": "Pixel Range",
            "value": (
                f"{test_info['min_pixel']}-"
                f"{test_info['max_pixel']}"
            ),
            "status": (
                "PASS"
                if (
                    test_info["min_pixel"] >= 0
                    and test_info["max_pixel"] <= 255
                )
                else "FAIL"
            ),
        },
        {
            "dataset": "TRAIN",
            "check": "Invalid Labels",
            "value": train_info["invalid_labels"],
            "status": (
                "PASS"
                if train_info["invalid_labels"] == 0
                else "FAIL"
            ),
        },
        {
            "dataset": "TEST",
            "check": "Invalid Labels",
            "value": test_info["invalid_labels"],
            "status": (
                "PASS"
                if test_info["invalid_labels"] == 0
                else "FAIL"
            ),
        },
        {
            "dataset": "TRAIN",
            "check": "Duplicate Images",
            "value": train_duplicates,
            "status": (
                "PASS"
                if train_duplicates == 0
                else "REVIEW"
            ),
        },
        {
            "dataset": "TEST",
            "check": "Duplicate Images",
            "value": test_duplicates,
            "status": (
                "PASS"
                if test_duplicates == 0
                else "REVIEW"
            ),
        },
        {
            "dataset": "TRAIN/TEST",
            "check": "Exact Image Overlap",
            "value": leakage_count,
            "status": (
                "PASS"
                if leakage_count == 0
                else "FAIL"
            ),
        },
    ])

    # ------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------

    for label, values in train_distribution.items():

        report_rows.append({
            "dataset": "TRAIN",
            "check": (
                f"Class Distribution - "
                f"{CLASS_NAMES[label]}"
            ),
            "value": (
                f"{values['count']} "
                f"({values['percentage']:.2f}%)"
            ),
            "status": "INFO",
        })

    for label, values in test_distribution.items():

        report_rows.append({
            "dataset": "TEST",
            "check": (
                f"Class Distribution - "
                f"{CLASS_NAMES[label]}"
            ),
            "value": (
                f"{values['count']} "
                f"({values['percentage']:.2f}%)"
            ),
            "status": "INFO",
        })

    report = pd.DataFrame(report_rows)

    report.to_csv(
        REPORT_PATH,
        index=False
    )

    print(
        f"\nData quality report saved to:\n"
        f"{REPORT_PATH}"
    )


# ==========================================================
# MAIN PIPELINE
# ==========================================================

def main():

    print(
        "SentriVision AI - "
        "Data Preprocessing Pipeline"
    )

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------

    X_train, y_train = load_training_data()
    X_test, y_test = load_test_data()

    print("\nRaw data loaded successfully.")

    # ------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------

    train_info = validate_data(
        X_train,
        y_train,
        "TRAIN"
    )

    test_info = validate_data(
        X_test,
        y_test,
        "TEST"
    )

    # ------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------

    train_distribution = check_class_distribution(
        y_train,
        "TRAIN"
    )

    test_distribution = check_class_distribution(
        y_test,
        "TEST"
    )

    # ------------------------------------------------------
    # Duplicate checks
    # ------------------------------------------------------

    train_hashes, train_duplicates = (
        check_duplicates(
            X_train,
            "TRAIN"
        )
    )

    test_hashes, test_duplicates = (
        check_duplicates(
            X_test,
            "TEST"
        )
    )

    # ------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------

    leakage_count = check_train_test_leakage(
        train_hashes,
        X_test
    )

    # ------------------------------------------------------
    # Generate audit report
    # ------------------------------------------------------

    generate_quality_report(
        train_info,
        test_info,
        train_distribution,
        test_distribution,
        train_duplicates,
        test_duplicates,
        leakage_count
    )

    # ------------------------------------------------------
    # Reshape images
    # ------------------------------------------------------

    X_train = reshape_images(X_train)
    X_test = reshape_images(X_test)

    print("\nAfter reshaping:")

    print(
        "Training images:",
        X_train.shape
    )

    print(
        "Test images:",
        X_test.shape
    )

    # ------------------------------------------------------
    # Class mapping
    # ------------------------------------------------------

    print("\nClass mapping:")

    for index, class_name in enumerate(CLASS_NAMES):

        print(
            f"{index}: {class_name}"
        )

    # ------------------------------------------------------
    # Final summary
    # ------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("DATA QUALITY SUMMARY")
    print(f"{'=' * 60}")

    print(
        f"Training duplicates : {train_duplicates}"
    )

    print(
        f"Test duplicates     : {test_duplicates}"
    )

    print(
        f"Train/Test overlap  : {leakage_count}"
    )

    if (
        train_duplicates == 0
        and test_duplicates == 0
        and leakage_count == 0
    ):
        print(
            "\nOverall data quality status : PASSED"
        )
    else:
        print(
            "\nOverall data quality status : REVIEW REQUIRED"
        )

    print(
        "\nData preprocessing pipeline "
        "completed successfully."
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()