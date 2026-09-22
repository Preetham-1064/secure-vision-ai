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
- Create reproducible train/validation/test splits
- Normalize image pixels
- Provide ML-ready data
- Save preprocessing metadata
"""

from pathlib import Path
import pickle
import hashlib
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


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

CONFIG_PATH = (
    PROCESSED_DIR / "preprocessing_config.json"
)

SPLIT_PATH = (
    PROCESSED_DIR / "dataset_split_indices.csv"
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

RANDOM_STATE = 42

VALIDATION_SIZE = 0.10


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

    missing_values = np.isnan(images).sum()

    print(f"Missing values    : {missing_values}")

    min_pixel = images.min()
    max_pixel = images.max()

    print(f"Minimum pixel     : {min_pixel}")
    print(f"Maximum pixel     : {max_pixel}")

    unique_labels = np.unique(labels)

    print(f"Unique labels     : {unique_labels}")

    invalid_labels = [
        label
        for label in unique_labels
        if label < 0 or label >= NUM_CLASSES
    ]

    print(f"Invalid labels    : {len(invalid_labels)}")

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
# NORMALIZE IMAGES
# ==========================================================

def normalize_images(images):
    """
    Normalize image pixel values from [0, 255]
    to [0, 1].

    Output dtype:
        float32
    """

    images = images.astype(
        np.float32
    )

    images = images / 255.0

    return images


# ==========================================================
# CREATE TRAIN / VALIDATION SPLIT
# ==========================================================

def create_train_validation_split(
    images,
    labels
):
    """
    Create a reproducible stratified train/validation split.

    Training data:
        90%

    Validation data:
        10%

    Test data is never passed into this function.
    """

    (
        X_train,
        X_validation,
        y_train,
        y_validation
    ) = train_test_split(
        images,
        labels,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    return (
        X_train,
        X_validation,
        y_train,
        y_validation
    )


# ==========================================================
# CREATE SPLIT INDICES
# ==========================================================

def create_split_indices(labels):
    """
    Create reproducible train/validation indices.

    This keeps a lightweight record of the split
    without saving the large image arrays.
    """

    indices = np.arange(len(labels))

    train_indices, validation_indices = (
        train_test_split(
            indices,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels
        )
    )

    return (
        train_indices,
        validation_indices
    )


# ==========================================================
# SAVE SPLIT METADATA
# ==========================================================

def save_split_metadata(
    train_indices,
    validation_indices,
    test_size
):
    """
    Save train/validation/test split information
    as a lightweight CSV.
    """

    train_records = pd.DataFrame({
        "source_index": train_indices,
        "split": "train"
    })

    validation_records = pd.DataFrame({
        "source_index": validation_indices,
        "split": "validation"
    })

    test_indices = np.arange(test_size)

    test_records = pd.DataFrame({
        "source_index": test_indices,
        "split": "test"
    })

    split_records = pd.concat(
        [
            train_records,
            validation_records,
            test_records
        ],
        ignore_index=True
    )

    split_records.to_csv(
        SPLIT_PATH,
        index=False
    )

    print(
        f"\nSplit metadata saved to:\n"
        f"{SPLIT_PATH}"
    )


# ==========================================================
# SAVE PREPROCESSING CONFIGURATION
# ==========================================================

def save_preprocessing_config():
    """Save preprocessing settings for reproducibility."""

    config = {
        "dataset": "CIFAR-10",
        "image_height": IMAGE_HEIGHT,
        "image_width": IMAGE_WIDTH,
        "image_channels": IMAGE_CHANNELS,
        "number_of_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "normalization": {
            "method": "divide_by_255",
            "input_range": [0, 255],
            "output_range": [0.0, 1.0],
            "output_dtype": "float32"
        },
        "split": {
            "training_samples": 45000,
            "validation_samples": 5000,
            "test_samples": 10000,
            "validation_size": VALIDATION_SIZE,
            "stratified": True,
            "random_state": RANDOM_STATE
        },
        "leakage_prevention": {
            "duplicate_check": "MD5 exact pixel hash",
            "train_test_overlap_check": True,
            "test_set_used_for_split": False
        }
    }

    with open(
        CONFIG_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            config,
            file,
            indent=4
        )

    print(
        f"Preprocessing configuration saved to:\n"
        f"{CONFIG_PATH}"
    )


# ==========================================================
# PREPARE ML-READY DATA
# ==========================================================

def prepare_ml_data(
    X_train,
    y_train,
    X_test,
    y_test
):
    """
    Create ML-ready train, validation and test datasets.

    Steps:
        1. Split training data.
        2. Reshape images.
        3. Normalize pixel values.
    """

    (
        X_train,
        X_validation,
        y_train,
        y_validation
    ) = create_train_validation_split(
        X_train,
        y_train
    )

    # Reshape
    X_train = reshape_images(X_train)
    X_validation = reshape_images(X_validation)
    X_test = reshape_images(X_test)

    # Normalize
    X_train = normalize_images(X_train)
    X_validation = normalize_images(X_validation)
    X_test = normalize_images(X_test)

    return (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )


# ==========================================================
# VALIDATE PROCESSED DATA
# ==========================================================

def validate_processed_data(
    X_train,
    X_validation,
    X_test,
    y_train,
    y_validation,
    y_test
):
    """Validate the final ML-ready datasets."""

    print(f"\n{'=' * 60}")
    print("ML-READY DATA VALIDATION")
    print(f"{'=' * 60}")

    print(
        f"Training data     : "
        f"{X_train.shape}"
    )

    print(
        f"Validation data   : "
        f"{X_validation.shape}"
    )

    print(
        f"Test data         : "
        f"{X_test.shape}"
    )

    print(
        f"Training labels   : "
        f"{y_train.shape}"
    )

    print(
        f"Validation labels : "
        f"{y_validation.shape}"
    )

    print(
        f"Test labels       : "
        f"{y_test.shape}"
    )

    print(
        f"\nTraining range    : "
        f"{X_train.min():.4f} - "
        f"{X_train.max():.4f}"
    )

    print(
        f"Validation range  : "
        f"{X_validation.min():.4f} - "
        f"{X_validation.max():.4f}"
    )

    print(
        f"Test range        : "
        f"{X_test.min():.4f} - "
        f"{X_test.max():.4f}"
    )

    # Shape checks
    expected_shape = (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        IMAGE_CHANNELS
    )

    if X_train.shape[1:] != expected_shape:
        raise ValueError(
            "Training image shape is incorrect."
        )

    if X_validation.shape[1:] != expected_shape:
        raise ValueError(
            "Validation image shape is incorrect."
        )

    if X_test.shape[1:] != expected_shape:
        raise ValueError(
            "Test image shape is incorrect."
        )

    # Range checks
    if (
        X_train.min() < 0
        or X_train.max() > 1
        or X_validation.min() < 0
        or X_validation.max() > 1
        or X_test.min() < 0
        or X_test.max() > 1
    ):
        raise ValueError(
            "Normalized pixel values are outside "
            "the expected [0, 1] range."
        )

    # Data type
    if X_train.dtype != np.float32:
        raise ValueError(
            "Training data must use float32."
        )

    print(
        "\nML-ready data validation : PASSED"
    )


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
    """Generate a CSV data-quality audit report."""

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report_rows = []

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

    X_train_raw, y_train_raw = (
        load_training_data()
    )

    X_test_raw, y_test_raw = (
        load_test_data()
    )

    print("\nRaw data loaded successfully.")

    # ------------------------------------------------------
    # Basic validation
    # ------------------------------------------------------

    train_info = validate_data(
        X_train_raw,
        y_train_raw,
        "TRAIN"
    )

    test_info = validate_data(
        X_test_raw,
        y_test_raw,
        "TEST"
    )

    # ------------------------------------------------------
    # Class distribution
    # ------------------------------------------------------

    train_distribution = check_class_distribution(
        y_train_raw,
        "TRAIN"
    )

    test_distribution = check_class_distribution(
        y_test_raw,
        "TEST"
    )

    # ------------------------------------------------------
    # Duplicate checks
    # ------------------------------------------------------

    train_hashes, train_duplicates = (
        check_duplicates(
            X_train_raw,
            "TRAIN"
        )
    )

    test_hashes, test_duplicates = (
        check_duplicates(
            X_test_raw,
            "TEST"
        )
    )

    # ------------------------------------------------------
    # Leakage check
    # ------------------------------------------------------

    leakage_count = check_train_test_leakage(
        train_hashes,
        X_test_raw
    )

    # ------------------------------------------------------
    # Generate quality report
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
    # Create split metadata
    # ------------------------------------------------------

    (
        train_indices,
        validation_indices
    ) = create_split_indices(
        y_train_raw
    )

    save_split_metadata(
        train_indices,
        validation_indices,
        len(y_test_raw)
    )

    # ------------------------------------------------------
    # Save preprocessing configuration
    # ------------------------------------------------------

    save_preprocessing_config()

    # ------------------------------------------------------
    # Prepare ML-ready data
    # ------------------------------------------------------

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    ) = prepare_ml_data(
        X_train_raw,
        y_train_raw,
        X_test_raw,
        y_test_raw
    )

    # ------------------------------------------------------
    # Validate processed data
    # ------------------------------------------------------

    validate_processed_data(
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test
    )

    # ------------------------------------------------------
    # Final information
    # ------------------------------------------------------

    print(f"\n{'=' * 60}")
    print("FINAL DATASET SUMMARY")
    print(f"{'=' * 60}")

    print(
        f"Training samples   : {len(X_train)}"
    )

    print(
        f"Validation samples : {len(X_validation)}"
    )

    print(
        f"Test samples       : {len(X_test)}"
    )

    print(
        f"Image dimensions   : "
        f"{IMAGE_HEIGHT} x "
        f"{IMAGE_WIDTH} x "
        f"{IMAGE_CHANNELS}"
    )

    print(
        "Pixel range        : 0.0 - 1.0"
    )

    print(
        f"Random state       : {RANDOM_STATE}"
    )

    print(
        "\nData Engineer preprocessing "
        "pipeline completed successfully."
    )


# ==========================================================
# RUN
# ==========================================================

if __name__ == "__main__":
    main()