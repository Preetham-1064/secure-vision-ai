# SentriVision AI - Data Engineering Pipeline

## 1. Purpose

This document describes the data acquisition, validation,
quality checks, preprocessing, and dataset preparation
pipeline used for the SentriVision AI project.

The Data Engineer pipeline prepares CIFAR-10 image data
for downstream machine learning development.

---

## 2. Dataset

### Dataset Name

CIFAR-10

### Dataset Type

Image classification dataset.

### Number of Classes

10

### Image Dimensions

32 × 32 pixels

### Number of Channels

3 (RGB)

### Training Samples

50,000

### Test Samples

10,000

### Classes

1. airplane
2. automobile
3. bird
4. cat
5. deer
6. dog
7. frog
8. horse
9. ship
10. truck

---

## 3. Data Source and Provenance

The project uses the CIFAR-10 Python dataset archive.

Because the complete dataset is larger than the GitHub
repository file-size limit, the original archive is stored
externally and downloaded using:

    data/raw/download_data.py

The downloader uses the Google Drive file ID configured
inside the script.

The complete dataset is intentionally NOT stored inside
the Git repository.

Only the following files are maintained inside the repository:

- download_data.py
- README.md
- sample.csv

The full dataset is downloaded and extracted outside the
repository.

---

## 4. Local Dataset Location

After running the downloader, the dataset is expected at:

    ../cifar-10-data/cifar-10-batches-py/

relative to the project repository.

The directory contains:

    data_batch_1
    data_batch_2
    data_batch_3
    data_batch_4
    data_batch_5
    test_batch
    batches.meta

---

## 5. Dataset Download

From the project root, activate the virtual environment
and run:

    python data/raw/download_data.py

The script:

1. Creates the external dataset directory.
2. Downloads the CIFAR-10 archive.
3. Extracts the archive.
4. Verifies that the expected dataset directory exists.

---

## 6. Raw Data Loading

The preprocessing pipeline loads:

- data_batch_1
- data_batch_2
- data_batch_3
- data_batch_4
- data_batch_5

as the training dataset.

The test_batch is loaded separately as the test dataset.

The training batches are concatenated into one training
dataset containing 50,000 samples.

The test batch contains 10,000 samples.

---

## 7. Data Quality Checks

The pipeline performs the following checks.

### 7.1 Sample Count

Expected:

- Training: 50,000
- Test: 10,000

Unexpected sample counts cause the validation pipeline
to raise an error.

### 7.2 Image Data Type

The raw CIFAR-10 image data is loaded as an unsigned
8-bit integer array.

### 7.3 Pixel Range

Raw pixel values must be within:

    0 - 255

Values outside this range are treated as invalid.

### 7.4 Missing Values

The pipeline checks for missing numerical values.

### 7.5 Label Validation

Valid class labels are:

    0 - 9

Any label outside this range is considered invalid.

### 7.6 Class Distribution

The pipeline calculates the number and percentage of
samples belonging to every class.

This provides a basic check for unexpected class imbalance
or missing classes.

---

## 8. Duplicate Detection

The pipeline checks for exact duplicate images.

An MD5 hash is generated from the raw pixel contents of
each image.

The hash is used to identify identical images without
performing expensive pairwise image comparisons.

Duplicate checks are performed separately for:

- Training data
- Test data

Duplicates are reported in the data-quality audit report.

---

## 9. Train/Test Leakage Check

The pipeline checks whether an identical image appears
in both the training and test datasets.

The MD5 hash of every training image is compared with
the hashes of the test images.

Any exact overlap is reported as potential data leakage.

The test dataset is not used when creating the
training/validation split.

---

## 10. Train/Validation/Test Split

The original CIFAR-10 training dataset contains:

    50,000 samples

The training data is divided into:

    Training      : 45,000 samples
    Validation    : 5,000 samples

The original CIFAR-10 test dataset remains:

    Test          : 10,000 samples

Therefore:

    Training      : 45,000
    Validation    : 5,000
    Test          : 10,000

---

## 11. Stratified Splitting

The training/validation split uses stratification.

This ensures that the relative distribution of the
10 CIFAR-10 classes is maintained between the training
and validation subsets.

The split uses:

    Validation size : 10%
    Random state    : 42
    Stratified      : Yes

Using a fixed random state makes the split reproducible.

---

## 12. Image Reshaping

The raw CIFAR-10 Python format stores each image as a
flattened vector containing 3,072 values.

The preprocessing pipeline converts:

    (N, 3072)

into:

    (N, 32, 32, 3)

The three channels represent:

    Red
    Green
    Blue

---

## 13. Pixel Normalization

Raw pixel values are represented in the range:

    0 - 255

The preprocessing pipeline converts them to:

    0.0 - 1.0

using:

    normalized_pixel = pixel / 255.0

The resulting arrays use:

    float32

This representation is suitable for downstream
machine learning and CNN processing.

---

## 14. ML-Ready Dataset

The final datasets produced by the preprocessing pipeline
have the following shapes:

    Training:
    (45000, 32, 32, 3)

    Validation:
    (5000, 32, 32, 3)

    Test:
    (10000, 32, 32, 3)

Pixel values are normalized to:

    0.0 - 1.0

---

## 15. Generated Data Engineering Artifacts

The preprocessing pipeline generates:

### data_quality_report.csv

Location:

    data/processed/data_quality_report.csv

Contains:

- Sample counts
- Missing-value checks
- Pixel-range checks
- Invalid-label checks
- Duplicate-image checks
- Train/test overlap checks
- Class distributions

### dataset_split_indices.csv

Location:

    data/processed/dataset_split_indices.csv

Contains the source indices and assigned split:

- train
- validation
- test

### preprocessing_config.json

Location:

    data/processed/preprocessing_config.json

Contains the reproducibility configuration including:

- Image dimensions
- Number of classes
- Class names
- Normalization method
- Split sizes
- Random state
- Stratification
- Leakage-prevention settings

---

## 16. Reproducibility

The complete preprocessing pipeline can be reproduced
using:

    python src/preprocessing.py

The dataset must first be downloaded using:

    python data/raw/download_data.py

The preprocessing pipeline uses a fixed random state:

    42

Therefore, the train/validation split can be reproduced
consistently.

---

## 17. Data Leakage Prevention

The following measures are implemented:

1. Training and test data are loaded separately.
2. Exact train/test image overlap is checked.
3. The test set is not used during train/validation splitting.
4. The validation split is created only from the original
   training data.
5. A fixed random state is used for reproducibility.

---

## 18. Assumptions

The pipeline assumes:

- The CIFAR-10 archive is complete.
- The expected CIFAR-10 batch files exist.
- Images contain RGB pixel data.
- Pixel values are within 0-255.
- Labels belong to classes 0-9.
- The external dataset archive has not been modified.
- The external dataset location is accessible.

---

## 19. Limitations

This data pipeline performs exact duplicate detection
based on pixel equality.

It does not detect:

- Visually similar images
- Resized duplicates
- Cropped duplicates
- Slightly modified copies
- Semantically identical images

The duplicate check therefore identifies exact pixel-level
duplicates only.

The pipeline also does not perform image augmentation.
Augmentation is left to the downstream machine learning
workflow.

---

## 20. Data Engineer Handoff

The Data Engineer pipeline provides downstream team
members with:

- Validated raw data
- Leakage checks
- Reproducible dataset splitting
- Normalized image inputs
- Dataset metadata
- Data-quality audit information
- Reproducible preprocessing configuration

The Data Scientist can use the resulting train,
validation, and test datasets for model development
without modifying the original raw dataset.