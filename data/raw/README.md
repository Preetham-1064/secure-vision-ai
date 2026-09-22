# Raw Data

This directory contains the files required to obtain and
inspect the CIFAR-10 dataset used by the SentriVision AI
project.

## Dataset

SentriVision AI uses the CIFAR-10 image classification
dataset.

The dataset contains:

- 50,000 training images
- 10,000 test images
- 10 classes
- 32 × 32 RGB images

## Dataset Classes

The ten CIFAR-10 classes are:

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

## Dataset Storage

The complete CIFAR-10 archive is NOT stored in this
GitHub repository because it exceeds the repository's
large-file constraints.

Instead, the full dataset is stored externally and
downloaded using:

    download_data.py

The downloaded dataset is extracted outside the GitHub
repository.

## Download the Dataset

From the project root, activate the virtual environment
and run:

    python data/raw/download_data.py

The script:

1. Creates the external dataset directory.
2. Downloads the CIFAR-10 archive.
3. Extracts the archive.
4. Makes the dataset available to the preprocessing
   pipeline.

## Expected Local Dataset Location

After downloading, the dataset should be available at:

    ../cifar-10-data/cifar-10-batches-py/

The directory should contain:

    data_batch_1
    data_batch_2
    data_batch_3
    data_batch_4
    data_batch_5
    test_batch
    batches.meta

## Repository Files

The following files are intentionally tracked in this
directory:

### download_data.py

Automates downloading and extracting the complete
CIFAR-10 dataset from the configured external storage.

### sample.csv

A 500-row sample of the CIFAR-10 dataset metadata.

It is provided for quick inspection without requiring
the complete dataset.

Columns include:

    image_id
    filename
    label
    label_name

### README.md

Documents the dataset source, storage approach, download
procedure, and expected directory structure.

## Data Pipeline

After obtaining the dataset, run:

    python src/preprocessing.py

The preprocessing pipeline performs:

- Dataset loading
- Sample-count validation
- Missing-value checks
- Pixel-range validation
- Label validation
- Class-distribution checks
- Duplicate-image detection
- Train/test leakage detection
- Train/validation splitting
- Image reshaping
- Pixel normalization

## Large File Policy

The complete CIFAR-10 archive and extracted dataset files
must not be committed to Git.

The repository `.gitignore` is configured to prevent the
full raw dataset from being added accidentally.

Only the lightweight sample file and dataset download
script are maintained in Git.

## Reproducibility

A new team member can reproduce the dataset setup by
running:

    python data/raw/download_data.py

followed by:

    python src/preprocessing.py

The preprocessing pipeline uses a fixed random state
for reproducible train/validation splitting.