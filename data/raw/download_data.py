"""
Downloads and prepares the CIFAR-10 dataset for SecureVision AI.

The complete dataset is intentionally not stored in GitHub
because it exceeds the project's large-file limit.
"""

"""
SecureVision AI - CIFAR-10 Dataset Downloader

Downloads the CIFAR-10 dataset from Google Drive and extracts it
outside the GitHub repository.
"""

from pathlib import Path
import tarfile
import gdown


# ==========================================================
# CONFIGURATION
# ==========================================================

FILE_ID = "1jg6CM6xILVykYT4yx16Wjui21iDJVpfL"

FILE_NAME = "cifar-10-python.tar.gz"


# ==========================================================
# PATHS
# ==========================================================

# Project root:
# secure-vision-ai/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Store the actual dataset OUTSIDE the GitHub repository
DATASET_DIR = PROJECT_ROOT.parent / "cifar-10-data"

ARCHIVE_PATH = DATASET_DIR / FILE_NAME

EXTRACTED_DIR = DATASET_DIR / "cifar-10-batches-py"


# ==========================================================
# DOWNLOAD
# ==========================================================

def download_dataset():
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if ARCHIVE_PATH.exists():
        print("Dataset archive already exists.")
        return

    print("Downloading CIFAR-10 dataset from Google Drive...")
    print(f"Saving to: {ARCHIVE_PATH}")

    url = f"https://drive.google.com/uc?id={FILE_ID}"

    gdown.download(
        url,
        str(ARCHIVE_PATH),
        quiet=False
    )

    print("Download completed.")


# ==========================================================
# EXTRACTION
# ==========================================================

def extract_dataset():
    if EXTRACTED_DIR.exists():
        print("Dataset is already extracted.")
        return

    print("Extracting CIFAR-10 dataset...")

    with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
        tar.extractall(DATASET_DIR)

    print("Extraction completed.")
    print(f"Dataset location: {EXTRACTED_DIR}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("=" * 60)
    print("SecureVision AI - CIFAR-10 Dataset Setup")
    print("=" * 60)

    download_dataset()
    extract_dataset()

    print("\nDataset setup completed successfully!")
    print(f"Dataset location: {EXTRACTED_DIR}")


if __name__ == "__main__":
    main()