# Secure Vision AI

> AI-powered computer vision system for 10-class image classification using a baseline CNN and a custom ResNet18 architecture.

## Overview

**Secure Vision AI** is a team-based machine learning project built around image classification with **CIFAR-10**. The project focuses on developing a reproducible data pipeline, training multiple CNN architectures, evaluating model performance, and exposing the trained models through a Streamlit dashboard.

The current project contains:

- Data validation and preprocessing
- Reproducible train/validation/test preparation
- Data-quality and leakage checks
- A baseline CNN model
- A CIFAR-10-adapted ResNet18 model
- Model training and weight saving
- Prediction/inference utilities
- A Streamlit dashboard
- Notebooks, reports, documentation, and tests
- Performance, cost, and threshold-analysis artifacts

## Dataset

The project uses **CIFAR-10**, containing 60,000 RGB images across 10 classes.

| Class | Description |
|---|---|
| 0 | airplane |
| 1 | automobile |
| 2 | bird |
| 3 | cat |
| 4 | deer |
| 5 | dog |
| 6 | frog |
| 7 | horse |
| 8 | ship |
| 9 | truck |

Each image is **32 × 32 pixels with 3 RGB channels**.

The preprocessing pipeline validates the dataset, checks class distribution, checks for exact duplicate images, checks for exact train/test overlap, creates a reproducible stratified 90/10 train/validation split, and normalizes image pixels to `[0, 1]`.

> **Important:** `src/preprocessing.py` expects the raw CIFAR-10 batch files in a sibling directory named `cifar-10-data/cifar-10-batches-py/` relative to the project directory. The expected files are `data_batch_1` through `data_batch_5` and `test_batch`.

## Project Structure

```text
secure-vision-ai/
│
├── dashboard/                  # Dashboard-related resources
├── data/                       # Processed data and analysis artifacts
├── docs/                       # Project documentation
├── models/                     # Saved model weights
├── notebooks/                  # Exploratory analysis and experiments
├── presentation/               # Presentation material
├── report/                     # Project report material
├── src/                        # Core data, model, and prediction code
│   ├── preprocessing.py        # Data loading, validation and preprocessing
│   ├── model.py                # CNN and ResNet18 training
│   └── predict.py              # Image preprocessing/inference utilities
│
├── tests/                      # Project tests
├── app.py                      # Streamlit application
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Machine Learning Pipeline

```text
CIFAR-10 Raw Data
       │
       ▼
Data Loading
       │
       ▼
Data Validation
       │
       ├── Shape / range checks
       ├── Label validation
       ├── Class distribution
       ├── Duplicate detection
       └── Train/Test leakage check
       │
       ▼
Train / Validation Split
       │
       ▼
Reshape + Normalize
       │
       ▼
Data Augmentation
       │
       ├───────────────┐
       ▼               ▼
Baseline CNN       ResNet18
       │               │
       └───────┬───────┘
               ▼
        Saved Model Weights
               │
               ▼
        Prediction / Dashboard
```

## Models

### 1. Baseline CNN

The baseline model is a compact three-block convolutional neural network:

- Conv2D: 32 filters
- MaxPooling
- Conv2D: 64 filters
- MaxPooling
- Conv2D: 128 filters
- MaxPooling
- Flatten
- Dense: 256 units
- Output: 10 classes

It provides a simple reference point against which the deeper residual model can be compared.

### 2. ResNet18

The project also implements a CIFAR-10-adapted **ResNet18** using TensorFlow/Keras.

The implementation uses:

- A 3×3 convolutional stem
- Residual `BasicBlock`s
- Four residual stages
- Global average pooling
- A 10-class output layer

For CIFAR-10, the architecture uses a 32×32 input-compatible stem rather than the standard ImageNet-style large-image stem.

### Training Configuration

The current training module uses:

```text
Epochs:          15
Batch size:      128
Learning rate:   0.001
Random seed:     42
Optimizer:       defined in the training module
Input size:      32 × 32 × 3
Output classes:  10
```

Training uses data augmentation equivalent to random cropping with padding and random horizontal flipping, followed by CIFAR-10 channel standardization.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Preetham-1064/secure-vision-ai.git
cd secure-vision-ai
```

### 2. Create a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The repository currently pins the main scientific/ML dependencies, including NumPy, pandas, Matplotlib, scikit-learn, TensorFlow, Jupyter, Pillow, and Streamlit.

## Data Setup

Before running the preprocessing/training pipeline, make sure the expected CIFAR-10 files exist at:

```text
../cifar-10-data/cifar-10-batches-py/
```

relative to the repository root.

Expected structure:

```text
parent-directory/
│
├── secure-vision-ai/
│
└── cifar-10-data/
    └── cifar-10-batches-py/
        ├── data_batch_1
        ├── data_batch_2
        ├── data_batch_3
        ├── data_batch_4
        ├── data_batch_5
        └── test_batch
```

This path is defined directly in `src/preprocessing.py`. If your raw dataset is stored somewhere else, update the dataset path in that file rather than downloading another copy unnecessarily.

## Running the Training Pipeline

From the **repository root**, run:

```bash
python -m src.model
```

The training module loads the prepared data through the preprocessing pipeline and trains:

1. `baseline_cnn`
2. `resnet18`

The trained weights are saved in the `models/` directory as:

```text
models/baseline_cnn.weights.h5
models/resnet18.weights.h5
```

Running from the repository root is recommended because the project uses package imports such as:

```python
from src.preprocessing import ...
```

## Running the Dashboard

The project includes a Streamlit application in `app.py`.

Run:

```bash
streamlit run app.py
```

The dashboard loads the project models and provides an interface for interacting with the classification system and project analysis artifacts.

## Reproducibility

The project is designed to keep the ML workflow reproducible.

Key controls include:

- Fixed random state: `42`
- Stratified train/validation split
- Test data kept separate from the train/validation split
- Dataset validation before training
- Exact duplicate detection using image hashes
- Exact train/test leakage checking
- Saved preprocessing configuration
- Saved dataset split indices
- Version-pinned core Python dependencies

## Team Responsibilities

| Student ID | Team Member | Role | Primary Responsibility |
|---|---|---|---|
| 241BCADA54 | Reyaan | Data Engineer | Dataset ingestion, validation, preprocessing pipeline, data quality and leakage checks |
| 241BCADA66 | Prithvi | Data Analyst | Exploratory/statistical analysis, distributions, trends, and analytical insights |
| 241BCADA03 | Tarun | Data Scientist | Model development, CNN/ResNet experimentation, training strategy, and ML analysis |
| 241BCADA72 | Preetham M | ML Engineer | Model engineering, training integration, model artifacts, and deployment/inference integration |
| 241BCADA16 | Jason | Analytics Engineer | Analytical pipelines, metrics integration, data transformation, and analytics support |
| 241BCADA29 | Tusharr | BI Developer | Dashboard, visualization, reporting interface, and business-facing presentation of results |

## Role Boundaries

To keep the project organized:

### Data Engineer — Reyaan

Responsible for making sure the data entering the ML pipeline is correct, reproducible, and usable.

Key areas:

- Raw dataset loading
- Data validation
- Data quality checks
- Duplicate detection
- Leakage detection
- Train/validation/test preparation
- Preprocessing metadata

### Data Analyst — Prithvi

Responsible for understanding the dataset and extracting statistical insights.

Key areas:

- Exploratory data analysis
- Class distributions
- Image-level statistics
- RGB/channel analysis
- Brightness and contrast analysis
- Distribution and skewness analysis
- Anomaly analysis
- Analytical reporting

### Data Scientist — Tarun

Responsible for developing and analysing the machine learning models.

Key areas:

- Model architecture
- Baseline CNN
- ResNet18
- Training experiments
- Hyperparameter analysis
- Model comparison
- ML interpretation
- Model-related conclusions

### ML Engineer — Preetham M

Responsible for integrating the ML components into a reliable engineering workflow.

Key areas:

- Training pipeline integration
- Model artifact management
- Inference integration
- Model reproducibility
- ML system engineering

### Analytics Engineer — Jason

Responsible for connecting analytical outputs into reusable data/metrics workflows.

Key areas:

- Metric pipelines
- Data transformations
- Analytical datasets
- Model-output analysis
- Analytics integration

### BI Developer — Tusharr

Responsible for communicating results through the project dashboard and reporting layer.

Key areas:

- Streamlit dashboard
- Visualizations
- KPI presentation
- Model/result reporting
- User-facing analytics

## Outputs

Depending on which components have been executed, the project can generate artifacts such as:

```text
data/processed/
├── data_quality_report.csv
├── preprocessing_config.json
├── dataset_split_indices.csv
├── metrics/
├── cost_analysis/
└── threshold_tuning/

models/
├── baseline_cnn.weights.h5
└── resnet18.weights.h5
```

Additional notebooks, reports, presentations, and dashboard assets are stored in their respective project directories.

## Testing

The repository contains a `tests/` directory for project tests.

Run the test suite with:

```bash
pytest
```

If `pytest` is not installed in the current environment:

```bash
pip install pytest
```

## Common Issues

### `ModuleNotFoundError: No module named 'src'`

Run the model from the **repository root**:

```bash
cd secure-vision-ai
python -m src.model
```

Avoid running `model.py` directly from inside the `src/` directory.

### Missing CIFAR-10 batch

If you see an error such as:

```text
Missing CIFAR-10 batch: ...
```

check that the raw CIFAR-10 files are located at:

```text
../cifar-10-data/cifar-10-batches-py/
```

and that the filenames match the expected CIFAR-10 batch names.

### Streamlit cannot import project modules

Make sure you launch Streamlit from the repository root:

```bash
streamlit run app.py
```

and that the virtual environment containing the project's dependencies is active.

## Technology Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Pillow**
- **Jupyter**
- **Streamlit**

## Project Goals

The project aims to demonstrate an end-to-end machine learning workflow:

1. Acquire and validate data
2. Perform exploratory analysis
3. Build a reproducible preprocessing pipeline
4. Train a baseline CNN
5. Train a ResNet18 model
6. Evaluate and compare model behaviour
7. Analyse performance and inference considerations
8. Present results through a dashboard and project documentation

## Limitations

This project is primarily an academic/educational machine learning system.

CIFAR-10 consists of small 32×32 images, so performance on CIFAR-10 should not be interpreted as equivalent to performance on real-world high-resolution surveillance imagery.

The system should therefore be treated as a computer-vision classification project and experimental prototype rather than a production security or surveillance solution.

## Contributors

- **Reyaan** — Data Engineer
- **Prithvi** — Data Analyst
- **Tarun** — Data Scientist
- **Preetham M** — ML Engineer
- **Jason** — Analytics Engineer
- **Tusharr** — BI Developer

---

## Quick Start

```bash
git clone https://github.com/Preetham-1064/secure-vision-ai.git
cd secure-vision-ai

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
# .venv\Scripts\activate

pip install -r requirements.txt

# Train models
python -m src.model

# Launch dashboard
streamlit run app.py
```

