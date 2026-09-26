"""
SentriVision AI - ML Engineer Inference Pipeline

Role:
    ML Engineer

Purpose:
    Reproducible inference pipeline for SentriVision AI.

Features:
    1. Loads trained CNN / ResNet18 weights
    2. Loads a CIFAR-10 test image
    3. Preprocesses the image consistently with training
    4. Runs inference using BOTH trained models
    5. Measures inference latency
    6. Reports prediction confidence
    7. Logs predictions to CSV
"""

from pathlib import Path
import time
import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from src.model import (
    build_baseline_cnn,
    ResNet18
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODELS_DIR = PROJECT_ROOT / "models"

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PREDICTION_LOG = (
    PROCESSED_DIR / "prediction_log.csv"
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
# CIFAR-10 NORMALIZATION
# Same values used during model training
# ============================================================

MEAN = tf.constant(
    [0.4914, 0.4822, 0.4465],
    dtype=tf.float32
)

STD = tf.constant(
    [0.2470, 0.2435, 0.2616],
    dtype=tf.float32
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(
    "SecureVision"
)


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(model_name):
    """
    Load a trained SentriVision model.

    Supported models:
        baseline_cnn
        resnet18
    """

    if model_name == "baseline_cnn":

        model = build_baseline_cnn()

    elif model_name == "resnet18":

        model = ResNet18()

    else:

        raise ValueError(
            "Unknown model. Choose "
            "'baseline_cnn' or 'resnet18'."
        )

    weights_path = (
        MODELS_DIR /
        f"{model_name}.weights.h5"
    )

    if not weights_path.exists():

        raise FileNotFoundError(
            f"Model weights not found:\n"
            f"{weights_path}"
        )

    # Build model variables before loading weights
    model(
        tf.zeros((1, 32, 32, 3)),
        training=False
    )

    model.load_weights(
        weights_path
    )

    logger.info(
        "Loaded model: %s",
        weights_path
    )

    return model


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    """
    Preprocess a single CIFAR-10 image.

    Input:
        NumPy array with shape (32, 32, 3)

    Pixel range:
        0-255 or 0-1

    Output:
        Tensor with shape (1, 32, 32, 3)
    """

    image = np.asarray(image)

    # Validate image shape
    if image.shape != (32, 32, 3):

        raise ValueError(
            f"Expected image shape "
            f"(32, 32, 3), "
            f"got {image.shape}"
        )

    # Convert to float32
    image = image.astype(
        np.float32
    )

    # Convert 0-255 → 0-1
    if image.max() > 1.0:

        image = image / 255.0

    # Convert NumPy → TensorFlow tensor
    image = tf.convert_to_tensor(
        image,
        dtype=tf.float32
    )

    # Standardize using CIFAR-10
    # mean and standard deviation
    image = (
        image - MEAN
    ) / STD

    # Add batch dimension
    image = tf.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# SINGLE MODEL PREDICTION
# ============================================================

def predict(
    model,
    image,
    true_label=None
):
    """
    Run inference using one model.

    Returns:
        Dictionary containing prediction,
        confidence and latency.
    """

    processed_image = (
        preprocess_image(image)
    )

    # --------------------------------------------------------
    # Warm-up inference
    # --------------------------------------------------------
    # This prevents TensorFlow's first-call
    # initialization overhead from affecting
    # the measured latency.

    model(
        processed_image,
        training=False
    )

    # --------------------------------------------------------
    # Measure inference latency
    # --------------------------------------------------------

    start_time = time.perf_counter()

    logits = model(
        processed_image,
        training=False
    )

    end_time = time.perf_counter()

    latency_ms = (
        end_time - start_time
    ) * 1000

    # --------------------------------------------------------
    # Convert logits → probabilities
    # --------------------------------------------------------

    probabilities = tf.nn.softmax(
        logits,
        axis=1
    )

    probabilities = (
        probabilities
        .numpy()[0]
    )

    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = (
        CLASS_NAMES[
            predicted_index
        ]
    )

    confidence = float(
        probabilities[
            predicted_index
        ]
    )

    # --------------------------------------------------------
    # Store result
    # --------------------------------------------------------

    result = {

        "predicted_class":
            predicted_class,

        "predicted_index":
            predicted_index,

        "confidence":
            confidence,

        "latency_ms":
            latency_ms
    }

    # Add true label information
    # when available

    if true_label is not None:

        result["true_class"] = (
            CLASS_NAMES[
                true_label
            ]
        )

        result["correct"] = (
            predicted_index
            == true_label
        )

    return result


# ============================================================
# PREDICTION LOGGING
# ============================================================

def log_prediction(
    model_name,
    result
):
    """
    Append prediction result
    to prediction_log.csv.
    """

    record = {

        "model":
            model_name,

        "predicted_class":
            result[
                "predicted_class"
            ],

        "predicted_index":
            result[
                "predicted_index"
            ],

        "confidence":
            result[
                "confidence"
            ],

        "latency_ms":
            result[
                "latency_ms"
            ]
    }

    # Add true class information
    # when available

    if "true_class" in result:

        record["true_class"] = (
            result["true_class"]
        )

        record["correct"] = (
            result["correct"]
        )

    new_row = pd.DataFrame(
        [record]
    )

    # Append if log already exists
    if PREDICTION_LOG.exists():

        new_row.to_csv(
            PREDICTION_LOG,
            mode="a",
            header=False,
            index=False
        )

    # Otherwise create new log
    else:

        new_row.to_csv(
            PREDICTION_LOG,
            index=False
        )

    logger.info(
        "Prediction logged to: %s",
        PREDICTION_LOG
    )


# ============================================================
# MAIN INFERENCE PIPELINE
# ============================================================

def main():

    print("=" * 70)
    print(
        "SentriVision AI - ML Engineer Inference"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # Models to evaluate
    # --------------------------------------------------------

    model_names = [
        "baseline_cnn",
        "resnet18"
    ]

    # --------------------------------------------------------
    # Load CIFAR-10 test data
    # --------------------------------------------------------

    print(
        "\nLoading a CIFAR-10 test image..."
    )

    from src.preprocessing import (
        load_test_data
    )

    X_test, y_test = (
        load_test_data()
    )

    # --------------------------------------------------------
    # Select one test image
    # --------------------------------------------------------

    sample_index = 0

    # CIFAR-10 raw test data is stored
    # as a flattened 3072-element vector.
    #
    # Convert:
    # (3072,)
    #
    # into:
    # (32, 32, 3)

    image = (
        X_test[
            sample_index
        ].reshape(
            32,
            32,
            3
        )
    )

    true_label = int(
        y_test[
            sample_index
        ]
    )

    print(
        "Test image index :",
        sample_index
    )

    print(
        "True class       :",
        CLASS_NAMES[
            true_label
        ]
    )

    # --------------------------------------------------------
    # Run BOTH models
    # --------------------------------------------------------

    for model_name in model_names:

        print("\n")
        print("=" * 60)
        print(
            f"Running model: {model_name}"
        )
        print("=" * 60)

        # Load model
        model = load_model(
            model_name
        )

        # Run prediction
        result = predict(
            model,
            image,
            true_label
        )

        # ----------------------------------------------------
        # Display prediction
        # ----------------------------------------------------

        print(
            "\nPrediction Result"
        )

        print("-" * 40)

        print(
            "Model           :",
            model_name
        )

        print(
            "True class      :",
            result[
                "true_class"
            ]
        )

        print(
            "Predicted class :",
            result[
                "predicted_class"
            ]
        )

        print(
            "Confidence      :",
            f"{result['confidence']:.4f}"
        )

        print(
            "Latency         :",
            f"{result['latency_ms']:.2f} ms"
        )

        print(
            "Correct         :",
            result[
                "correct"
            ]
        )

        # ----------------------------------------------------
        # Log prediction
        # ----------------------------------------------------

        log_prediction(
            model_name,
            result
        )

    # --------------------------------------------------------
    # Completion message
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print(
        "Both models completed inference successfully."
    )
    print("=" * 70)

    print(
        f"\nPrediction log:"
        f"\n{PREDICTION_LOG}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()