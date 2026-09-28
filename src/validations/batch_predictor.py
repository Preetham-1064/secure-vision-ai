"""
Secure Vision AI - Batch Prediction Generator

Role:
    Analytics Engineer (supporting script)

Purpose:
    Run BOTH trained models over the whole validation and test splits
    and save one row per image, including the confidence score.
    threshold_tuning.py reads this file.

Why a separate file:
    predict.py logs a single image per run, so prediction_log.csv is far
    too small to tune a threshold. This script does not change
    predict.py or prediction_log.csv. It reuses predict.load_model and the
    same prepare_ml_data() split and standardization used for training.

Run from the project root:
    python -m src.validations.batch_predictor

Output:
    data/processed/batch_predictions.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.model import MEAN, STD
from src.predict import CLASS_NAMES, load_model
from src.preprocessing import (
    load_training_data,
    load_test_data,
    prepare_ml_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "batch_predictions.csv"

MODEL_NAMES = ("baseline_cnn", "resnet18")

BATCH_SIZE = 256


def load_splits():
    """Return {'validation': (X, y), 'test': (X, y)} from prepare_ml_data."""

    X_train_raw, y_train_raw = load_training_data()
    X_test_raw, y_test_raw = load_test_data()

    (
        X_train, X_validation, X_test,
        y_train, y_validation, y_test,
    ) = prepare_ml_data(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw
    )

    splits = {}

    for name, images, labels in (
        ("validation", X_validation, y_validation),
        ("test", X_test, y_test),
    ):
        images = np.asarray(images, dtype=np.float32)
        labels = np.asarray(labels).reshape(-1).astype(int)

        if images.ndim != 4 or images.shape[1:] != (32, 32, 3):
            raise ValueError(
                f"Expected {name} images with shape (N, 32, 32, 3), "
                f"got {images.shape}."
            )

        if len(images) != len(labels):
            raise ValueError(
                f"{name}: {len(images)} images but {len(labels)} labels."
            )

        # Training used pixels in [0, 1] before standardization.
        if images.max() > 1.0:
            images = images / 255.0

        splits[name] = (images, labels)

    return splits


def predict_probabilities(model, images):
    """Softmax probabilities for every image, computed in batches."""

    outputs = []

    for start in range(0, len(images), BATCH_SIZE):
        batch = tf.convert_to_tensor(
            images[start:start + BATCH_SIZE], dtype=tf.float32
        )

        # Same standardization as training and predict.py.
        batch = (batch - MEAN) / STD

        logits = model(batch, training=False)

        outputs.append(tf.nn.softmax(logits, axis=1).numpy())

    return np.vstack(outputs)


def build_records(model_name, split_name, probabilities, labels):
    """One row per image."""

    predicted = probabilities.argmax(axis=1)

    frame = pd.DataFrame({
        "model": model_name,
        "split": split_name,
        "sample_index": np.arange(len(labels)),
        "true_index": labels,
        "true_class": [CLASS_NAMES[i] for i in labels],
        "predicted_index": predicted,
        "predicted_class": [CLASS_NAMES[i] for i in predicted],
        "confidence": probabilities.max(axis=1),
        "correct": predicted == labels,
    })

    for index, class_name in enumerate(CLASS_NAMES):
        frame[f"prob_{class_name}"] = probabilities[:, index]

    return frame


def main():

    print("=" * 60)
    print("Secure Vision AI - Batch Prediction Generator")
    print("=" * 60)

    print("\nLoading validation and test splits...")
    splits = load_splits()

    for name, (images, _) in splits.items():
        print(f"  {name:<10}: {len(images)} images")

    frames = []

    for model_name in MODEL_NAMES:

        print(f"\nLoading model: {model_name}")
        model = load_model(model_name)

        for split_name, (images, labels) in splits.items():

            print(f"  Predicting on {split_name}...")

            probabilities = predict_probabilities(model, images)

            frame = build_records(
                model_name, split_name, probabilities, labels
            )

            accuracy = frame["correct"].mean()

            print(
                f"  {model_name} / {split_name}: "
                f"accuracy {accuracy:.4f}"
            )

            frames.append(frame)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "=" * 60)
    print(f"Saved {len(result)} rows to:")
    print(OUTPUT_PATH)
    print("=" * 60)
    print("\nNext: python -m src.validations.batch_predictor")


if __name__ == "__main__":
    main()