"""
SentriVision AI - Inference Latency Benchmark

Purpose:
    Measure single-image inference latency for the trained models.

Models:
    - Baseline CNN
    - ResNet18

Dataset:
    CIFAR-10 held-out test split

Benchmark:
    - 500 test images per model
    - Batch size = 1
    - Warm-up runs excluded
    - Latency measured only for model inference
    - TensorFlow output is converted to NumPy to ensure
      the operation has completed before timing is recorded.

Output:
    data/processed/metrics/inference_latency.csv

Run from project root:
    python -m src.validations.latency_benchmark
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from src.preprocessing import load_test_data
from src.predict import load_model


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "metrics"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "inference_latency.csv"
)

# Number of held-out test images to benchmark
NUM_SAMPLES = 500

# Warm-up runs are excluded from latency statistics
WARMUP_RUNS = 20

MODEL_NAMES = (
    "baseline_cnn",
    "resnet18",
)


# ============================================================
# Preprocessing
# ============================================================

CIFAR_MEAN = np.array(
    [0.4914, 0.4822, 0.4465],
    dtype=np.float32,
)

CIFAR_STD = np.array(
    [0.2470, 0.2435, 0.2616],
    dtype=np.float32,
)


def preprocess_for_inference(image):
    """
    Apply the same standardization used during model training.

    Input:
        image -> uint8 array with shape (32, 32, 3)

    Output:
        float32 tensor with shape (1, 32, 32, 3)
    """

    image = image.astype(
        np.float32
    ) / 255.0

    image = (
        image - CIFAR_MEAN
    ) / CIFAR_STD

    return np.expand_dims(
        image,
        axis=0,
    ).astype(
        np.float32
    )


# ============================================================
# Latency Measurement
# ============================================================

def benchmark_model(
    model_name,
    test_images,
):
    """
    Measure single-image inference latency.
    """

    print()
    print("=" * 60)
    print(f"Benchmarking: {model_name}")
    print("=" * 60)

    print(
        f"Loading model: {model_name}"
    )

    model = load_model(model_name)

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(
        f"Running {WARMUP_RUNS} warm-up inferences..."
    )

    warmup_image = preprocess_for_inference(
        test_images[0]
    )

    for _ in range(WARMUP_RUNS):

        output = model(
            warmup_image,
            training=False,
        )

        # Force TensorFlow to complete the operation
        _ = output.numpy()

    print("Warm-up complete.")

    # --------------------------------------------------------
    # Actual benchmark
    # --------------------------------------------------------

    print(
        f"Measuring {len(test_images)} "
        "single-image inferences..."
    )

    latencies_ms = []

    for index, image in enumerate(
        test_images
    ):

        processed_image = (
            preprocess_for_inference(
                image
            )
        )

        start_time = time.perf_counter()

        output = model(
            processed_image,
            training=False,
        )

        # Force computation to finish
        _ = output.numpy()

        end_time = time.perf_counter()

        latency_ms = (
            end_time - start_time
        ) * 1000

        latencies_ms.append(
            latency_ms
        )

        if (
            (index + 1) % 100 == 0
        ):
            print(
                f"  Completed "
                f"{index + 1}/"
                f"{len(test_images)}"
            )

    latencies_ms = np.array(
        latencies_ms,
        dtype=np.float64,
    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    average_latency = float(
        np.mean(latencies_ms)
    )

    median_latency = float(
        np.median(latencies_ms)
    )

    p95_latency = float(
        np.percentile(
            latencies_ms,
            95,
        )
    )

    minimum_latency = float(
        np.min(latencies_ms)
    )

    maximum_latency = float(
        np.max(latencies_ms)
    )

    throughput = (
        1000.0
        / average_latency
    )

    print()
    print(
        f"Average latency : "
        f"{average_latency:.3f} ms"
    )

    print(
        f"Median latency  : "
        f"{median_latency:.3f} ms"
    )

    print(
        f"P95 latency     : "
        f"{p95_latency:.3f} ms"
    )

    print(
        f"Minimum latency : "
        f"{minimum_latency:.3f} ms"
    )

    print(
        f"Maximum latency : "
        f"{maximum_latency:.3f} ms"
    )

    print(
        f"Throughput      : "
        f"{throughput:.2f} images/sec"
    )

    return {
        "model": model_name,
        "samples": len(test_images),
        "average_latency_ms": average_latency,
        "median_latency_ms": median_latency,
        "p95_latency_ms": p95_latency,
        "min_latency_ms": minimum_latency,
        "max_latency_ms": maximum_latency,
        "throughput_images_per_sec": throughput,
    }


# ============================================================
# Main
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SentriVision AI - Inference Latency Benchmark")
    print("=" * 60)

    print()
    print(
        f"Project root: {PROJECT_ROOT}"
    )

    print(
        f"Benchmark samples per model: "
        f"{NUM_SAMPLES}"
    )

    print(
        f"Warm-up runs: "
        f"{WARMUP_RUNS}"
    )

    # --------------------------------------------------------
    # Load CIFAR-10 test data
    # --------------------------------------------------------

    print()
    print("Loading CIFAR-10 test data...")

    _, test_labels = load_test_data()

    # Load images separately through the existing loader
    # and discard labels after verifying the test set size.
    #
    # load_test_data() returns the flattened raw images,
    # so reshape them into CIFAR-10 image format.

    test_images = _load_test_images()

    print(
        f"Available test images: "
        f"{len(test_images)}"
    )

    if len(test_images) < NUM_SAMPLES:

        raise ValueError(
            f"Only {len(test_images)} test images "
            f"are available, but {NUM_SAMPLES} "
            "were requested."
        )

    # --------------------------------------------------------
    # Select deterministic subset
    # --------------------------------------------------------

    benchmark_images = test_images[
        :NUM_SAMPLES
    ]

    print(
        f"Using first {len(benchmark_images)} "
        "held-out test images."
    )

    # --------------------------------------------------------
    # Benchmark models
    # --------------------------------------------------------

    results = []

    for model_name in MODEL_NAMES:

        result = benchmark_model(
            model_name,
            benchmark_images,
        )

        results.append(result)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print()
    print("=" * 60)
    print("LATENCY BENCHMARK COMPLETED")
    print("=" * 60)

    print(
        f"Saved results to:"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print(results_df.to_string(
        index=False
    ))

    print("=" * 60)


# ============================================================
# Helper: Load raw CIFAR-10 test images
# ============================================================

def _load_test_images():

    """
    Load raw CIFAR-10 test images using the same
    project dataset location as preprocessing.py.

    Returns:
        numpy array with shape:
        (10000, 32, 32, 3)
    """

    data_root = (
        PROJECT_ROOT.parent
        / "cifar-10-data"
        / "cifar-10-batches-py"
    )

    test_file = (
        data_root
        / "test_batch"
    )

    if not test_file.exists():

        raise FileNotFoundError(
            "CIFAR-10 test_batch not found at: "
            f"{test_file}"
        )

    import pickle

    with open(
        test_file,
        "rb",
    ) as file:

        batch = pickle.load(
            file,
            encoding="bytes",
        )

    images = batch[
        b"data"
    ]

    images = images.reshape(
        -1,
        3,
        32,
        32,
    )

    images = np.transpose(
        images,
        (0, 2, 3, 1),
    )

    return images.astype(
        np.uint8
    )


if __name__ == "__main__":
    main()