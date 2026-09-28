"""
Secure Vision AI - Classification Cost Matrix Module

Role:
    Analytics Engineer

Purpose:
    Quantify the cost assigned to classification outcomes
    using the complete batch prediction dataset.

Input:
    data/processed/batch_predictions.csv

Required columns:
    model
    split
    true_index
    predicted_index

Default cost model:
    Correct prediction = 0
    Incorrect prediction = 1

These are illustrative unit costs, NOT measured financial costs.

Outputs:
    data/processed/cost_analysis/

    <model>_<split>_cost_summary.csv
    <model>_<split>_cost_matrix.csv
    <model>_<split>_confusion_matrix.csv
    <model>_<split>_cost_by_class.csv
"""


from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import confusion_matrix


# ==========================================================
# PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_PATH = (
    PROCESSED_DIR
    / "batch_predictions.csv"
)

COST_DIR = (
    PROCESSED_DIR
    / "cost_analysis"
)


# ==========================================================
# CLASSES
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

NUM_CLASSES = len(CLASS_NAMES)

MODEL_NAMES = (
    "baseline_cnn",
    "resnet18",
)

SPLITS = (
    "validation",
    "test",
)


# ==========================================================
# COST CONFIGURATION
# ==========================================================

DEFAULT_INCORRECT_COST = 1.0


# ==========================================================
# COST MATRIX
# ==========================================================

def create_cost_matrix(
    incorrect_prediction_cost=DEFAULT_INCORRECT_COST
):
    """
    Create a 10x10 classification cost matrix.

    Correct predictions:
        cost = 0

    Incorrect predictions:
        cost = incorrect_prediction_cost
    """

    if not np.isscalar(
        incorrect_prediction_cost
    ):
        raise ValueError(
            "Incorrect prediction cost must be a number."
        )

    if (
        not np.isfinite(
            incorrect_prediction_cost
        )
        or incorrect_prediction_cost < 0
    ):
        raise ValueError(
            "Cost must be finite and non-negative."
        )

    matrix = np.full(
        (
            NUM_CLASSES,
            NUM_CLASSES,
        ),
        float(
            incorrect_prediction_cost
        ),
    )

    np.fill_diagonal(
        matrix,
        0.0,
    )

    return matrix


# ==========================================================
# LABEL VALIDATION
# ==========================================================

def validate_labels(
    y_true,
    y_pred,
):
    """
    Validate actual and predicted class indices.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if (
        y_true.ndim != 1
        or y_pred.ndim != 1
    ):
        raise ValueError(
            "Labels must be one-dimensional."
        )

    if (
        len(y_true) == 0
        or len(y_true) != len(y_pred)
    ):
        raise ValueError(
            "Actual and predicted labels must have "
            "equal, non-zero lengths."
        )

    if (
        not np.all(
            np.isfinite(y_true)
        )
        or not np.all(
            np.isfinite(y_pred)
        )
    ):
        raise ValueError(
            "Labels must contain finite values."
        )

    if not np.all(
        y_true == np.floor(y_true)
    ):
        raise ValueError(
            "Actual labels must be integers."
        )

    if not np.all(
        y_pred == np.floor(y_pred)
    ):
        raise ValueError(
            "Predicted labels must be integers."
        )

    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    if (
        np.any(y_true < 0)
        or np.any(
            y_true >= NUM_CLASSES
        )
        or np.any(y_pred < 0)
        or np.any(
            y_pred >= NUM_CLASSES
        )
    ):
        raise ValueError(
            "Class indices must be between 0 and 9."
        )

    return y_true, y_pred


# ==========================================================
# CALCULATE COSTS
# ==========================================================

def calculate_costs(
    y_true,
    y_pred,
    cost_matrix=None,
):
    """
    Calculate:

        total cost
        average cost
        incorrect predictions
        confusion matrix
        cost by actual class
    """

    y_true, y_pred = validate_labels(
        y_true,
        y_pred,
    )

    if cost_matrix is None:

        cost_matrix = create_cost_matrix()

    cost_matrix = np.asarray(
        cost_matrix,
        dtype=float,
    )

    if cost_matrix.shape != (
        NUM_CLASSES,
        NUM_CLASSES,
    ):
        raise ValueError(
            "Cost matrix must have shape (10, 10)."
        )

    if not np.all(
        np.isfinite(cost_matrix)
    ):
        raise ValueError(
            "Cost matrix must contain finite values."
        )

    if np.any(
        cost_matrix < 0
    ):
        raise ValueError(
            "Cost matrix cannot contain negative costs."
        )

    if not np.allclose(
        np.diag(cost_matrix),
        0,
    ):
        raise ValueError(
            "Correct predictions must have zero cost."
        )

    # ------------------------------------------------------
    # Cost for each individual prediction
    # ------------------------------------------------------

    individual_costs = (
        cost_matrix[
            y_true,
            y_pred
        ]
    )

    total_cost = float(
        np.sum(individual_costs)
    )

    average_cost = float(
        np.mean(individual_costs)
    )

    incorrect = int(
        np.sum(
            y_true != y_pred
        )
    )

    # ------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------

    conf_matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=list(
            range(NUM_CLASSES)
        ),
    )

    # ------------------------------------------------------
    # Cost by actual class
    # ------------------------------------------------------

    class_cost_rows = []

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        mask = y_true == index

        count = int(
            np.sum(mask)
        )

        class_total = float(
            np.sum(
                individual_costs[mask]
            )
        )

        class_cost_rows.append({

            "class_index": index,

            "class_name": class_name,

            "sample_count": count,

            "total_cost": class_total,

            "average_cost": (
                class_total / count
                if count
                else 0.0
            ),

        })

    return {

        "sample_count": len(y_true),

        "incorrect_predictions": incorrect,

        "total_cost": total_cost,

        "average_cost_per_sample":
            average_cost,

        "confusion_matrix":
            conf_matrix,

        "cost_matrix":
            cost_matrix,

        "cost_by_class":
            pd.DataFrame(
                class_cost_rows
            ),

    }


# ==========================================================
# LOAD BATCH PREDICTIONS
# ==========================================================

def load_batch_predictions(
    csv_path
):
    """
    Load and validate batch_predictions.csv.
    """

    csv_path = Path(
        csv_path
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Batch prediction file not found:\n"
            f"{csv_path}\n\n"
            "Run first:\n"
            "python -m src.validations.batch_predictor"
        )

    df = pd.read_csv(
        csv_path
    )

    required_columns = {
        "model",
        "split",
        "true_index",
        "predicted_index",
    }

    missing_columns = (
        required_columns
        - set(df.columns)
    )

    if missing_columns:

        raise ValueError(
            "batch_predictions.csv is missing "
            "required columns: "
            + ", ".join(
                sorted(missing_columns)
            )
        )

    df = df.dropna(
        subset=[
            "model",
            "split",
            "true_index",
            "predicted_index",
        ]
    ).copy()

    if df.empty:

        raise ValueError(
            "batch_predictions.csv contains "
            "no usable rows."
        )

    # ------------------------------------------------------
    # Convert labels
    # ------------------------------------------------------

    for column in (
        "true_index",
        "predicted_index",
    ):

        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

        values = df[column].to_numpy()

        if not np.all(
            np.isfinite(values)
        ):
            raise ValueError(
                f"{column} contains "
                "NaN or infinite values."
            )

        if not np.all(
            values == np.floor(values)
        ):
            raise ValueError(
                f"{column} must contain "
                "whole-number labels."
            )

        if (
            np.any(values < 0)
            or np.any(
                values >= NUM_CLASSES
            )
        ):
            raise ValueError(
                f"{column} must contain "
                "labels from 0 to "
                f"{NUM_CLASSES - 1}."
            )

        df[column] = values.astype(int)

    return df.reset_index(
        drop=True
    )


# ==========================================================
# EVALUATE ONE MODEL / SPLIT
# ==========================================================

def evaluate_model_split(
    df,
    model_name,
    split_name,
    cost_matrix=None,
):
    """
    Evaluate classification cost for one
    model on one dataset split.
    """

    subset = df[
        (df["model"] == model_name)
        & (df["split"] == split_name)
    ].copy()

    if subset.empty:

        raise ValueError(
            f"No predictions found for "
            f"model='{model_name}', "
            f"split='{split_name}'."
        )

    y_true = subset[
        "true_index"
    ].to_numpy()

    y_pred = subset[
        "predicted_index"
    ].to_numpy()

    results = calculate_costs(
        y_true,
        y_pred,
        cost_matrix,
    )

    # ------------------------------------------------------
    # Display
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "CLASSIFICATION COST ANALYSIS"
    )
    print("=" * 60)

    print(
        f"Model                    : "
        f"{model_name}"
    )

    print(
        f"Split                    : "
        f"{split_name}"
    )

    print(
        f"Samples evaluated        : "
        f"{results['sample_count']}"
    )

    print(
        f"Incorrect predictions    : "
        f"{results['incorrect_predictions']}"
    )

    print(
        f"Total classification cost: "
        f"{results['total_cost']:.2f}"
    )

    print(
        "Average cost per sample : "
        f"{results['average_cost_per_sample']:.4f}"
    )

    # ------------------------------------------------------
    # Cost matrix
    # ------------------------------------------------------

    print(
        "\nCost Matrix "
        "(rows = actual, columns = predicted):"
    )

    print(
        pd.DataFrame(
            results["cost_matrix"],
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_string()
    )

    # ------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        pd.DataFrame(
            results["confusion_matrix"],
            index=CLASS_NAMES,
            columns=CLASS_NAMES,
        ).to_string()
    )

    # ------------------------------------------------------
    # Cost by class
    # ------------------------------------------------------

    print("\nCost by Actual Class:")

    print(
        results["cost_by_class"]
        .round(4)
        .to_string(index=False)
    )

    # ------------------------------------------------------
    # Save reports
    # ------------------------------------------------------

    COST_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prefix = (
        f"{model_name}_{split_name}"
    )

    summary_path = (
        COST_DIR
        / f"{prefix}_cost_summary.csv"
    )

    cost_matrix_path = (
        COST_DIR
        / f"{prefix}_cost_matrix.csv"
    )

    confusion_matrix_path = (
        COST_DIR
        / f"{prefix}_confusion_matrix.csv"
    )

    cost_by_class_path = (
        COST_DIR
        / f"{prefix}_cost_by_class.csv"
    )

    # Summary

    pd.DataFrame([{

        "model":
            model_name,

        "split":
            split_name,

        "sample_count":
            results["sample_count"],

        "incorrect_predictions":
            results["incorrect_predictions"],

        "total_cost":
            results["total_cost"],

        "average_cost_per_sample":
            results[
                "average_cost_per_sample"
            ],

    }]).to_csv(
        summary_path,
        index=False,
    )

    # Cost matrix

    pd.DataFrame(
        results["cost_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        cost_matrix_path
    )

    # Confusion matrix

    pd.DataFrame(
        results["confusion_matrix"],
        index=CLASS_NAMES,
        columns=CLASS_NAMES,
    ).to_csv(
        confusion_matrix_path
    )

    # Cost by class

    results[
        "cost_by_class"
    ].to_csv(
        cost_by_class_path,
        index=False,
    )

    print(
        f"\nReports saved to:\n"
        f"{COST_DIR}"
    )

    return results


# ==========================================================
# MAIN PROGRAM
# ==========================================================

def main():

    print("=" * 60)
    print(
        "Secure Vision AI - "
        "Classification Cost Analysis"
    )
    print("=" * 60)

    print(
        f"\nInput file:\n"
        f"{INPUT_PATH}"
    )

    try:

        df = load_batch_predictions(
            INPUT_PATH
        )

    except (
        ValueError,
        FileNotFoundError,
    ) as error:

        print(
            f"\nError: {error}"
        )

        return

    print(
        f"\nLoaded {len(df):,} "
        "prediction rows."
    )

    print(
        "\nAvailable models: "
        + ", ".join(
            sorted(
                df["model"].unique()
            )
        )
    )

    print(
        "Available splits: "
        + ", ".join(
            sorted(
                df["split"].unique()
            )
        )
    )

    # Create the default cost matrix once.
    cost_matrix = create_cost_matrix()

    evaluated = 0

    for model_name in MODEL_NAMES:

        for split_name in SPLITS:

            print(
                f"\nEvaluating "
                f"{model_name} / "
                f"{split_name}..."
            )

            try:

                evaluate_model_split(
                    df=df,
                    model_name=model_name,
                    split_name=split_name,
                    cost_matrix=cost_matrix,
                )

                evaluated += 1

            except ValueError as error:

                print(
                    f"\nSkipping "
                    f"{model_name} / "
                    f"{split_name}: "
                    f"{error}"
                )

    print("\n" + "=" * 60)

    if evaluated:

        print(
            "Classification cost analysis "
            f"completed "
            f"({evaluated} model/split "
            "evaluations)."
        )

        print(
            f"Reports are stored in:\n"
            f"{COST_DIR}"
        )

    else:

        print(
            "No model/split combinations "
            "were evaluated."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()