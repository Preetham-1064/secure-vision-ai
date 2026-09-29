import base64
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

from src.model import build_baseline_cnn, ResNet18
from src.predict import preprocess_image


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "data" / "processed" / "metrics"
COST_DIR = PROJECT_ROOT / "data" / "processed" / "cost_analysis"
THRESHOLD_DIR = PROJECT_ROOT / "data" / "processed" / "threshold_tuning"

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

CIFAR_MEAN = np.array(
    [0.4914, 0.4822, 0.4465],
    dtype=np.float32,
)

CIFAR_STD = np.array(
    [0.2470, 0.2435, 0.2616],
    dtype=np.float32,
)


# ============================================================
# HTML Helper
# ============================================================

def clean_html(html: str) -> str:
    """Strip per-line indentation and blank lines so Markdown
    never mistakes indented HTML for a code block."""
    lines = (line.strip() for line in html.splitlines())
    return "\n".join(line for line in lines if line)


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="SecureVision AI",
    page_icon="◈",
    layout="wide",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    clean_html(
        """
        <style>

        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500;700&display=swap');

        :root {
            --bg: #0D1117;
            --panel: #151B23;
            --border: #212B36;
            --text-dim: #9FB4C7;
            --text: #EAF0F6;
            --signal: #5EEAD4;
            --signal-dim: rgba(94, 234, 212, 0.12);
            --alert: #F59E0B;
            --alert-dim: rgba(245, 158, 11, 0.12);
        }

        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }

        .stApp {
            background: var(--bg);
            color: var(--text);
        }

        #MainMenu, footer, header {
            visibility: hidden;
        }

        .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }

        .console-header {
            display: flex;
            align-items: baseline;
            gap: 14px;
            margin-bottom: 2px;
        }

        .console-mark {
            font-family: 'JetBrains Mono', monospace;
            color: var(--signal);
            font-size: 22px;
        }

        .console-title {
            font-size: 26px;
            font-weight: 600;
            letter-spacing: 0.2px;
            color: var(--text);
        }

        .console-subtitle {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-dim);
            margin: 4px 0 28px 0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 28px;
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            background: transparent;
            color: var(--text-dim);
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            letter-spacing: 0.3px;
            border-radius: 0;
            padding: 0 4px;
            margin-right: 24px;
        }

        .stTabs [aria-selected="true"] {
            color: var(--signal) !important;
            border-bottom: 2px solid var(--signal) !important;
            background: transparent !important;
        }

        .panel {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 20px 22px;
        }

        .panel-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 1.2px;
            color: var(--text-dim);
            margin-bottom: 14px;
        }

        .feed-frame {
            position: relative;
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px;
            background: #090D12;
            margin-bottom: 16px;
        }

        .feed-frame img {
            display: block;
            width: 100%;
            border-radius: 3px;
        }

        .feed-frame::before,
        .feed-frame::after,
        .feed-corner-tl,
        .feed-corner-br {
            content: "";
            position: absolute;
            width: 16px;
            height: 16px;
            border-color: var(--signal);
            border-style: solid;
        }

        .feed-frame::before {
            top: 6px;
            left: 6px;
            border-width: 2px 0 0 2px;
        }

        .feed-frame::after {
            bottom: 6px;
            right: 6px;
            border-width: 0 2px 2px 0;
        }

        .feed-corner-tl {
            top: 6px;
            right: 6px;
            border-width: 2px 2px 0 0;
        }

        .feed-corner-br {
            bottom: 6px;
            left: 6px;
            border-width: 0 0 2px 2px;
        }

        .feed-status {
            position: absolute;
            top: 22px;
            left: 22px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            letter-spacing: 1px;
            color: var(--signal);
            background: rgba(13, 17, 23, 0.75);
            padding: 3px 8px;
            border-radius: 3px;
        }

        .readout-empty {
            font-family: 'JetBrains Mono', monospace;
            font-size: 13px;
            color: var(--text-dim);
            padding: 40px 0;
            text-align: center;
            border: 1px dashed var(--border);
            border-radius: 6px;
        }

        .readout-class-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 11px;
            letter-spacing: 1.2px;
            color: var(--text-dim);
            margin-bottom: 6px;
        }

        .readout-class-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 34px;
            font-weight: 700;
            color: var(--signal);
            letter-spacing: 0.5px;
            margin-bottom: 18px;
        }

        .readout-metrics {
            display: flex;
            gap: 28px;
            margin-bottom: 22px;
            padding-bottom: 20px;
            border-bottom: 1px solid var(--border);
        }

        .readout-metric-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 10px;
            letter-spacing: 1px;
            color: var(--text-dim);
            margin-bottom: 4px;
        }

        .readout-metric-value {
            font-family: 'JetBrains Mono', monospace;
            font-size: 17px;
            font-weight: 500;
            color: var(--text);
        }

        .prob-row {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 7px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
        }

        .prob-name {
            width: 84px;
            color: var(--text-dim);
            flex-shrink: 0;
        }

        .prob-name.top {
            color: var(--text);
        }

        .prob-track {
            flex: 1;
            height: 6px;
            background: #0A0E14;
            border-radius: 3px;
            overflow: hidden;
        }

        .prob-fill {
            height: 100%;
            background: var(--signal);
            border-radius: 3px;
        }

        .prob-fill.low {
            background: #2A3644;
        }

        .prob-value {
            width: 46px;
            text-align: right;
            color: var(--text-dim);
            flex-shrink: 0;
        }

        .stSelectbox label,
        .stFileUploader label {
            font-family: 'JetBrains Mono', monospace !important;
            font-size: 11px !important;
            letter-spacing: 1px;
            color: var(--text-dim) !important;
        }

        .stSelectbox div[data-baseweb="select"] > div {
            background: var(--panel);
            border-color: var(--border);
            color: var(--text);
        }

        [data-testid="stFileUploaderDropzone"] {
            background: var(--panel);
            border: 1px dashed var(--border);
        }

        .stButton button {
            font-family: 'JetBrains Mono', monospace;
            font-size: 12px;
            letter-spacing: 1px;
            background: var(--signal-dim);
            color: var(--signal);
            border: 1px solid var(--signal);
            border-radius: 4px;
            padding: 10px 0;
        }

        .stButton button:hover {
            background: var(--signal);
            color: #04211C;
        }

        .metrics-section-title {
            font-family: 'JetBrains Mono', monospace;
            font-size: 14px;
            letter-spacing: 1px;
            color: var(--text);
            margin-top: 12px;
            margin-bottom: 14px;
        }

        [data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 14px;
        }

        [data-testid="stMetricLabel"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--text-dim) !important;
            font-size: 10px !important;
        }

        [data-testid="stMetricValue"] {
            font-family: 'JetBrains Mono', monospace !important;
            color: var(--signal) !important;
        }

        </style>
        """
    ),
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_model(model_name):

    if model_name == "Baseline CNN":

        model = build_baseline_cnn()
        weight_path = MODEL_DIR / "baseline_cnn.weights.h5"

    elif model_name == "ResNet18":

        model = ResNet18()
        weight_path = MODEL_DIR / "resnet18.weights.h5"

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weight_path}"
        )

    model(tf.zeros((1, 32, 32, 3)), training=False)
    model.load_weights(str(weight_path))

    return model


# ============================================================
# Prediction
# ============================================================

def predict_image(model, image):

    image = image.convert("RGB")
    image = image.resize((32, 32))

    image_array = np.array(
        image,
        dtype=np.uint8,
    )

    processed_image = preprocess_image(
        image_array
    )

    start_time = time.perf_counter()

    predictions = model.predict(
        processed_image,
        verbose=0,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    probabilities = tf.nn.softmax(
        predictions[0]
    ).numpy()

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities[predicted_index]
    )

    return (
        predicted_class,
        confidence,
        latency_ms,
        probabilities,
    )


# ============================================================
# Helpers
# ============================================================

def image_to_data_uri(image):

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode()

    return f"data:image/png;base64,{encoded}"


def render_feed_frame(image, status_text):

    data_uri = image_to_data_uri(image)

    html = f"""
    <div class="feed-frame">
        <div class="feed-status">{status_text}</div>
        <div class="feed-corner-tl"></div>
        <div class="feed-corner-br"></div>
        <img src="{data_uri}" />
    </div>
    """

    st.markdown(
        clean_html(html),
        unsafe_allow_html=True,
    )


def render_readout(
    predicted_class,
    confidence,
    latency_ms,
    model_name,
    probabilities,
):

    order = np.argsort(
        probabilities
    )[::-1]

    rows = ""

    for idx in order:

        pct = probabilities[idx] * 100
        is_top = idx == order[0]

        fill_class = (
            ""
            if is_top
            else "low"
        )

        name_class = (
            "top"
            if is_top
            else ""
        )

        rows += f"""
        <div class="prob-row">
            <div class="prob-name {name_class}">
                {CLASS_NAMES[idx]}
            </div>

            <div class="prob-track">
                <div
                    class="prob-fill {fill_class}"
                    style="width:{pct:.1f}%"
                ></div>
            </div>

            <div class="prob-value">
                {pct:.1f}%
            </div>
        </div>
        """

    html = f"""
    <div class="panel">

        <div class="readout-class-label">
            PREDICTED CLASS
        </div>

        <div class="readout-class-value">
            {predicted_class.upper()}
        </div>

        <div class="readout-metrics">

            <div>
                <div class="readout-metric-label">
                    CONFIDENCE
                </div>

                <div class="readout-metric-value">
                    {confidence * 100:.2f}%
                </div>
            </div>

            <div>
                <div class="readout-metric-label">
                    LATENCY
                </div>

                <div class="readout-metric-value">
                    {latency_ms:.2f} ms
                </div>
            </div>

            <div>
                <div class="readout-metric-label">
                    MODEL
                </div>

                <div class="readout-metric-value">
                    {model_name}
                </div>
            </div>

        </div>

        <div
            class="readout-class-label"
            style="margin-bottom: 12px;"
        >
            CLASS PROBABILITIES
        </div>

        {rows}

    </div>
    """

    st.markdown(
        clean_html(html),
        unsafe_allow_html=True,
    )


# ============================================================
# Metrics Helpers
# ============================================================

def load_csv_safe(path):

    if not path.exists():
        return None

    try:
        return pd.read_csv(path)

    except Exception as error:

        st.warning(
            f"Could not read {path.name}: {error}"
        )

        return None


def normalize_metric_name(value):

    return (
        str(value)
        .strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .replace("  ", " ")
    )


def extract_metric(
    df,
    possible_names,
):

    if df is None or df.empty:
        return None

    normalized_names = {
        normalize_metric_name(name)
        for name in possible_names
    }

    metric_col = None
    value_col = None

    for column in df.columns:

        normalized_column = (
            normalize_metric_name(column)
        )

        if normalized_column in {
            "metric",
            "metrics",
            "name",
        }:
            metric_col = column

        if normalized_column in {
            "value",
            "score",
            "metric value",
        }:
            value_col = column

    if (
        metric_col is not None
        and value_col is not None
    ):

        for _, row in df.iterrows():

            metric_name = normalize_metric_name(
                row[metric_col]
            )

            if metric_name in normalized_names:
                return row[value_col]

    for column in df.columns:

        normalized_column = (
            normalize_metric_name(column)
        )

        if normalized_column in normalized_names:

            return df.iloc[0][column]

    return None


def format_metric(value):

    if value is None:
        return "N/A"

    try:

        numeric_value = float(value)

        return f"{numeric_value:.4f}"

    except (
        TypeError,
        ValueError,
    ):

        return str(value)


def model_file_prefix(model_name):

    if model_name == "Baseline CNN":
        return "baseline_cnn"

    return "resnet18"


def filter_model_rows(
    df,
    model_name,
):

    if df is None or df.empty:
        return df

    model_key = model_file_prefix(
        model_name
    ).lower()

    for column in df.columns:

        if normalize_metric_name(column) == "model":

            values = (
                df[column]
                .astype(str)
                .str.lower()
            )

            filtered = df[
                values.str.contains(
                    model_key,
                    na=False,
                )
            ]

            if not filtered.empty:
                return filtered

    return df


# ============================================================
# Main Model Metrics
# ============================================================

def show_main_model_metrics(
    model_name,
    split_name,
):

    prefix = model_file_prefix(
        model_name
    )

    summary_path = (
        METRICS_DIR
        / f"{prefix}_{split_name}_summary.csv"
    )

    summary_df = load_csv_safe(
        summary_path
    )

    if (
        summary_df is None
        or summary_df.empty
    ):

        st.warning(
            f"No summary metrics found for "
            f"{model_name} ({split_name})."
        )

        return

    st.markdown(
        clean_html(
            f"""
            <div class="metrics-section-title">
                {model_name.upper()} ·
                {split_name.upper()} EVALUATION
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    accuracy = extract_metric(
        summary_df,
        [
            "Accuracy",
            "accuracy",
        ],
    )

    macro_precision = extract_metric(
        summary_df,
        [
            "Macro Precision",
            "macro_precision",
        ],
    )

    macro_recall = extract_metric(
        summary_df,
        [
            "Macro Recall",
            "macro_recall",
        ],
    )

    macro_f1 = extract_metric(
        summary_df,
        [
            "Macro F1",
            "Macro F1 Score",
            "Macro F1-Score",
            "macro_f1",
        ],
    )

    weighted_f1 = extract_metric(
        summary_df,
        [
            "Weighted F1",
            "Weighted F1 Score",
            "weighted_f1",
        ],
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric(
            "ACCURACY",
            format_metric(accuracy),
        )

    with col2:
        st.metric(
            "MACRO PRECISION",
            format_metric(macro_precision),
        )

    with col3:
        st.metric(
            "MACRO RECALL",
            format_metric(macro_recall),
        )

    with col4:
        st.metric(
            "MACRO F1",
            format_metric(macro_f1),
        )

    with col5:
        st.metric(
            "WEIGHTED F1",
            format_metric(weighted_f1),
        )

    with st.expander(
        f"View {model_name} {split_name} summary data"
    ):

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )


def show_model_comparison(split_name):

    rows = []

    for model_name in [
        "Baseline CNN",
        "ResNet18",
    ]:

        prefix = model_file_prefix(
            model_name
        )

        path = (
            METRICS_DIR
            / f"{prefix}_{split_name}_summary.csv"
        )

        df = load_csv_safe(path)

        if df is None or df.empty:
            continue

        rows.append(
            {
                "Model": model_name,

                "Accuracy": extract_metric(
                    df,
                    ["Accuracy", "accuracy"],
                ),

                "Macro Precision": extract_metric(
                    df,
                    [
                        "Macro Precision",
                        "macro_precision",
                    ],
                ),

                "Macro Recall": extract_metric(
                    df,
                    [
                        "Macro Recall",
                        "macro_recall",
                    ],
                ),

                "Macro F1": extract_metric(
                    df,
                    [
                        "Macro F1",
                        "Macro F1 Score",
                        "macro_f1",
                    ],
                ),

                "Weighted F1": extract_metric(
                    df,
                    [
                        "Weighted F1",
                        "Weighted F1 Score",
                        "weighted_f1",
                    ],
                ),
            }
        )

    if not rows:

        st.warning(
            "No model comparison data available."
        )

        return

    comparison_df = pd.DataFrame(rows)

    st.markdown(
        '<div class="metrics-section-title">MODEL COMPARISON</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
    )


def show_per_class_metrics(
    model_name,
    split_name,
):

    prefix = model_file_prefix(
        model_name
    )

    path = (
        METRICS_DIR
        / f"{prefix}_{split_name}_per_class.csv"
    )

    df = load_csv_safe(path)

    if df is None or df.empty:

        st.info(
            f"No per-class metrics available for "
            f"{model_name} ({split_name})."
        )

        return

    st.markdown(
        clean_html(
            f"""
            <div class="metrics-section-title">
                {model_name.upper()} · PER-CLASS PERFORMANCE
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


def show_confusion_matrix(
    model_name,
    split_name,
):

    prefix = model_file_prefix(
        model_name
    )

    path = (
        METRICS_DIR
        / f"{prefix}_{split_name}_confusion_matrix.csv"
    )

    df = load_csv_safe(path)

    if df is None or df.empty:

        st.info(
            f"No confusion matrix available for "
            f"{model_name} ({split_name})."
        )

        return

    st.markdown(
        clean_html(
            f"""
            <div class="metrics-section-title">
                {model_name.upper()} ·
                {split_name.upper()} CONFUSION MATRIX
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# Cost Analysis
# ============================================================

def show_cost_analysis(
    model_name,
    split_name,
):

    prefix = model_file_prefix(
        model_name
    )

    st.markdown(
        clean_html(
            f"""
            <div class="metrics-section-title">
                {model_name.upper()} ·
                {split_name.upper()} COST ANALYSIS
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    summary_path = (
        COST_DIR
        / f"{prefix}_{split_name}_cost_summary.csv"
    )

    summary_df = load_csv_safe(
        summary_path
    )

    if (
        summary_df is not None
        and not summary_df.empty
    ):

        st.markdown(
            "**Cost Summary**"
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Cost summary is not available."
        )

    matrix_path = (
        COST_DIR
        / f"{prefix}_{split_name}_cost_matrix.csv"
    )

    matrix_df = load_csv_safe(
        matrix_path
    )

    if (
        matrix_df is not None
        and not matrix_df.empty
    ):

        st.markdown(
            "**Cost Matrix**"
        )

        st.dataframe(
            matrix_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Cost matrix is not available."
        )

    class_path = (
        COST_DIR
        / f"{prefix}_{split_name}_cost_by_class.csv"
    )

    class_df = load_csv_safe(
        class_path
    )

    if (
        class_df is not None
        and not class_df.empty
    ):

        st.markdown(
            "**Cost by Class**"
        )

        st.dataframe(
            class_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Class-level cost analysis is not available."
        )


# ============================================================
# Threshold Analysis
# ============================================================

def show_threshold_analysis(
    model_name,
):

    st.markdown(
        clean_html(
            f"""
            <div class="metrics-section-title">
                {model_name.upper()} ·
                DECISION THRESHOLD ANALYSIS
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    threshold_summary_path = (
        THRESHOLD_DIR
        / "threshold_summary.csv"
    )

    threshold_summary = load_csv_safe(
        threshold_summary_path
    )

    if (
        threshold_summary is not None
        and not threshold_summary.empty
    ):

        threshold_summary = filter_model_rows(
            threshold_summary,
            model_name,
        )

        st.markdown(
            "**Threshold Summary**"
        )

        st.dataframe(
            threshold_summary,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Threshold summary is not available."
        )

    threshold_sweep_path = (
        THRESHOLD_DIR
        / "threshold_sweep.csv"
    )

    threshold_sweep = load_csv_safe(
        threshold_sweep_path
    )

    if (
        threshold_sweep is not None
        and not threshold_sweep.empty
    ):

        threshold_sweep = filter_model_rows(
            threshold_sweep,
            model_name,
        )

        st.markdown(
            "**Threshold Sweep**"
        )

        st.dataframe(
            threshold_sweep,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Threshold sweep is not available."
        )

    type_error_path = (
        THRESHOLD_DIR
        / "type_i_ii_by_class.csv"
    )

    type_error_df = load_csv_safe(
        type_error_path
    )

    if (
        type_error_df is not None
        and not type_error_df.empty
    ):

        type_error_df = filter_model_rows(
            type_error_df,
            model_name,
        )

        st.markdown(
            "**Type I / Type II Error Analysis**"
        )

        st.dataframe(
            type_error_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Type I / Type II analysis is not available."
        )

    sensitivity_path = (
        THRESHOLD_DIR
        / "cost_sensitivity.csv"
    )

    sensitivity_df = load_csv_safe(
        sensitivity_path
    )

    if (
        sensitivity_df is not None
        and not sensitivity_df.empty
    ):

        sensitivity_df = filter_model_rows(
            sensitivity_df,
            model_name,
        )

        st.markdown(
            "**Cost Sensitivity**"
        )

        st.dataframe(
            sensitivity_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "Cost sensitivity analysis is not available."
        )

    dollar_path = (
        THRESHOLD_DIR
        / "cost_matrix_dollars.csv"
    )

    dollar_df = load_csv_safe(
        dollar_path
    )

    if (
        dollar_df is not None
        and not dollar_df.empty
    ):

        dollar_df = filter_model_rows(
            dollar_df,
            model_name,
        )

        st.markdown(
            "**Cost Matrix — Dollar Analysis**"
        )

        st.dataframe(
            dollar_df,
            use_container_width=True,
            hide_index=True,
        )

    plot_path = (
        THRESHOLD_DIR
        / "cost_vs_threshold.png"
    )

    if plot_path.exists():

        st.markdown(
            "**Cost vs Threshold**"
        )

        st.image(
            str(plot_path),
            use_container_width=True,
        )


# ============================================================
# Inference Latency
# ============================================================

def show_inference_latency():

    path = (
        METRICS_DIR
        / "inference_latency.csv"
    )

    df = load_csv_safe(path)

    if df is None or df.empty:

        st.info(
            "Inference latency data is not available."
        )

        return

    st.markdown(
        clean_html(
            """
            <div class="metrics-section-title">
                INFERENCE LATENCY
            </div>
            """
        ),
        unsafe_allow_html=True,
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    latency_column = None

    for column in df.columns:

        if "latency" in str(
            column
        ).lower():

            latency_column = column
            break

    if latency_column is None:
        return

    latency_values = pd.to_numeric(
        df[latency_column],
        errors="coerce",
    ).dropna()

    if latency_values.empty:
        return

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "AVERAGE LATENCY",
            f"{latency_values.mean():.2f} ms",
        )

    with col2:

        st.metric(
            "MIN LATENCY",
            f"{latency_values.min():.2f} ms",
        )

    with col3:

        st.metric(
            "MAX LATENCY",
            f"{latency_values.max():.2f} ms",
        )

    model_column = None

    for column in df.columns:

        if normalize_metric_name(
            column
        ) == "model":

            model_column = column
            break

    if model_column is not None:

        st.markdown(
            "**Model-wise Inference Latency**"
        )

        latency_summary = (
            df.groupby(
                model_column
            )[latency_column]
            .agg(
                [
                    "count",
                    "mean",
                    "min",
                    "max",
                ]
            )
            .reset_index()
        )

        latency_summary = latency_summary.rename(
            columns={
                "count": "Samples",
                "mean": "Average Latency (ms)",
                "min": "Minimum Latency (ms)",
                "max": "Maximum Latency (ms)",
            }
        )

        st.dataframe(
            latency_summary,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# Header
# ============================================================

header_html = """
<div class="console-header">
    <span class="console-mark">◈</span>
    <span class="console-title">SecureVision AI</span>
</div>

<div class="console-subtitle">
    SURVEILLANCE IMAGE CLASSIFICATION SYSTEM · CIFAR-10
</div>
"""

st.markdown(
    clean_html(header_html),
    unsafe_allow_html=True,
)


# ============================================================
# Top Navigation
# ============================================================

tab_predict, tab_metrics, tab_about = st.tabs(
    [
        "◎  PREDICT",
        "▤  PROJECT METRICS",
        "▤  ABOUT PROJECT",
    ]
)


# ============================================================
# PREDICT TAB
# ============================================================

with tab_predict:

    feed_col, readout_col = st.columns(
        [1, 1],
        gap="large",
    )

    with feed_col:

        st.markdown(
            '<div class="panel-label">MODEL SELECTION</div>',
            unsafe_allow_html=True,
        )

        model_name = st.selectbox(
            "Select Model",
            [
                "Baseline CNN",
                "ResNet18",
            ],
            label_visibility="collapsed",
        )

        st.markdown(
            '<div class="panel-label" style="margin-top: 18px;">IMAGE FEED</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Upload an image",
            type=[
                "jpg",
                "jpeg",
                "png",
            ],
            label_visibility="collapsed",
        )

        run_clicked = False

        if uploaded_file is not None:

            image = Image.open(
                uploaded_file
            ).convert("RGB")

            render_feed_frame(
                image,
                "32×32 · RGB · READY",
            )

            run_clicked = st.button(
                "▶  RUN DETECTION",
                type="primary",
                use_container_width=True,
            )

        else:

            empty_html = """
            <div class="readout-empty">
                NO IMAGE LOADED<br>
                <span style="opacity: 0.6;">
                    upload a feed image to begin
                </span>
            </div>
            """

            st.markdown(
                clean_html(empty_html),
                unsafe_allow_html=True,
            )

    with readout_col:

        st.markdown(
            '<div class="panel-label">DETECTION READOUT</div>',
            unsafe_allow_html=True,
        )

        if (
            uploaded_file is not None
            and run_clicked
        ):

            try:

                with st.spinner(
                    "Running inference..."
                ):

                    model = load_model(
                        model_name
                    )

                    (
                        predicted_class,
                        confidence,
                        latency_ms,
                        probabilities,
                    ) = predict_image(
                        model,
                        image,
                    )

                render_readout(
                    predicted_class,
                    confidence,
                    latency_ms,
                    model_name,
                    probabilities,
                )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)

        else:

            empty_html = """
            <div class="readout-empty">
                STANDING BY<br>
                <span style="opacity: 0.6;">
                    awaiting detection run
                </span>
            </div>
            """

            st.markdown(
                clean_html(empty_html),
                unsafe_allow_html=True,
            )


# ============================================================
# PROJECT METRICS TAB
# ============================================================

with tab_metrics:

    metrics_intro = """
    <div class="panel">

        <div class="panel-label">
            MODEL EVALUATION & OPERATIONAL ANALYTICS
        </div>

        <p style="color: var(--text); line-height: 1.6;">
            Held-out model evaluation, per-class performance,
            confusion analysis, inference latency, cost analysis,
            and decision-threshold analysis for SecureVision AI.
        </p>

    </div>
    """

    st.markdown(
        clean_html(metrics_intro),
        unsafe_allow_html=True,
    )

    st.markdown("")

    # --------------------------------------------------------
    # Evaluation Split
    # --------------------------------------------------------

    evaluation_split = st.selectbox(
        "Evaluation Split",
        [
            "test",
            "validation",
        ],
        format_func=lambda value: value.upper(),
        key="evaluation_split_selector",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Model Comparison
    # --------------------------------------------------------

    st.markdown(
        "## Model Evaluation Comparison"
    )

    show_model_comparison(
        evaluation_split
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Baseline CNN
    # --------------------------------------------------------

    st.markdown(
        "## Baseline CNN"
    )

    show_main_model_metrics(
        "Baseline CNN",
        evaluation_split,
    )

    st.markdown("")

    show_per_class_metrics(
        "Baseline CNN",
        evaluation_split,
    )

    st.markdown("")

    show_confusion_matrix(
        "Baseline CNN",
        evaluation_split,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # ResNet18
    # --------------------------------------------------------

    st.markdown(
        "## ResNet18"
    )

    show_main_model_metrics(
        "ResNet18",
        evaluation_split,
    )

    st.markdown("")

    show_per_class_metrics(
        "ResNet18",
        evaluation_split,
    )

    st.markdown("")

    show_confusion_matrix(
        "ResNet18",
        evaluation_split,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Operational Model Selector
    # --------------------------------------------------------

    detailed_model = st.selectbox(
        "Select Model for Operational Analysis",
        [
            "Baseline CNN",
            "ResNet18",
        ],
        key="operational_model_selector",
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Cost Analysis
    # --------------------------------------------------------

    st.markdown(
        "## Classification Cost Analysis"
    )

    show_cost_analysis(
        detailed_model,
        evaluation_split,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Threshold Analysis
    # --------------------------------------------------------

    st.markdown(
        "## Decision Threshold Analysis"
    )

    show_threshold_analysis(
        detailed_model,
    )

    st.markdown("---")

    # --------------------------------------------------------
    # Inference Latency
    # --------------------------------------------------------

    st.markdown(
        "## Inference Latency"
    )

    show_inference_latency()


# ============================================================
# ABOUT PROJECT TAB
# ============================================================

with tab_about:

    col_left, col_right = st.columns(
        [3, 2],
        gap="large",
    )

    with col_left:

        about_left_html = """
        <div class="panel">

            <div class="panel-label">
                OVERVIEW
            </div>

            <p style="color: var(--text); line-height: 1.6;">
                SecureVision AI is a computer vision system that
                classifies surveillance images into 10 target
                categories, built on the CIFAR-10 dataset.
                It reports a prediction alongside its confidence
                and inference latency.
            </p>

        </div>

        <div style="height: 16px;"></div>

        <div class="panel">

            <div class="panel-label">
                ML PIPELINE
            </div>

            <p
                style="
                    color: var(--text-dim);
                    line-height: 1.9;
                    font-family: 'JetBrains Mono', monospace;
                    font-size: 13px;
                "
            >
                01 · Data acquisition and validation<br>
                02 · Data preprocessing<br>
                03 · Exploratory data analysis<br>
                04 · Feature / data preparation<br>
                05 · Model training<br>
                06 · Model inference<br>
                07 · Performance evaluation<br>
                08 · Interactive prediction
            </p>

        </div>

        <div style="height: 16px;"></div>

        <div class="panel">

            <div class="panel-label">
                ML ENGINEER COMPONENT
            </div>

            <p style="color: var(--text); line-height: 1.6;">
                The inference pipeline handles image preprocessing,
                model loading, prediction, confidence calculation,
                inference latency measurement, prediction logging,
                and pipeline sanity tests.
            </p>

        </div>
        """

        st.markdown(
            clean_html(about_left_html),
            unsafe_allow_html=True,
        )

    with col_right:

        class_list = "".join(
            f"""
            <div style="
                padding: 6px 0;
                border-bottom: 1px solid var(--border);
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                color: var(--text);
            ">
                {i:02d} · {name}
            </div>
            """
            for i, name in enumerate(
                CLASS_NAMES
            )
        )

        about_right_html = f"""
        <div class="panel">

            <div class="panel-label">
                DATASET · CIFAR-10 CLASSES
            </div>

            {class_list}

        </div>

        <div style="height: 16px;"></div>

        <div class="panel">

            <div class="panel-label">
                MODELS AVAILABLE
            </div>

            <div style="
                font-family: 'JetBrains Mono', monospace;
                font-size: 13px;
                color: var(--text);
                line-height: 2;
            ">

                <span style="color: var(--signal);">
                    ▸
                </span>
                Baseline CNN
                <br>

                <span style="color: var(--signal);">
                    ▸
                </span>
                ResNet18

            </div>

        </div>
        """

        st.markdown(
            clean_html(about_right_html),
            unsafe_allow_html=True,
        )