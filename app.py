import base64
import io
import time
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

from src.model import build_baseline_cnn, ResNet18
from src.predict import preprocess_image


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"


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
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="SecureVision AI",
    page_icon="◈",
    layout="wide",
)


# ============================================================
# Custom CSS — dark console / detection-readout theme
# ============================================================

st.markdown(
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

    /* ---------------------------------------------------- */
    /* Header                                                */
    /* ---------------------------------------------------- */

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

    /* ---------------------------------------------------- */
    /* Top nav tabs                                          */
    /* ---------------------------------------------------- */

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

    /* ---------------------------------------------------- */
    /* Panels                                                */
    /* ---------------------------------------------------- */

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

    /* ---------------------------------------------------- */
    /* Image feed frame — corner brackets                    */
    /* ---------------------------------------------------- */

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

    /* ---------------------------------------------------- */
    /* Detection readout                                     */
    /* ---------------------------------------------------- */

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

    /* ---------------------------------------------------- */
    /* Streamlit widget overrides                            */
    /* ---------------------------------------------------- */

    .stSelectbox label, .stFileUploader label {
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

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Model Loading
# ============================================================

@st.cache_resource
def load_model(model_name):
    """
    Load a trained model and its weights.
    """

    if model_name == "Baseline CNN":

        model = build_baseline_cnn()

        weight_path = (
            MODEL_DIR / "baseline_cnn.weights.h5"
        )

    elif model_name == "ResNet18":

        model = ResNet18()

        weight_path = (
            MODEL_DIR / "resnet18.weights.h5"
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}"
        )

    if not weight_path.exists():
        raise FileNotFoundError(
            f"Model weights not found: {weight_path}"
        )

    # Subclassed models (ResNet18) have no variables until built;
    # Sequential models (Baseline CNN) are already built via their
    # Input layer, so this is a no-op for them.
    model(tf.zeros((1, 32, 32, 3)), training=False)

    model.load_weights(str(weight_path))

    return model


# ============================================================
# Prediction
# ============================================================

def predict_image(model, image):
    """
    Preprocess image and generate prediction.
    """

    # Convert PIL image to RGB
    image = image.convert("RGB")

    # CIFAR-10 models expect 32x32 images
    image = image.resize((32, 32))

    image_array = np.array(
        image,
        dtype=np.uint8,
    )

    # Use the existing preprocessing pipeline
    processed_image = preprocess_image(
        image_array
    )

    # Warm-up / prediction
    start_time = time.perf_counter()

    predictions = model.predict(
        processed_image,
        verbose=0,
    )

    latency_ms = (
        time.perf_counter() - start_time
    ) * 1000

    # Convert raw logits -> probabilities
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
    """Encode a PIL image as a base64 data URI for embedding in HTML."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def render_feed_frame(image, status_text):
    """Render the uploaded image inside a bracketed 'camera feed' frame."""
    data_uri = image_to_data_uri(image)
    st.markdown(
        f"""
        <div class="feed-frame">
            <div class="feed-status">{status_text}</div>
            <div class="feed-corner-tl"></div>
            <div class="feed-corner-br"></div>
            <img src="{data_uri}" />
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_readout(predicted_class, confidence, latency_ms, model_name, probabilities):
    """Render the detection readout panel: class, metrics, probability bars."""

    order = np.argsort(probabilities)[::-1]

    rows = ""
    for idx in order:
        pct = probabilities[idx] * 100
        is_top = idx == order[0]
        fill_class = "" if is_top else "low"
        name_class = "top" if is_top else ""
        rows += f"""
        <div class="prob-row">
            <div class="prob-name {name_class}">{CLASS_NAMES[idx]}</div>
            <div class="prob-track">
                <div class="prob-fill {fill_class}" style="width:{pct:.1f}%"></div>
            </div>
            <div class="prob-value">{pct:.1f}%</div>
        </div>
        """

    st.markdown(
        f"""
        <div class="panel">
            <div class="readout-class-label">PREDICTED CLASS</div>
            <div class="readout-class-value">{predicted_class.upper()}</div>
            <div class="readout-metrics">
                <div>
                    <div class="readout-metric-label">CONFIDENCE</div>
                    <div class="readout-metric-value">{confidence * 100:.2f}%</div>
                </div>
                <div>
                    <div class="readout-metric-label">LATENCY</div>
                    <div class="readout-metric-value">{latency_ms:.2f} ms</div>
                </div>
                <div>
                    <div class="readout-metric-label">MODEL</div>
                    <div class="readout-metric-value">{model_name}</div>
                </div>
            </div>
            <div class="readout-class-label" style="margin-bottom: 12px;">CLASS PROBABILITIES</div>
            {rows}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# Header
# ============================================================

st.markdown(
    """
    <div class="console-header">
        <span class="console-mark">◈</span>
        <span class="console-title">SentriVision AI</span>
    </div>
    <div class="console-subtitle">SURVEILLANCE IMAGE CLASSIFICATION SYSTEM · CIFAR-10</div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Top Navigation
# ============================================================

tab_predict, tab_about = st.tabs(["◎  PREDICT", "▤  ABOUT PROJECT"])


# ============================================================
# PREDICT TAB
# ============================================================

with tab_predict:

    feed_col, readout_col = st.columns([1, 1], gap="large")

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
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )

        run_clicked = False

        if uploaded_file is not None:

            image = Image.open(uploaded_file).convert("RGB")

            render_feed_frame(image, "32×32 · RGB · READY")

            run_clicked = st.button(
                "▶  RUN DETECTION",
                type="primary",
                use_container_width=True,
            )

        else:
            st.markdown(
                """
                <div class="readout-empty">
                    NO IMAGE LOADED<br>
                    <span style="opacity: 0.6;">upload a feed image to begin</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with readout_col:

        st.markdown(
            '<div class="panel-label">DETECTION READOUT</div>',
            unsafe_allow_html=True,
        )

        if uploaded_file is not None and run_clicked:

            try:
                with st.spinner("Running inference..."):

                    model = load_model(model_name)

                    (
                        predicted_class,
                        confidence,
                        latency_ms,
                        probabilities,
                    ) = predict_image(model, image)

                render_readout(
                    predicted_class,
                    confidence,
                    latency_ms,
                    model_name,
                    probabilities,
                )

            except Exception as error:
                st.error("Prediction failed.")
                st.exception(error)

        else:
            st.markdown(
                """
                <div class="readout-empty">
                    STANDING BY<br>
                    <span style="opacity: 0.6;">awaiting detection run</span>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ============================================================
# ABOUT PROJECT TAB
# ============================================================

with tab_about:

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:

        st.markdown(
            """
            <div class="panel">
                <div class="panel-label">OVERVIEW</div>
                <p style="color: var(--text); line-height: 1.6;">
                    SecureVision AI is a computer vision system that classifies
                    surveillance images into 10 target categories, built on the
                    CIFAR-10 dataset. It reports a prediction alongside its
                    confidence and inference latency.
                </p>
            </div>
            <div style="height: 16px;"></div>
            <div class="panel">
                <div class="panel-label">ML PIPELINE</div>
                <p style="color: var(--text-dim); line-height: 1.9; font-family: 'JetBrains Mono', monospace; font-size: 13px;">
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
                <div class="panel-label">ML ENGINEER COMPONENT</div>
                <p style="color: var(--text); line-height: 1.6;">
                    The inference pipeline handles image preprocessing, model
                    loading, prediction, confidence calculation, inference
                    latency measurement, prediction logging, and pipeline
                    sanity tests.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_right:

        class_list = "".join(
            f'<div style="padding: 6px 0; border-bottom: 1px solid var(--border); '
            f'font-family: \'JetBrains Mono\', monospace; font-size: 13px; color: var(--text);">'
            f'{i:02d} · {name}</div>'
            for i, name in enumerate(CLASS_NAMES)
        )

        st.markdown(
            f"""
            <div class="panel">
                <div class="panel-label">DATASET · CIFAR-10 CLASSES</div>
                {class_list}
            </div>
            <div style="height: 16px;"></div>
            <div class="panel">
                <div class="panel-label">MODELS AVAILABLE</div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: var(--text); line-height: 2;">
                    <span style="color: var(--signal);">▸</span> Baseline CNN<br>
                    <span style="color: var(--signal);">▸</span> ResNet18
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
