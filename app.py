import time
from pathlib import Path

import numpy as np
import streamlit as st
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
    page_title="SentriVision AI",
    page_icon="🔐",
    layout="wide",
)


# ============================================================
# Custom CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #666666;
        margin-bottom: 30px;
    }

    .prediction-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 20px;
    }

    .prediction-label {
        font-size: 16px;
        color: #666666;
    }

    .prediction-value {
        font-size: 32px;
        font-weight: 700;
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
    model.build(input_shape=(None, 32, 32, 3))

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

    probabilities = predictions[0]

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
# Sidebar Navigation
# ============================================================

st.sidebar.title("SentriVision AI")

page = st.sidebar.radio(
    "Menu",
    [
        "Predict",
        "About Project",
    ],
)


# ============================================================
# Header
# ============================================================

st.markdown(
    '<div class="main-title">🔐 SentriVision AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "AI-powered surveillance image classification"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# PREDICT PAGE
# ============================================================

if page == "Predict":

    st.header("Image Prediction")

    st.write(
        "Upload an image and use the trained "
        "CNN model to classify it into one of "
        "the 10 CIFAR-10 categories."
    )

    # --------------------------------------------------------
    # Model selection
    # --------------------------------------------------------

    model_name = st.selectbox(
        "Select Model",
        [
            "Baseline CNN",
            "ResNet18",
        ],
    )

    # --------------------------------------------------------
    # Image upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
        ],
    )

    if uploaded_file is not None:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.subheader("Uploaded Image")

        # Display original image
        st.image(
            image,
            caption="Original Image",
            width=400,
        )

        st.info(
            "The image will be resized to "
            "32 × 32 pixels because the "
            "CIFAR-10 models were trained "
            "using 32 × 32 images."
        )

        # ----------------------------------------------------
        # Predict button
        # ----------------------------------------------------

        if st.button(
            "🔍 Predict",
            type="primary",
            use_container_width=True,
        ):

            try:

                with st.spinner(
                    "Running prediction..."
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

                # ------------------------------------------------
                # Result
                # ------------------------------------------------

                st.success(
                    "Prediction completed successfully."
                )

                st.markdown(
                    '<div class="prediction-box">',
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="prediction-label">'
                    "Predicted Class"
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    f'<div class="prediction-value">'
                    f"{predicted_class.upper()}"
                    "</div>",
                    unsafe_allow_html=True,
                )

                st.markdown(
                    "</div>",
                    unsafe_allow_html=True,
                )

                # ------------------------------------------------
                # Metrics
                # ------------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Confidence",
                        f"{confidence * 100:.2f}%",
                    )

                with col2:

                    st.metric(
                        "Inference Latency",
                        f"{latency_ms:.2f} ms",
                    )

                with col3:

                    st.metric(
                        "Model",
                        model_name,
                    )

                # ------------------------------------------------
                # Prediction probabilities
                # ------------------------------------------------

                st.subheader(
                    "Class Probabilities"
                )

                probability_data = {
                    CLASS_NAMES[i]: float(
                        probabilities[i]
                    )
                    for i in range(
                        len(CLASS_NAMES)
                    )
                }

                st.bar_chart(
                    probability_data
                )

                # ------------------------------------------------
                # Original image + result
                # ------------------------------------------------

                st.subheader(
                    "Prediction Result"
                )

                result_col1, result_col2 = (
                    st.columns(2)
                )

                with result_col1:

                    st.image(
                        image,
                        caption="Original Image",
                        width=350,
                    )

                with result_col2:

                    st.markdown(
                        "### Prediction"
                    )

                    st.markdown(
                        f"## {predicted_class.upper()}"
                    )

                    st.write(
                        f"Confidence: "
                        f"**{confidence * 100:.2f}%**"
                    )

                    st.write(
                        f"Latency: "
                        f"**{latency_ms:.2f} ms**"
                    )

            except Exception as error:

                st.error(
                    "Prediction failed."
                )

                st.exception(error)


# ============================================================
# ABOUT PROJECT PAGE
# ============================================================

elif page == "About Project":

    st.header("About SentriVision AI")

    st.write(
        """
        **SentriVision AI** is an AI-powered computer
        vision system designed to classify surveillance
        images into 10 target categories.
        """
    )

    st.subheader("Project Objective")

    st.write(
        """
        The system uses machine learning and deep learning
        techniques to classify images automatically and
        provide a prediction with confidence and inference
        latency.
        """
    )

    st.subheader("Dataset")

    st.write(
        """
        The project uses the CIFAR-10 dataset containing
        10 image categories:
        """
    )

    for class_name in CLASS_NAMES:

        st.write(
            f"• {class_name.capitalize()}"
        )

    st.subheader("Models")

    st.write(
        """
        Two trained deep learning models are available
        in the application:
        """
    )

    st.write(
        "• Baseline CNN"
    )

    st.write(
        "• ResNet18"
    )

    st.subheader("ML Pipeline")

    st.write(
        """
        The project follows an end-to-end machine learning
        pipeline covering:

        1. Data acquisition and validation
        2. Data preprocessing
        3. Exploratory data analysis
        4. Feature/data preparation
        5. Model training
        6. Model inference
        7. Performance evaluation
        8. Interactive prediction
        """
    )

    st.subheader("ML Engineer Component")

    st.write(
        """
        The ML Engineer component provides the inference
        pipeline, including image preprocessing, model
        loading, prediction, confidence calculation,
        inference latency measurement, prediction logging,
        and pipeline sanity tests.
        """
    )

    st.info(
        "Secure" \
        "Vision AI — Machine Learning Lab Project"
    )
