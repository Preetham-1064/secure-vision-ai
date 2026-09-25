"""
Secure Vision AI - Model Training Module (TensorFlow)

Role:
    Data Scientist

Purpose:
    Define and train CNN architectures (baseline CNN and ResNet18) for
    CIFAR-10 surveillance image classification, using TensorFlow/Keras
    to match the framework already pinned in requirements.txt.

Scope:
    This module is responsible for TRAINING only.
    Evaluation / metrics reporting on the held-out test set is owned by
    another team member and must not be duplicated here.

Data source:
    Uses the Data Engineer's validated, split, normalized arrays from
    src/preprocessing.py::prepare_ml_data() -- NOT a separate CIFAR-10
    download. This keeps training reproducible against the same split
    the rest of the team is using.

Outputs:
    Trained model weights, saved under models/.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.preprocessing import (
    load_training_data,
    load_test_data,
    prepare_ml_data,
)


# ==========================================================
# PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


# ==========================================================
# CONFIG
# ==========================================================

tf.random.set_seed(42)

EPOCHS = 15
BATCH_SIZE = 128
LEARNING_RATE = 1e-3

# CIFAR-10 per-channel mean/std (same values as the original PyTorch code)
MEAN = tf.constant([0.4914, 0.4822, 0.4465], dtype=tf.float32)
STD = tf.constant([0.2470, 0.2435, 0.2616], dtype=tf.float32)


# ==========================================================
# DATA
# ==========================================================
# prepare_ml_data() returns images already reshaped to (N, 32, 32, 3)
# and normalized to [0, 1] float32. We standardize with CIFAR mean/std
# on top of that, matching the original transform pipeline.

def _standardize(image, label):
    image = (image - MEAN) / STD
    return image, label


def _augment(image, label):
    # Equivalent to RandomCrop(32, padding=4) + RandomHorizontalFlip
    image = tf.image.resize_with_crop_or_pad(image, 32 + 8, 32 + 8)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image, label


def build_train_dataset():
    X_train_raw, y_train_raw = load_training_data()
    X_test_raw, y_test_raw = load_test_data()

    X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_ml_data(
        X_train_raw, y_train_raw, X_test_raw, y_test_raw
    )

    dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    dataset = dataset.shuffle(buffer_size=len(X_train), seed=42)
    dataset = dataset.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(_standardize, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# ==========================================================
# MODEL ARCHITECTURES
# ==========================================================

def build_baseline_cnn():
    """Simple 3-block CNN used as a performance baseline."""
    return keras.Sequential([
        layers.Input(shape=(32, 32, 3)),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(10),
    ], name="baseline_cnn")


class BasicBlock(layers.Layer):
    """Residual block with a skip connection, used by ResNet18."""

    def __init__(self, planes, stride=1, **kwargs):
        super().__init__(**kwargs)
        self.conv1 = layers.Conv2D(planes, 3, strides=stride, padding="same", use_bias=False)
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(planes, 3, strides=1, padding="same", use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.stride = stride
        self.planes = planes
        self.shortcut_conv = None
        self.shortcut_bn = None
        if stride != 1:
            self.shortcut_conv = layers.Conv2D(planes, 1, strides=stride, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()

    def build(self, input_shape):
        in_planes = input_shape[-1]
        if self.stride == 1 and in_planes != self.planes:
            self.shortcut_conv = layers.Conv2D(self.planes, 1, strides=1, use_bias=False)
            self.shortcut_bn = layers.BatchNormalization()
        super().build(input_shape)

    def call(self, x, training=False):
        out = tf.nn.relu(self.bn1(self.conv1(x), training=training))
        out = self.bn2(self.conv2(out), training=training)

        shortcut = x
        if self.shortcut_conv is not None:
            shortcut = self.shortcut_bn(self.shortcut_conv(x), training=training)

        return tf.nn.relu(out + shortcut)


class ResNet18(keras.Model):
    """ResNet18 adapted for 32x32 CIFAR-10 input (3x3 stem, no initial pool)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.stem_conv = layers.Conv2D(64, 3, strides=1, padding="same", use_bias=False)
        self.stem_bn = layers.BatchNormalization()

        self.layer1 = [BasicBlock(64, 1), BasicBlock(64, 1)]
        self.layer2 = [BasicBlock(128, 2), BasicBlock(128, 1)]
        self.layer3 = [BasicBlock(256, 2), BasicBlock(256, 1)]
        self.layer4 = [BasicBlock(512, 2), BasicBlock(512, 1)]

        self.pool = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(10)

    def call(self, x, training=False):
        x = tf.nn.relu(self.stem_bn(self.stem_conv(x), training=training))

        for block in self.layer1 + self.layer2 + self.layer3 + self.layer4:
            x = block(x, training=training)

        x = self.pool(x)
        return self.fc(x)


# ==========================================================
# TRAINING
# ==========================================================

def train(model, name, dataset):
    """
    Train a model for EPOCHS on the training set.

    Deliberately does NOT touch the test/validation set or compute
    accuracy -- evaluation is owned by another team member and
    reported separately.
    """
    steps_per_epoch = tf.data.experimental.cardinality(dataset).numpy()
    if steps_per_epoch < 0:
        steps_per_epoch = None

    lr_schedule = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=EPOCHS * (steps_per_epoch or 1),
    )
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    for epoch in range(EPOCHS):
        epoch_loss = keras.metrics.Mean()

        for x_batch, y_batch in dataset:
            with tf.GradientTape() as tape:
                logits = model(x_batch, training=True)
                loss = loss_fn(y_batch, logits)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.update_state(loss)

        print(f"{name} epoch {epoch + 1}/{EPOCHS} - loss={epoch_loss.result():.4f}")

    return model


def save_model(model, name):
    """Save trained weights to models/<name>.weights.h5"""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MODELS_DIR / f"{name}.weights.h5"
    model.save_weights(out_path)
    print(f"Saved {name} weights to {out_path}")


# ==========================================================
# MAIN
# ==========================================================

def main():
    dataset = build_train_dataset()

    models = [
        ("baseline_cnn", build_baseline_cnn()),
        ("resnet18", ResNet18()),
    ]

    for name, model in models:
        trained = train(model, name, dataset)
        save_model(trained, name)


if __name__ == "__main__":
    main()