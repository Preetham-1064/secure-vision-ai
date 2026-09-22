"""
SentriVision AI - Data Preprocessing Tests

Tests for the Data Engineer preprocessing pipeline.
"""

import unittest

import numpy as np

from src.preprocessing import (
    reshape_images,
    normalize_images,
    create_train_validation_split,
    calculate_image_hash,
)


class TestPreprocessing(unittest.TestCase):

    # ======================================================
    # Test image reshaping
    # ======================================================

    def test_reshape_images(self):

        # Two flattened CIFAR-10 images
        images = np.zeros(
            (2, 3072),
            dtype=np.uint8
        )

        reshaped = reshape_images(images)

        self.assertEqual(
            reshaped.shape,
            (2, 32, 32, 3)
        )

    # ======================================================
    # Test image normalization
    # ======================================================

    def test_normalize_images(self):

        images = np.array(
            [
                [[[0, 128, 255]]]
            ],
            dtype=np.uint8
        )

        normalized = normalize_images(images)

        self.assertEqual(
            normalized.dtype,
            np.float32
        )

        self.assertAlmostEqual(
            float(normalized.min()),
            0.0
        )

        self.assertAlmostEqual(
            float(normalized.max()),
            1.0
        )

    # ======================================================
    # Test normalization range
    # ======================================================

    def test_normalization_range(self):

        images = np.random.randint(
            0,
            256,
            size=(10, 32, 32, 3),
            dtype=np.uint8
        )

        normalized = normalize_images(images)

        self.assertGreaterEqual(
            normalized.min(),
            0.0
        )

        self.assertLessEqual(
            normalized.max(),
            1.0
        )

    # ======================================================
    # Test stratified train/validation split
    # ======================================================

    def test_train_validation_split(self):

        images = np.zeros(
            (1000, 3072),
            dtype=np.uint8
        )

        labels = np.array(
            [i % 10 for i in range(1000)]
        )

        (
            X_train,
            X_validation,
            y_train,
            y_validation
        ) = create_train_validation_split(
            images,
            labels
        )

        self.assertEqual(
            len(X_train),
            900
        )

        self.assertEqual(
            len(X_validation),
            100
        )

        self.assertEqual(
            len(y_train),
            900
        )

        self.assertEqual(
            len(y_validation),
            100
        )

    # ======================================================
    # Test reproducibility
    # ======================================================

    def test_split_reproducibility(self):

        images = np.arange(
            1000 * 10,
            dtype=np.uint8
        ).reshape(
            1000,
            10
        )

        labels = np.array(
            [i % 10 for i in range(1000)]
        )

        first_split = create_train_validation_split(
            images,
            labels
        )

        second_split = create_train_validation_split(
            images,
            labels
        )

        for first, second in zip(
            first_split,
            second_split
        ):

            np.testing.assert_array_equal(
                first,
                second
            )

    # ======================================================
    # Test image hashing
    # ======================================================

    def test_image_hash(self):

        image = np.zeros(
            (3072,),
            dtype=np.uint8
        )

        hash_1 = calculate_image_hash(
            image
        )

        hash_2 = calculate_image_hash(
            image.copy()
        )

        self.assertEqual(
            hash_1,
            hash_2
        )

    # ======================================================
    # Test different images produce different hashes
    # ======================================================

    def test_different_image_hashes(self):

        image_1 = np.zeros(
            (3072,),
            dtype=np.uint8
        )

        image_2 = np.ones(
            (3072,),
            dtype=np.uint8
        )

        hash_1 = calculate_image_hash(
            image_1
        )

        hash_2 = calculate_image_hash(
            image_2
        )

        self.assertNotEqual(
            hash_1,
            hash_2
        )


if __name__ == "__main__":
    unittest.main()