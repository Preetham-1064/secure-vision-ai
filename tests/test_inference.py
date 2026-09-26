import unittest

import numpy as np

from src.predict import preprocess_image


class TestInferencePipeline(unittest.TestCase):

    def test_preprocess_shape(self):
        """Processed image should have batch shape (1, 32, 32, 3)."""

        image = np.zeros(
            (32, 32, 3),
            dtype=np.uint8
        )

        processed = preprocess_image(image)

        self.assertEqual(
            tuple(processed.shape),
            (1, 32, 32, 3)
        )

    def test_preprocess_dtype(self):
        """Processed image should be float32."""

        image = np.zeros(
            (32, 32, 3),
            dtype=np.uint8
        )

        processed = preprocess_image(image)

        self.assertEqual(
            processed.dtype.name,
            "float32"
        )

    def test_preprocess_normalization(self):
        """Preprocessing should convert image values correctly."""

        image = np.zeros(
            (32, 32, 3),
            dtype=np.uint8
        )

        processed = preprocess_image(image)

        self.assertTrue(
            np.isfinite(
                processed.numpy()
            ).all()
        )

    def test_invalid_image_shape(self):
        """Invalid image dimensions should raise ValueError."""

        image = np.zeros(
            (28, 28, 3),
            dtype=np.uint8
        )

        with self.assertRaises(
            ValueError
        ):

            preprocess_image(image)


if __name__ == "__main__":
    unittest.main()