"""
Unit tests for the image preprocessing module.

Verifies:
1. Valid image converts to grayscale.
2. Grayscale output has one channel (2D shape).
3. Gaussian blur returns an image.
4. Gaussian blur preserves image dimensions.
5. process() returns the expected processed image.
6. None input raises ValueError.
7. Invalid Gaussian kernel size raises ValueError.
"""

import numpy as np
import pytest

from src.config import GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, PreprocessingConfig
from src.preprocessing import Preprocessor


def test_preprocessor_initialization():
    """Verify that Preprocessor initializes with expected defaults and custom arguments."""
    # Default initialization
    prep = Preprocessor()
    assert prep.gaussian_kernel_size == (5, 5)
    assert prep.gaussian_sigma == 0

    # Custom tuple initialization
    custom_prep = Preprocessor(gaussian_kernel_size=(7, 7), gaussian_sigma=1.5)
    assert custom_prep.gaussian_kernel_size == (7, 7)
    assert custom_prep.gaussian_sigma == 1.5

    # Config-based initialization
    cfg = PreprocessingConfig(gaussian_kernel_size=(9, 9), gaussian_sigma_x=2.0)
    cfg_prep = Preprocessor(config=cfg)
    assert cfg_prep.gaussian_kernel_size == (9, 9)
    assert cfg_prep.gaussian_sigma == 2.0


def test_valid_image_converts_to_grayscale():
    """Requirement 1: Valid image converts to grayscale."""
    prep = Preprocessor()
    # Synthetic BGR image with distinct color values
    bgr_image = np.zeros((100, 150, 3), dtype=np.uint8)
    bgr_image[:, :, 0] = 255  # Blue channel
    bgr_image[:, :, 1] = 128  # Green channel
    bgr_image[:, :, 2] = 64   # Red channel

    gray_image = prep.to_grayscale(bgr_image)
    assert gray_image is not None
    assert isinstance(gray_image, np.ndarray)
    assert gray_image.dtype == np.uint8


def test_grayscale_output_has_one_channel():
    """Requirement 2: Grayscale output has one channel (2D shape)."""
    prep = Preprocessor()
    h, w = 120, 160
    bgr_image = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    gray = prep.to_grayscale(bgr_image)
    # Single-channel 2D matrix
    assert gray.ndim == 2
    assert gray.shape == (h, w)

    # If already a 2D grayscale image, it remains 2D with single channel
    gray_again = prep.to_grayscale(gray)
    assert gray_again.ndim == 2
    assert gray_again.shape == (h, w)


def test_gaussian_blur_returns_an_image():
    """Requirement 3: Gaussian blur returns an image."""
    prep = Preprocessor()
    gray_image = np.random.randint(0, 256, (80, 80), dtype=np.uint8)

    blurred = prep.gaussian_blur(gray_image)
    assert blurred is not None
    assert isinstance(blurred, np.ndarray)
    assert blurred.dtype == np.uint8
    assert blurred.size > 0


def test_gaussian_blur_preserves_image_dimensions():
    """Requirement 4: Gaussian blur preserves image dimensions."""
    prep = Preprocessor(gaussian_kernel_size=(5, 5))
    h, w = 240, 320
    test_image = np.ones((h, w), dtype=np.uint8) * 100

    blurred = prep.gaussian_blur(test_image)
    assert blurred.shape == (h, w)

    # Test also with color 3-channel image
    color_image = np.ones((h, w, 3), dtype=np.uint8) * 100
    blurred_color = prep.gaussian_blur(color_image)
    assert blurred_color.shape == (h, w, 3)


def test_process_returns_expected_image():
    """Requirement 5: process() returns the expected processed image."""
    prep = Preprocessor(gaussian_kernel_size=(5, 5), gaussian_sigma=0)
    h, w = 100, 100
    raw_bgr_frame = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    result = prep.process(raw_bgr_frame)

    # Verifies end-to-end output: valid 2D uint8 smoothed array
    assert isinstance(result, np.ndarray)
    assert result.ndim == 2
    assert result.shape == (h, w)
    assert result.dtype == np.uint8

    # Verify alias preprocess() produces identical result
    result_alias = prep.preprocess(raw_bgr_frame)
    np.testing.assert_array_equal(result, result_alias)


def test_none_input_raises_value_error():
    """Requirement 6: None input raises ValueError."""
    prep = Preprocessor()

    with pytest.raises(ValueError, match="cannot be None"):
        prep.to_grayscale(None)

    with pytest.raises(ValueError, match="cannot be None"):
        prep.gaussian_blur(None)

    with pytest.raises(ValueError, match="cannot be None"):
        prep.process(None)


def test_invalid_gaussian_kernel_size_raises_value_error():
    """Requirement 7: Invalid Gaussian kernel size raises ValueError."""
    # Even dimensions
    with pytest.raises(ValueError, match="odd numbers"):
        Preprocessor(gaussian_kernel_size=(4, 4))

    with pytest.raises(ValueError, match="odd numbers"):
        Preprocessor(gaussian_kernel_size=(5, 6))

    # Negative or zero dimensions
    with pytest.raises(ValueError, match="strictly positive"):
        Preprocessor(gaussian_kernel_size=(-3, 3))

    with pytest.raises(ValueError, match="strictly positive"):
        Preprocessor(gaussian_kernel_size=(0, 5))

    # Invalid types or non-2-tuple
    with pytest.raises(ValueError, match="tuple of 2 elements"):
        Preprocessor(gaussian_kernel_size=(5,))

    # Method-level override with invalid kernel
    prep = Preprocessor(gaussian_kernel_size=(5, 5))
    dummy = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError, match="odd numbers"):
        prep.gaussian_blur(dummy, kernel_size=(4, 4))
