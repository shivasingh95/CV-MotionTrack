"""
Unit tests for the object detection module (Step 4).

Tests cover all eight required cases:
  1. ObjectDetector initialises correctly (default and custom params).
  2. Valid frame produces a foreground mask.
  3. Foreground mask is single-channel (2-D array).
  4. Invalid / None frame raises ValueError.
  5. clean_mask() returns a valid binary mask.
  6. detect_objects() handles an empty (all-black) mask.
  7. detect_objects() detects a sufficiently large synthetic foreground region.
  8. Small regions below MIN_OBJECT_AREA are ignored.

All tests use synthetic NumPy arrays — no real video file is required.
"""

import cv2
import numpy as np
import pytest

from src.config import (
    BACKGROUND_HISTORY,
    BACKGROUND_VAR_THRESHOLD,
    DETECT_SHADOWS,
    MIN_OBJECT_AREA,
    MORPH_KERNEL_SIZE,
    DetectorConfig,
)
from src.data_models import BoundingBox, Detection
from src.detector import ObjectDetector


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _blank_gray(h: int = 300, w: int = 300) -> np.ndarray:
    """Return an all-zero grayscale frame (simulates static background)."""
    return np.zeros((h, w), dtype=np.uint8)


def _white_rect_mask(
    h: int = 300,
    w: int = 300,
    rect: tuple = (100, 100, 60, 60),
) -> np.ndarray:
    """
    Return a binary mask (0/255) with a white filled rectangle.
    rect = (x, y, width, height)
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    x, y, rw, rh = rect
    mask[y : y + rh, x : x + rw] = 255
    return mask


# ──────────────────────────────────────────────
# Test 1 – Initialisation
# ──────────────────────────────────────────────

class TestObjectDetectorInitialisation:
    """Test 1: ObjectDetector initialises correctly."""

    def test_default_initialisation(self):
        """Detector created with no arguments uses module-level defaults."""
        det = ObjectDetector()
        assert det.history == BACKGROUND_HISTORY
        assert det.var_threshold == BACKGROUND_VAR_THRESHOLD
        assert det.detect_shadows == DETECT_SHADOWS
        assert det.min_area == MIN_OBJECT_AREA
        assert det._subtractor is not None
        assert det.morph_kernel is not None

    def test_custom_parameters(self):
        """Detector accepts and stores custom constructor arguments."""
        det = ObjectDetector(
            history=200,
            var_threshold=32.0,
            detect_shadows=False,
            min_area=300.0,
            max_area=50_000.0,
            morph_kernel_size=(5, 5),
        )
        assert det.history == 200
        assert det.var_threshold == 32.0
        assert det.detect_shadows is False
        assert det.min_area == 300.0
        assert det.max_area == 50_000.0

    def test_config_based_initialisation(self):
        """Detector can be driven entirely from a DetectorConfig dataclass."""
        cfg = DetectorConfig(history=100, var_threshold=8.0, min_contour_area=200.0)
        det = ObjectDetector(config=cfg)
        assert det.history == 100
        assert det.var_threshold == 8.0
        assert det.min_area == 200.0


# ──────────────────────────────────────────────
# Test 2 – Valid frame produces a foreground mask
# ──────────────────────────────────────────────

class TestForegroundMaskProduction:
    """Test 2: Valid frame produces a foreground mask."""

    def test_mask_returned_on_valid_frame(self):
        """apply_background_subtraction returns a non-None NumPy array."""
        det = ObjectDetector()
        frame = _blank_gray()
        mask = det.apply_background_subtraction(frame)
        assert mask is not None
        assert isinstance(mask, np.ndarray)
        assert mask.size > 0

    def test_get_foreground_mask_pipeline(self):
        """get_foreground_mask runs subtraction + cleaning without error."""
        det = ObjectDetector()
        frame = _blank_gray()
        clean = det.get_foreground_mask(frame)
        assert isinstance(clean, np.ndarray)
        assert clean.size > 0


# ──────────────────────────────────────────────
# Test 3 – Foreground mask is single-channel
# ──────────────────────────────────────────────

class TestMaskShape:
    """Test 3: Foreground mask is single-channel (2-D)."""

    def test_raw_mask_is_2d(self):
        """Raw mask from apply_background_subtraction is (H, W) not (H, W, C)."""
        det = ObjectDetector()
        frame = _blank_gray(240, 320)
        mask = det.apply_background_subtraction(frame)
        assert mask.ndim == 2
        assert mask.shape == (240, 320)

    def test_clean_mask_is_2d_same_shape(self):
        """clean_mask output preserves (H, W) dimensions."""
        det = ObjectDetector()
        frame = _blank_gray(240, 320)
        raw = det.apply_background_subtraction(frame)
        clean = det.clean_mask(raw)
        assert clean.ndim == 2
        assert clean.shape == (240, 320)


# ──────────────────────────────────────────────
# Test 4 – Invalid / None frame raises ValueError
# ──────────────────────────────────────────────

class TestInvalidFrameHandling:
    """Test 4: None or invalid frame raises ValueError."""

    def test_none_frame_raises_on_subtraction(self):
        det = ObjectDetector()
        with pytest.raises(ValueError, match="None"):
            det.apply_background_subtraction(None)

    def test_empty_array_raises_on_subtraction(self):
        det = ObjectDetector()
        with pytest.raises(ValueError, match="empty"):
            det.apply_background_subtraction(np.array([]))

    def test_none_frame_raises_on_full_pipeline(self):
        det = ObjectDetector()
        with pytest.raises(ValueError):
            det.get_foreground_mask(None)

    def test_none_mask_raises_on_clean_mask(self):
        det = ObjectDetector()
        with pytest.raises(ValueError, match="None"):
            det.clean_mask(None)

    def test_none_mask_raises_on_detect_objects(self):
        det = ObjectDetector()
        with pytest.raises(ValueError, match="None"):
            det.detect_objects(None)


# ──────────────────────────────────────────────
# Test 5 – clean_mask returns a valid binary mask
# ──────────────────────────────────────────────

class TestCleanMask:
    """Test 5: clean_mask() returns a valid binary mask."""

    def test_clean_mask_binary_values(self):
        """After cleaning, every pixel must be either 0 or 255."""
        det = ObjectDetector()
        # Simulate a raw MOG2 mask that contains shadows (127) and foreground (255)
        raw = np.zeros((100, 100), dtype=np.uint8)
        raw[20:40, 20:40] = 127  # shadow region
        raw[60:80, 60:80] = 255  # foreground region

        clean = det.clean_mask(raw)
        unique_vals = set(np.unique(clean).tolist())
        assert unique_vals.issubset({0, 255}), (
            f"clean_mask produced unexpected pixel values: {unique_vals}"
        )

    def test_clean_mask_shape_preserved(self):
        """clean_mask preserves the spatial dimensions of the input mask."""
        det = ObjectDetector()
        h, w = 180, 240
        raw = np.zeros((h, w), dtype=np.uint8)
        raw[50:130, 80:180] = 255
        clean = det.clean_mask(raw)
        assert clean.shape == (h, w)

    def test_clean_mask_removes_shadow_pixels(self):
        """Shadow pixels (127) must be thresholded to 0 by clean_mask."""
        det = ObjectDetector(shadow_threshold=127)
        shadow_mask = np.full((50, 50), 127, dtype=np.uint8)  # all shadows
        clean = det.clean_mask(shadow_mask)
        # All shadow pixels → background (0)
        assert np.all(clean == 0), "Shadow pixels should be removed (set to 0)."


# ──────────────────────────────────────────────
# Test 6 – detect_objects handles empty mask
# ──────────────────────────────────────────────

class TestDetectObjectsEmptyMask:
    """Test 6: detect_objects() returns empty list for an all-black mask."""

    def test_empty_mask_returns_no_detections(self):
        det = ObjectDetector()
        empty_mask = np.zeros((300, 300), dtype=np.uint8)
        detections = det.detect_objects(empty_mask)
        assert isinstance(detections, list)
        assert len(detections) == 0


# ──────────────────────────────────────────────
# Test 7 – Detects a sufficiently large region
# ──────────────────────────────────────────────

class TestDetectObjectsWithRegion:
    """Test 7: detect_objects() detects a sufficiently large foreground region."""

    def test_large_white_rectangle_detected(self):
        """A 60×60 white rectangle (area = 3600 > MIN_OBJECT_AREA = 500) is found."""
        det = ObjectDetector(min_area=500.0)
        # 60×60 white square at (100,100)
        mask = _white_rect_mask(rect=(100, 100, 60, 60))

        detections = det.detect_objects(mask)
        assert len(detections) >= 1, "Expected at least one detection."

        d = detections[0]
        assert isinstance(d, Detection)
        assert isinstance(d.bbox, BoundingBox)
        assert d.bbox.width > 0
        assert d.bbox.height > 0
        assert d.area > 500.0

        # Centroid should be near the expected centre of the 60×60 square
        expected_cx, expected_cy = 130, 130          # 100 + 60//2
        assert abs(d.centroid[0] - expected_cx) <= 5
        assert abs(d.centroid[1] - expected_cy) <= 5

    def test_detect_pipeline_with_synthetic_foreground(self):
        """
        Verify the full detect() pipeline end-to-end on synthetic data.

        Train the background model on a static black frame (>=20 frames),
        then introduce a white rectangle and confirm it is detected.
        """
        det = ObjectDetector(
            history=15,
            var_threshold=10.0,
            detect_shadows=True,
            min_area=200.0,
            morph_kernel_size=(3, 3),
        )
        bg = _blank_gray()
        # Warm up the background model
        for _ in range(20):
            det.detect(bg)

        # Introduce a 50×50 white foreground blob (area ≈ 2500)
        fg_frame = bg.copy()
        cv2.rectangle(fg_frame, (120, 120), (170, 170), 255, -1)

        detections, clean_mask = det.detect(fg_frame)

        assert isinstance(clean_mask, np.ndarray)
        assert clean_mask.ndim == 2
        assert len(detections) >= 1

        d = detections[0]
        assert d.area > 200.0
        assert isinstance(d.bbox, BoundingBox)

    def test_detection_dataclass_fields(self):
        """Each Detection has the expected fields with correct types."""
        det = ObjectDetector(min_area=100.0)
        mask = _white_rect_mask(rect=(50, 50, 80, 80))   # area = 6400
        detections = det.detect_objects(mask)

        assert len(detections) >= 1
        d = detections[0]
        assert hasattr(d, "bbox")
        assert hasattr(d, "centroid")
        assert hasattr(d, "area")
        assert hasattr(d, "confidence")
        assert isinstance(d.centroid, tuple) and len(d.centroid) == 2


# ──────────────────────────────────────────────
# Test 8 – Small regions are ignored
# ──────────────────────────────────────────────

class TestSmallRegionFiltering:
    """Test 8: Small regions below MIN_OBJECT_AREA are not returned."""

    def test_tiny_blob_excluded(self):
        """A 10×10 blob (area = 100) is below default MIN_OBJECT_AREA=500 → ignored."""
        det = ObjectDetector(min_area=500.0)
        # Draw a small 10×10 square
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (60, 60), 255, -1)

        detections = det.detect_objects(mask)
        assert len(detections) == 0, (
            "Tiny blob (area < min_area) should not produce a Detection."
        )

    def test_large_passes_while_small_filtered(self):
        """Two blobs: only the large one (≥ min_area) is returned."""
        det = ObjectDetector(min_area=500.0)
        mask = np.zeros((300, 300), dtype=np.uint8)

        # Small blob (10×10 = 100 px²) – should be filtered out
        cv2.rectangle(mask, (10, 10), (20, 20), 255, -1)

        # Large blob (60×60 = 3600 px²) – should be detected
        cv2.rectangle(mask, (150, 150), (210, 210), 255, -1)

        detections = det.detect_objects(mask)
        assert len(detections) == 1
        assert detections[0].area >= 500.0

    def test_custom_min_area_accepted(self):
        """A smaller min_area threshold allows smaller blobs through."""
        det = ObjectDetector(min_area=50.0)
        # 12×12 = 144 px² > 50
        mask = np.zeros((200, 200), dtype=np.uint8)
        cv2.rectangle(mask, (30, 30), (42, 42), 255, -1)

        detections = det.detect_objects(mask)
        assert len(detections) >= 1
