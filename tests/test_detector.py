"""
Unit tests for the object detection module.

Verifies:
- ObjectDetector class import and initialization with MOG2 and KNN backends.
- Input validation on empty frames and masks.
- Foreground segmentation mask generation and data structure contracts.
- Detection of synthetic moving foreground regions.
"""

import cv2
import numpy as np
import pytest

from src.config import DetectorConfig
from src.data_models import Detection
from src.detector import ObjectDetector


def test_detector_initialization():
    """Verify ObjectDetector initializes with default and custom backends."""
    detector_mog2 = ObjectDetector()
    assert detector_mog2.subtractor is not None

    knn_cfg = DetectorConfig(subtractor_type="KNN")
    detector_knn = ObjectDetector(config=knn_cfg)
    assert detector_knn.subtractor is not None


def test_detector_invalid_input():
    """Verify ObjectDetector validates input arguments."""
    detector = ObjectDetector()

    with pytest.raises(ValueError, match="empty"):
        detector.apply_background_subtraction(None)

    with pytest.raises(ValueError, match="empty"):
        detector.remove_noise(None)


def test_detector_output_contract():
    """Verify detect() returns expected types: Tuple[List[Detection], np.ndarray]."""
    detector = ObjectDetector()
    dummy_frame = np.zeros((200, 300), dtype=np.uint8)

    detections, mask = detector.detect(dummy_frame)
    assert isinstance(detections, list)
    assert isinstance(mask, np.ndarray)
    assert mask.shape == (200, 300)


def test_synthetic_blob_detection():
    """Verify that a high-contrast foreground square produces a valid Detection."""
    cfg = DetectorConfig(
        history=10,
        var_threshold=16.0,
        min_contour_area=50.0,
        morph_open_iterations=0,
        morph_close_iterations=0,
    )
    detector = ObjectDetector(config=cfg)

    # Train background model on static black frame
    background = np.zeros((300, 300), dtype=np.uint8)
    for _ in range(15):
        detector.detect(background)

    # Introduce a white foreground square (size 40x40 = 1600 area)
    frame_with_blob = background.copy()
    cv2.rectangle(frame_with_blob, (100, 100), (140, 140), 255, -1)

    detections, _ = detector.detect(frame_with_blob)
    assert len(detections) >= 1
    det = detections[0]
    assert isinstance(det, Detection)
    assert det.bbox.width > 0
    assert det.bbox.height > 0
    # Centroid should be approximately (120, 120)
    assert abs(det.centroid[0] - 120) <= 5
    assert abs(det.centroid[1] - 120) <= 5
