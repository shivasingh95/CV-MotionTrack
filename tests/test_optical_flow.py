"""
Unit tests for the optical flow analysis module (src/optical_flow.py).

Academic Coursework: CSE3010 Computer Vision - Step 6

Covers all 10 required evaluation criteria:
1. OpticalFlowAnalyzer initializes correctly.
2. Feature detection works on a synthetic image containing corners.
3. Empty/flat images are handled safely.
4. None input raises ValueError.
5. Incompatible frame dimensions raise ValueError.
6. Optical flow can track synthetic translated feature points.
7. Invalid status points are removed.
8. dx and dy are calculated correctly.
9. Motion magnitude is calculated correctly.
10. Reinitialization works when too few features remain.

Also includes integration test verifying coexistence with Preprocessor,
Detector, and Tracker.
"""

import math
import cv2
import numpy as np
import pytest

from src.config import OpticalFlowConfig
from src.data_models import OpticalFlowPoint
from src.detector import ObjectDetector
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker


def _create_synthetic_corner_image(width: int = 200, height: int = 200) -> np.ndarray:
    """Create a synthetic high-contrast image containing sharp rectangular corners."""
    img = np.zeros((height, width), dtype=np.uint8)
    # Draw high-contrast shapes
    cv2.rectangle(img, (30, 30), (70, 70), 255, -1)
    cv2.rectangle(img, (45, 45), (55, 55), 0, -1)
    cv2.rectangle(img, (120, 100), (170, 160), 200, -1)
    cv2.rectangle(img, (135, 120), (155, 140), 50, -1)
    return img


# ---------------------------------------------------------------------- #
# 1. Initialization Test                                                 #
# ---------------------------------------------------------------------- #

def test_optical_flow_initialization():
    """Verify OpticalFlowAnalyzer initializes with default and custom configurations."""
    analyzer = OpticalFlowAnalyzer()
    assert analyzer.win_size == (21, 21)
    assert analyzer.max_level == 3
    assert analyzer.max_corners == 100
    assert analyzer.quality_level == 0.01
    assert analyzer.min_distance == 7.0
    assert analyzer.min_features == 10
    assert analyzer.prev_gray is None
    assert analyzer.prev_points is None
    assert len(analyzer.flow_points) == 0

    # Custom configuration dataclass
    config = OpticalFlowConfig(
        win_size=(15, 15),
        max_level=2,
        max_corners=50,
        quality_level=0.05,
        min_distance=10.0,
        min_features=5,
    )
    analyzer_custom = OpticalFlowAnalyzer(config=config)
    assert analyzer_custom.win_size == (15, 15)
    assert analyzer_custom.max_level == 2
    assert analyzer_custom.max_corners == 50
    assert analyzer_custom.quality_level == 0.05
    assert analyzer_custom.min_distance == 10.0
    assert analyzer_custom.min_features == 5

    # Direct config object as first argument
    analyzer_direct = OpticalFlowAnalyzer(config)
    assert analyzer_direct.win_size == (15, 15)
    assert analyzer_direct.max_corners == 50


# ---------------------------------------------------------------------- #
# 2. Feature Detection on Synthetic Image with Corners                   #
# ---------------------------------------------------------------------- #

def test_feature_detection_on_synthetic_corners():
    """Verify Shi-Tomasi feature detection on synthetic image containing corners."""
    img = _create_synthetic_corner_image(200, 200)
    analyzer = OpticalFlowAnalyzer(max_corners=50)

    points = analyzer.detect_features(img)
    assert points is not None
    assert isinstance(points, np.ndarray)
    assert points.ndim == 3
    assert points.shape[1] == 1
    assert points.shape[2] == 2
    assert len(points) >= 4  # Should identify multiple distinct corner vertices


# ---------------------------------------------------------------------- #
# 3. Empty and Flat Images Handled Safely                                #
# ---------------------------------------------------------------------- #

def test_empty_and_flat_image_handling():
    """Verify flat/uniform images return None or empty results safely without crashing."""
    analyzer = OpticalFlowAnalyzer()

    # Completely uniform/flat image (no gradients)
    flat_frame = np.full((150, 150), 128, dtype=np.uint8)
    points = analyzer.detect_features(flat_frame)
    assert points is None or len(points) == 0

    # Optical flow update on flat image
    flow_points = analyzer.update(flat_frame, flat_frame)
    assert flow_points == []

    # Frame with zero dimensions raises ValueError
    zero_frame = np.empty((0, 0), dtype=np.uint8)
    with pytest.raises(ValueError):
        analyzer.detect_features(zero_frame)


# ---------------------------------------------------------------------- #
# 4. None Input Validation                                               #
# ---------------------------------------------------------------------- #

def test_none_input_raises_value_error():
    """Verify passing None raises clear ValueError exceptions."""
    analyzer = OpticalFlowAnalyzer()
    valid_frame = np.zeros((100, 100), dtype=np.uint8)

    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.detect_features(None)

    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.initialize(None)

    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.calculate_flow(None, valid_frame, None)

    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.calculate_flow(valid_frame, None, None)

    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.update(None)


# ---------------------------------------------------------------------- #
# 5. Incompatible Dimensions Validation                                  #
# ---------------------------------------------------------------------- #

def test_incompatible_dimensions_raise_value_error():
    """Verify mismatched frame shapes or non-2D images raise clear ValueError."""
    analyzer = OpticalFlowAnalyzer()

    frame_a = np.zeros((100, 100), dtype=np.uint8)
    frame_b = np.zeros((120, 100), dtype=np.uint8)

    # Incompatible dimensions
    with pytest.raises(ValueError, match="Incompatible frame dimensions"):
        analyzer.calculate_flow(frame_a, frame_b, None)

    with pytest.raises(ValueError, match="Incompatible frame dimensions"):
        analyzer.update(frame_a, frame_b)

    # Non-2D image (e.g. 3-channel BGR instead of grayscale)
    color_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="single-channel 2D grayscale"):
        analyzer.detect_features(color_frame)


# ---------------------------------------------------------------------- #
# 6. Tracking Synthetic Translated Feature Points                        #
# ---------------------------------------------------------------------- #

def test_optical_flow_synthetic_translation():
    """Verify Lucas-Kanade correctly tracks translated features between frames."""
    analyzer = OpticalFlowAnalyzer(win_size=(21, 21), max_level=3)

    # Base frame with high contrast corner patterns
    f1 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(f1, (60, 60), (100, 100), 255, -1)
    cv2.rectangle(f1, (75, 75), (85, 85), 0, -1)

    # Translated frame shifted by dx = +5, dy = +3
    shift_x = 5
    shift_y = 3
    f2 = np.zeros((200, 200), dtype=np.uint8)
    cv2.rectangle(
        f2,
        (60 + shift_x, 60 + shift_y),
        (100 + shift_x, 100 + shift_y),
        255,
        -1,
    )
    cv2.rectangle(
        f2,
        (75 + shift_x, 75 + shift_y),
        (85 + shift_x, 85 + shift_y),
        0,
        -1,
    )

    flow_points = analyzer.update(f1, f2)
    assert len(flow_points) > 0

    # Check estimated displacement vectors
    dx_vals = [pt.dx for pt in flow_points]
    dy_vals = [pt.dy for pt in flow_points]

    # Median estimated motion should closely match ground truth (+5, +3)
    assert pytest.approx(np.median(dx_vals), abs=1.0) == 5.0
    assert pytest.approx(np.median(dy_vals), abs=1.0) == 3.0


# ---------------------------------------------------------------------- #
# 7. Invalid Status Points Filtered                                      #
# ---------------------------------------------------------------------- #

def test_invalid_status_points_filtered():
    """Verify points with status == 0 are cleanly removed by filter_valid_points."""
    analyzer = OpticalFlowAnalyzer()

    p0 = np.array([[10, 10], [20, 20], [30, 30], [40, 40]], dtype=np.float32)
    p1 = np.array([[12, 11], [22, 21], [999, 999], [42, 41]], dtype=np.float32)
    status = np.array([[1], [1], [0], [1]], dtype=np.uint8)  # Index 2 lost
    error = np.array([[0.1], [0.2], [99.0], [0.1]], dtype=np.float32)

    valid_p0, valid_p1 = analyzer.filter_valid_points(p0, p1, status, error)

    assert len(valid_p0) == 3
    assert len(valid_p1) == 3
    # Index 2 (lost point at 30, 30) should not be present
    assert np.allclose(valid_p0, np.array([[10, 10], [20, 20], [40, 40]]))
    assert np.allclose(valid_p1, np.array([[12, 11], [22, 21], [42, 41]]))


# ---------------------------------------------------------------------- #
# 8. dx and dy Vector Calculation                                        #
# ---------------------------------------------------------------------- #

def test_dx_and_dy_calculation():
    """Verify differential displacement vectors dx = x_curr - x_prev, dy = y_curr - y_prev."""
    analyzer = OpticalFlowAnalyzer()

    p0 = np.array([[50.0, 100.0], [200.0, 150.0]], dtype=np.float32)
    p1 = np.array([[56.5, 96.0], [190.0, 175.0]], dtype=np.float32)

    points = analyzer.create_flow_points(p0, p1)
    assert len(points) == 2

    # Point 1: (50, 100) -> (56.5, 96.0) => dx = +6.5, dy = -4.0
    assert pytest.approx(points[0].previous_x) == 50.0
    assert pytest.approx(points[0].previous_y) == 100.0
    assert pytest.approx(points[0].current_x) == 56.5
    assert pytest.approx(points[0].current_y) == 96.0
    assert pytest.approx(points[0].dx) == 6.5
    assert pytest.approx(points[0].dy) == -4.0
    assert points[0].vector == (6.5, -4.0)

    # Point 2: (200, 150) -> (190, 175) => dx = -10.0, dy = +25.0
    assert pytest.approx(points[1].dx) == -10.0
    assert pytest.approx(points[1].dy) == 25.0


# ---------------------------------------------------------------------- #
# 9. Motion Magnitude Calculation                                        #
# ---------------------------------------------------------------------- #

def test_motion_magnitude_calculation():
    """Verify Euclidean magnitude = sqrt(dx^2 + dy^2)."""
    # 8-15-17 Pythagorean triangle
    pt = OpticalFlowPoint(
        previous_x=10.0,
        previous_y=20.0,
        current_x=18.0,
        current_y=35.0,
        dx=8.0,
        dy=15.0,
    )
    expected_magnitude = math.hypot(8.0, 15.0)  # sqrt(64 + 225) = sqrt(289) = 17.0
    assert pytest.approx(pt.magnitude, rel=1e-5) == 17.0
    assert pt.magnitude == expected_magnitude

    # Stationary point
    zero_pt = OpticalFlowPoint(5.0, 5.0, 5.0, 5.0, 0.0, 0.0)
    assert zero_pt.magnitude == 0.0


# ---------------------------------------------------------------------- #
# 10. Reinitialization When Features Drop Below Threshold                #
# ---------------------------------------------------------------------- #

def test_reinitialization_below_feature_threshold():
    """Verify automatic re-detection when tracked features drop below min_features."""
    min_features = 10
    analyzer = OpticalFlowAnalyzer(min_features=min_features, max_corners=50)

    f1 = _create_synthetic_corner_image(200, 200)
    analyzer.initialize(f1)
    assert analyzer.prev_points is not None
    initial_count = len(analyzer.prev_points)
    assert initial_count >= min_features

    # Artificially deplete active tracking points below threshold
    analyzer.prev_points = analyzer.prev_points[:3]  # Only 3 points remaining (< 10)
    assert len(analyzer.prev_points) < min_features

    # Sequential update with new frame should trigger automatic re-detection
    flow_points = analyzer.update(f1)
    assert analyzer.prev_points is not None
    # Feature count should have been replenished back above threshold
    assert len(analyzer.prev_points) >= min_features


# ---------------------------------------------------------------------- #
# Integration Test: Pipeline Coexistence                                 #
# ---------------------------------------------------------------------- #

def test_pipeline_integration_coexistence():
    """
    Verify Preprocessor -> Detector -> Tracker -> OpticalFlowAnalyzer
    coexist harmoniously without breaking each other.
    """
    preprocessor = Preprocessor()
    detector = ObjectDetector(min_area=50.0)
    tracker = ObjectTracker()
    optical_flow = OpticalFlowAnalyzer()

    # Synthetic moving square across 3 frames
    frames = []
    for offset in [0, 10, 20]:
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        cv2.rectangle(img, (50 + offset, 50), (90 + offset, 90), (255, 255, 255), -1)
        frames.append(img)

    for frame in frames:
        # 1. Preprocessing
        gray = preprocessor.process(frame)
        assert gray.ndim == 2

        # 2. Detection
        detections, fg_mask = detector.detect(gray)
        assert isinstance(detections, list)

        # 3. Tracking
        tracks = tracker.update(detections)
        assert isinstance(tracks, list)

        # 4. Optical Flow
        flow_pts = optical_flow.update(gray)
        assert isinstance(flow_pts, list)

    assert tracker.next_object_id >= 1
