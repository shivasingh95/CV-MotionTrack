"""
Configuration module for CV-MotionTrack.

Provides centralized configuration structures for video ingestion, preprocessing,
object detection, tracking, optical flow, motion analysis, visualization, and output recording.
Avoids hardcoded magic numbers throughout the system.

Organized by subsystem:
1. INPUT: Video capture and decoding
2. PREPROCESSING: Grayscale conversion and Gaussian spatial filtering
3. BACKGROUND SUBTRACTION & DETECTION: MOG2/KNN background model and contour extraction
4. TRACKING: Centroid-based identity association and trajectory recording
5. OPTICAL FLOW: Sparse Lucas-Kanade and Shi-Tomasi feature tracking
6. MOTION ANALYSIS: Image-space displacement, direction, velocity, and smoothing
7. VISUALIZATION: Bounding boxes, IDs, trajectories, optical flow, and statistics HUD
8. OUTPUT: Video recording and frame snapshot exporting
"""

from dataclasses import dataclass, field
import os
from typing import Optional, Tuple, Union


# ==============================================================================
# 1. INPUT CONFIGURATION DEFAULTS
# ==============================================================================
DEFAULT_INPUT_SOURCE: Union[int, str] = 0
DEFAULT_TARGET_WIDTH: int = 640
DEFAULT_TARGET_HEIGHT: int = 480
DEFAULT_FALLBACK_FPS: float = 30.0

# ==============================================================================
# 2. PREPROCESSING HYPERPARAMETER DEFAULTS
# ==============================================================================
GAUSSIAN_KERNEL_SIZE: Tuple[int, int] = (5, 5)
GAUSSIAN_SIGMA: float = 0.0

# ==============================================================================
# 3. DETECTOR & BACKGROUND SUBTRACTION DEFAULTS
# ==============================================================================
BACKGROUND_HISTORY: int = 500
BACKGROUND_VAR_THRESHOLD: float = 16.0
DETECT_SHADOWS: bool = True
SHADOW_PIXEL_VALUE: int = 127        # OpenCV MOG2 marks shadows as 127
MORPH_KERNEL_SIZE: Tuple[int, int] = (3, 3)
MORPH_OPEN_ITERATIONS: int = 1
MORPH_CLOSE_ITERATIONS: int = 2
MIN_OBJECT_AREA: float = 500.0
MAX_OBJECT_AREA: float = 100_000.0

# ==============================================================================
# 4. TRACKER HYPERPARAMETER DEFAULTS
# ==============================================================================
TRACKER_MAX_DISTANCE: float = 50.0   # px – max centroid distance to accept a match
TRACKER_MAX_DISAPPEARED: int = 10    # frames an object may be undetected before removal
MAX_TRAJECTORY_LENGTH: int = 100     # max historical centroid positions stored per object

# ==============================================================================
# 5. OPTICAL FLOW HYPERPARAMETER DEFAULTS
# ==============================================================================
OPTICAL_FLOW_WIN_SIZE: Tuple[int, int] = (21, 21)
OPTICAL_FLOW_MAX_LEVEL: int = 3
OPTICAL_FLOW_MAX_CORNERS: int = 100
OPTICAL_FLOW_QUALITY_LEVEL: float = 0.01
OPTICAL_FLOW_MIN_DISTANCE: float = 7.0
OPTICAL_FLOW_MIN_FEATURES: int = 10

# ==============================================================================
# 6. MOTION ANALYSIS HYPERPARAMETER DEFAULTS
# ==============================================================================
MOTION_SMOOTHING_WINDOW: int = 5
STATIONARY_THRESHOLD: float = 1.0
MOTION_HISTORY_LENGTH: int = 30

# ==============================================================================
# 7. VISUALIZATION DEFAULTS
# ==============================================================================
COLOR_BBOX: Tuple[int, int, int] = (0, 255, 0)          # BGR Green
COLOR_CENTROID: Tuple[int, int, int] = (0, 0, 255)      # BGR Red
COLOR_TRAJECTORY: Tuple[int, int, int] = (0, 255, 255)  # BGR Yellow
COLOR_OPTICAL_FLOW: Tuple[int, int, int] = (255, 255, 0)# BGR Cyan
COLOR_VECTOR: Tuple[int, int, int] = (255, 0, 255)      # BGR Magenta
COLOR_TEXT: Tuple[int, int, int] = (255, 255, 255)        # BGR White

# ==============================================================================
# 8. OUTPUT DEFAULTS
# ==============================================================================
SAVE_OUTPUT_VIDEO: bool = False
OUTPUT_VIDEO_PATH: Optional[str] = None
OUTPUT_CODEC: str = "XVID"
OUTPUT_FPS: float = 25.0
DEFAULT_SCREENSHOT_DIR: str = os.path.join("results", "screenshots")


# ==============================================================================
# CONFIGURATION DATACLASSES
# ==============================================================================

@dataclass
class VideoConfig:
    """Settings for video acquisition and decoding."""
    source: Union[int, str] = DEFAULT_INPUT_SOURCE
    target_width: int = DEFAULT_TARGET_WIDTH
    target_height: int = DEFAULT_TARGET_HEIGHT
    enforce_target_size: bool = False
    fps_limit: float = DEFAULT_FALLBACK_FPS


@dataclass
class PreprocessingConfig:
    """Settings for initial frame preprocessing and spatial noise filtering."""
    to_grayscale: bool = True
    gaussian_kernel_size: Tuple[int, int] = GAUSSIAN_KERNEL_SIZE
    gaussian_sigma_x: float = GAUSSIAN_SIGMA
    gaussian_sigma_y: float = GAUSSIAN_SIGMA
    normalize: bool = False


@dataclass
class DetectorConfig:
    """Settings for background modeling, subtraction, and contour detection."""
    subtractor_type: str = "MOG2"
    history: int = BACKGROUND_HISTORY
    var_threshold: float = BACKGROUND_VAR_THRESHOLD
    detect_shadows: bool = DETECT_SHADOWS
    shadow_threshold: int = SHADOW_PIXEL_VALUE
    min_contour_area: float = MIN_OBJECT_AREA
    max_contour_area: float = MAX_OBJECT_AREA
    morph_kernel_size: Tuple[int, int] = MORPH_KERNEL_SIZE
    morph_open_iterations: int = MORPH_OPEN_ITERATIONS
    morph_close_iterations: int = MORPH_CLOSE_ITERATIONS


@dataclass
class TrackerConfig:
    """Settings for centroid-based multi-object tracking and trajectory maintenance."""
    max_distance: float = TRACKER_MAX_DISTANCE
    max_disappeared: int = TRACKER_MAX_DISAPPEARED
    max_trajectory_length: int = MAX_TRAJECTORY_LENGTH
    max_distance_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.max_distance_threshold is not None:
            self.max_distance = self.max_distance_threshold
        else:
            self.max_distance_threshold = self.max_distance


@dataclass
class OpticalFlowConfig:
    """Settings for sparse Lucas-Kanade optical flow feature tracking."""
    win_size: Tuple[int, int] = OPTICAL_FLOW_WIN_SIZE
    max_level: int = OPTICAL_FLOW_MAX_LEVEL
    max_corners: int = OPTICAL_FLOW_MAX_CORNERS
    quality_level: float = OPTICAL_FLOW_QUALITY_LEVEL
    min_distance: float = OPTICAL_FLOW_MIN_DISTANCE
    min_features: int = OPTICAL_FLOW_MIN_FEATURES
    block_size: int = 7

    # Compatibility aliases for legacy interfaces
    @property
    def feature_max_corners(self) -> int:
        return self.max_corners

    @property
    def feature_quality_level(self) -> float:
        return self.quality_level

    @property
    def feature_min_distance(self) -> float:
        return self.min_distance

    @property
    def feature_block_size(self) -> int:
        return self.block_size

    @property
    def lk_win_size(self) -> Tuple[int, int]:
        return self.win_size

    @property
    def lk_max_level(self) -> int:
        return self.max_level


@dataclass
class MotionAnalysisConfig:
    """Settings for kinematic parameter estimation and spatio-temporal tracking."""
    fps: float = DEFAULT_FALLBACK_FPS
    smoothing_window: int = MOTION_SMOOTHING_WINDOW
    stationary_threshold: float = STATIONARY_THRESHOLD
    history_length: int = MOTION_HISTORY_LENGTH
    min_displacement_threshold: float = STATIONARY_THRESHOLD
    speed_smoothing_window: int = MOTION_SMOOTHING_WINDOW


@dataclass
class VisualizationConfig:
    """Settings for rendering detections, tracks, vectors, and overlay HUD."""
    show_bboxes: bool = True
    show_centroids: bool = True
    show_ids: bool = True
    show_trajectories: bool = True
    show_optical_flow: bool = True
    show_motion_info: bool = True
    show_motion_vectors: bool = True
    show_dashboard: bool = True
    bbox_color: Tuple[int, int, int] = COLOR_BBOX
    centroid_color: Tuple[int, int, int] = COLOR_CENTROID
    trajectory_color: Tuple[int, int, int] = COLOR_TRAJECTORY
    optical_flow_color: Tuple[int, int, int] = COLOR_OPTICAL_FLOW
    vector_color: Tuple[int, int, int] = COLOR_VECTOR
    text_color: Tuple[int, int, int] = COLOR_TEXT
    line_thickness: int = 2
    font_scale: float = 0.5
    max_displayed_flow_points: int = 150


@dataclass
class OutputConfig:
    """Settings for exporting annotated video and screenshots."""
    save_video: bool = SAVE_OUTPUT_VIDEO
    output_path: Optional[str] = OUTPUT_VIDEO_PATH
    codec: str = OUTPUT_CODEC
    fps: float = OUTPUT_FPS
    screenshot_dir: str = DEFAULT_SCREENSHOT_DIR


@dataclass
class AppConfig:
    """Root configuration aggregating all subsystem settings."""
    video: VideoConfig = field(default_factory=VideoConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    optical_flow: OpticalFlowConfig = field(default_factory=OpticalFlowConfig)
    motion: MotionAnalysisConfig = field(default_factory=MotionAnalysisConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
