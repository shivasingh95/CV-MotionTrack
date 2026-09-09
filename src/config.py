"""
Configuration module for CV-MotionTrack.

Provides centralized configuration structures for video ingestion, preprocessing,
object detection, tracking, optical flow, motion analysis, and visualization.
Avoids hardcoded hyperparameters throughout the system.
"""

from dataclasses import dataclass, field
from typing import Tuple, Union


# Preprocessing Hyperparameter Defaults
GAUSSIAN_KERNEL_SIZE: Tuple[int, int] = (5, 5)
GAUSSIAN_SIGMA: float = 0.0


@dataclass
class VideoConfig:
    """Settings for video acquisition and decoding."""
    # Video input source: integer for webcam device index (e.g. 0), or str path to video file
    source: Union[int, str] = 0
    target_width: int = 640
    target_height: int = 480
    enforce_target_size: bool = False
    fps_limit: float = 30.0


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
    # Background subtraction algorithm: 'MOG2', 'KNN', or 'FRAME_DIFF'
    subtractor_type: str = "MOG2"
    history: int = 500
    var_threshold: float = 25.0
    detect_shadows: bool = True
    shadow_threshold: int = 127
    min_contour_area: float = 500.0
    max_contour_area: float = 100000.0
    morph_kernel_size: Tuple[int, int] = (3, 3)
    morph_open_iterations: int = 1
    morph_close_iterations: int = 2


@dataclass
class TrackerConfig:
    """Settings for multi-object association and trajectory maintenance."""
    max_disappeared: int = 25
    max_distance_threshold: float = 60.0
    max_trajectory_length: int = 50


@dataclass
class OpticalFlowConfig:
    """Settings for sparse Lucas-Kanade optical flow feature tracking."""
    feature_max_corners: int = 100
    feature_quality_level: float = 0.03
    feature_min_distance: float = 10.0
    feature_block_size: int = 7
    lk_win_size: Tuple[int, int] = (15, 15)
    lk_max_level: int = 2


@dataclass
class MotionAnalysisConfig:
    """Settings for kinematic parameter estimation and spatio-temporal tracking."""
    fps: float = 30.0
    min_displacement_threshold: float = 2.0
    speed_smoothing_window: int = 5


@dataclass
class VisualizationConfig:
    """Settings for rendering detections, tracks, vectors, and overlay HUD."""
    show_bboxes: bool = True
    show_centroids: bool = True
    show_ids: bool = True
    show_trajectories: bool = True
    show_motion_vectors: bool = True
    show_dashboard: bool = True
    bbox_color: Tuple[int, int, int] = (0, 255, 0)         # BGR Green
    centroid_color: Tuple[int, int, int] = (0, 0, 255)     # BGR Red
    trajectory_color: Tuple[int, int, int] = (0, 255, 255) # BGR Yellow
    vector_color: Tuple[int, int, int] = (255, 0, 255)     # BGR Magenta
    text_color: Tuple[int, int, int] = (255, 255, 255)     # BGR White
    line_thickness: int = 2
    font_scale: float = 0.5


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
