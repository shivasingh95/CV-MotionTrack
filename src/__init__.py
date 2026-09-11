"""
CV-MotionTrack: Real-Time Object Detection, Tracking and Motion Analysis System.

Academic Computer Vision project for CSE3010.
"""

from src.config import (
    AppConfig,
    BACKGROUND_HISTORY,
    BACKGROUND_VAR_THRESHOLD,
    DETECT_SHADOWS,
    DetectorConfig,
    GAUSSIAN_KERNEL_SIZE,
    GAUSSIAN_SIGMA,
    MAX_OBJECT_AREA,
    MAX_TRAJECTORY_LENGTH,
    MIN_OBJECT_AREA,
    MORPH_CLOSE_ITERATIONS,
    MORPH_KERNEL_SIZE,
    MORPH_OPEN_ITERATIONS,
    MOTION_HISTORY_LENGTH,
    MOTION_SMOOTHING_WINDOW,
    MotionAnalysisConfig,
    OPTICAL_FLOW_MAX_CORNERS,
    OPTICAL_FLOW_MAX_LEVEL,
    OPTICAL_FLOW_MIN_DISTANCE,
    OPTICAL_FLOW_MIN_FEATURES,
    OPTICAL_FLOW_QUALITY_LEVEL,
    OPTICAL_FLOW_WIN_SIZE,
    OpticalFlowConfig,
    PreprocessingConfig,
    SHADOW_PIXEL_VALUE,
    STATIONARY_THRESHOLD,
    TRACKER_MAX_DISAPPEARED,
    TRACKER_MAX_DISTANCE,
    TrackerConfig,
    OutputConfig,
    VideoConfig,
    VisualizationConfig,
)
from src.data_models import (
    BoundingBox,
    Detection,
    FrameMetadata,
    MotionData,
    OpticalFlowPoint,
    TrackedObject,
)
from src.detector import ObjectDetector
from src.evaluation import (
    EvaluationManager,
    EvaluationSummary,
    Evaluator,
    FrameMetrics,
)
from src.motion_analysis import Displacement, MotionAnalyzer
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker
from src.ui import CVMotionTrackApp
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer

__version__ = "0.1.0"

__all__ = [
    # Subsystems
    "CVMotionTrackApp",
    "VideoProcessor",
    "Preprocessor",
    "ObjectDetector",
    "ObjectTracker",
    "OpticalFlowAnalyzer",
    "MotionAnalyzer",
    "Visualizer",
    "Evaluator",
    "EvaluationManager",
    "EvaluationSummary",
    "FrameMetrics",
    # Configurations and Defaults
    "AppConfig",
    "VideoConfig",
    "PreprocessingConfig",
    "DetectorConfig",
    "TrackerConfig",
    "OpticalFlowConfig",
    "MotionAnalysisConfig",
    "VisualizationConfig",
    "GAUSSIAN_KERNEL_SIZE",
    "GAUSSIAN_SIGMA",
    "BACKGROUND_HISTORY",
    "BACKGROUND_VAR_THRESHOLD",
    "DETECT_SHADOWS",
    "SHADOW_PIXEL_VALUE",
    "MORPH_KERNEL_SIZE",
    "MORPH_OPEN_ITERATIONS",
    "MORPH_CLOSE_ITERATIONS",
    "MIN_OBJECT_AREA",
    "MAX_OBJECT_AREA",
    "TRACKER_MAX_DISTANCE",
    "TRACKER_MAX_DISAPPEARED",
    "MAX_TRAJECTORY_LENGTH",
    "OPTICAL_FLOW_WIN_SIZE",
    "OPTICAL_FLOW_MAX_LEVEL",
    "OPTICAL_FLOW_MAX_CORNERS",
    "OPTICAL_FLOW_QUALITY_LEVEL",
    "OPTICAL_FLOW_MIN_DISTANCE",
    "OPTICAL_FLOW_MIN_FEATURES",
    "MOTION_SMOOTHING_WINDOW",
    "STATIONARY_THRESHOLD",
    "MOTION_HISTORY_LENGTH",
    # Data Models
    "BoundingBox",
    "Detection",
    "TrackedObject",
    "MotionData",
    "OpticalFlowPoint",
    "FrameMetadata",
    "Displacement",
    "OutputConfig",
]
