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
    MIN_OBJECT_AREA,
    MORPH_CLOSE_ITERATIONS,
    MORPH_KERNEL_SIZE,
    MORPH_OPEN_ITERATIONS,
    MotionAnalysisConfig,
    OpticalFlowConfig,
    PreprocessingConfig,
    SHADOW_PIXEL_VALUE,
    TrackerConfig,
    VideoConfig,
    VisualizationConfig,
)
from src.data_models import (
    BoundingBox,
    Detection,
    FrameMetadata,
    MotionData,
    TrackedObject,
)
from src.detector import ObjectDetector
from src.motion_analysis import MotionAnalyzer
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer

__version__ = "0.1.0"

__all__ = [
    # Subsystems
    "VideoProcessor",
    "Preprocessor",
    "ObjectDetector",
    "ObjectTracker",
    "OpticalFlowAnalyzer",
    "MotionAnalyzer",
    "Visualizer",
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
    # Data Models
    "BoundingBox",
    "Detection",
    "TrackedObject",
    "MotionData",
    "FrameMetadata",
]
