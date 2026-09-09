"""
CV-MotionTrack: Real-Time Object Detection, Tracking and Motion Analysis System.

Academic Computer Vision project for CSE3010.
"""

from src.config import (
    AppConfig,
    DetectorConfig,
    MotionAnalysisConfig,
    OpticalFlowConfig,
    PreprocessingConfig,
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
    # Configurations
    "AppConfig",
    "VideoConfig",
    "PreprocessingConfig",
    "DetectorConfig",
    "TrackerConfig",
    "OpticalFlowConfig",
    "MotionAnalysisConfig",
    "VisualizationConfig",
    # Data Models
    "BoundingBox",
    "Detection",
    "TrackedObject",
    "MotionData",
    "FrameMetadata",
]
