"""
Integration tests for the CV-MotionTrack pipeline (Step 8).

Verifies the end-to-end coordination of all modules:
1. All modules initialize together from configuration.
2. A synthetic frame sequence passes through the complete pipeline without errors.
3. Empty detections (e.g., static background) do not crash the pipeline.
4. Absence of optical-flow features (flat textureless frames) does not crash the pipeline.
5. Reset functionality clears state across tracker, motion analyzer, and optical flow analyzer.
6. Configuration loads correctly and propagates settings.
7. VideoProcessor handles synthetic video files, missing files, and property queries.
8. Visualizer rendering methods function correctly with simulated data.
"""

import os
import tempfile
import cv2
import numpy as np
import pytest

from src.config import (
    AppConfig,
    DetectorConfig,
    MotionAnalysisConfig,
    OpticalFlowConfig,
    OutputConfig,
    PreprocessingConfig,
    TrackerConfig,
    VideoConfig,
    VisualizationConfig,
)
from src.data_models import BoundingBox, Detection, MotionData, OpticalFlowPoint, TrackedObject
from src.detector import ObjectDetector
from src.motion_analysis import MotionAnalyzer
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer


# -------------------------------------------------------------------------- #
# 1. Pipeline Subsystems Initialize Together                                 #
# -------------------------------------------------------------------------- #

def test_modules_initialize_together():
    """Verify that all pipeline subsystems instantiate harmoniously from AppConfig."""
    config = AppConfig()

    video_processor = VideoProcessor(config=config.video)
    preprocessor = Preprocessor(config=config.preprocessing)
    detector = ObjectDetector(config=config.detector)
    tracker = ObjectTracker(config=config.tracker)
    optical_flow = OpticalFlowAnalyzer(config=config.optical_flow)
    motion_analyzer = MotionAnalyzer(config=config.motion)
    visualizer = Visualizer(config=config.visualization)

    assert video_processor is not None
    assert preprocessor is not None
    assert detector is not None
    assert tracker is not None
    assert optical_flow is not None
    assert motion_analyzer is not None
    assert visualizer is not None


# -------------------------------------------------------------------------- #
# 2. Synthetic Frame Sequence Passes End-to-End                             #
# -------------------------------------------------------------------------- #

def test_synthetic_frame_end_to_end_pipeline():
    """Verify that synthetic moving frames pass through the complete pipeline."""
    config = AppConfig()
    config.detector.history = 10
    config.detector.var_threshold = 16.0
    config.detector.min_contour_area = 50.0

    preprocessor = Preprocessor(config=config.preprocessing)
    detector = ObjectDetector(config=config.detector)
    tracker = ObjectTracker(config=config.tracker)
    optical_flow = OpticalFlowAnalyzer(config=config.optical_flow)
    motion_analyzer = MotionAnalyzer(config=config.motion)
    visualizer = Visualizer(config=config.visualization)

    h, w = 240, 320

    # Step 1: Prime the background model with static background frames
    bg_frame = np.full((h, w, 3), 40, dtype=np.uint8)
    for _ in range(12):
        gray = preprocessor.process(bg_frame)
        _ = detector.detect(gray)

    # Step 2: Feed moving object frames
    # Object moves from x=50 to x=80
    for step in range(4):
        frame = bg_frame.copy()
        obj_x = 50 + step * 10
        obj_y = 50 + step * 5
        cv2.rectangle(frame, (obj_x, obj_y), (obj_x + 30, obj_y + 30), (220, 220, 220), -1)

        # Preprocessing
        gray_frame = preprocessor.process(frame)
        assert gray_frame.ndim == 2
        assert gray_frame.shape == (h, w)

        # Detection & Background Subtraction
        detections, fg_mask = detector.detect(gray_frame)
        assert isinstance(detections, list)
        assert fg_mask.shape == (h, w)

        # Object Tracking
        tracks = tracker.update(detections)
        assert isinstance(tracks, list)

        # Optical Flow
        flow_points = optical_flow.update(gray_frame)
        assert isinstance(flow_points, list)

        # Motion Analysis
        motion_data = motion_analyzer.update(tracks, flow_points)
        assert isinstance(motion_data, dict)

        # Visualization Composite
        stats = motion_analyzer.get_summary_statistics()
        annotated = visualizer.render(
            frame=frame,
            detections=detections,
            tracked_objects=tracks,
            motion_data=motion_data,
            optical_flow_points=flow_points,
            fps=30.0,
            stats=stats,
        )

        assert annotated.shape == (h, w, 3)
        assert annotated.dtype == np.uint8


# -------------------------------------------------------------------------- #
# 3. Empty Detections Do Not Crash Pipeline                                  #
# -------------------------------------------------------------------------- #

def test_empty_detections_do_not_crash_pipeline():
    """Verify that frames with zero detected objects execute smoothly without exceptions."""
    config = AppConfig()
    preprocessor = Preprocessor(config=config.preprocessing)
    detector = ObjectDetector(config=config.detector)
    tracker = ObjectTracker(config=config.tracker)
    optical_flow = OpticalFlowAnalyzer(config=config.optical_flow)
    motion_analyzer = MotionAnalyzer(config=config.motion)
    visualizer = Visualizer(config=config.visualization)

    # Static uniform frame: no foreground movement
    static_frame = np.full((120, 160, 3), 100, dtype=np.uint8)

    for _ in range(3):
        gray = preprocessor.process(static_frame)
        detections, fg_mask = detector.detect(gray)
        tracks = tracker.update(detections)
        flow_points = optical_flow.update(gray)
        motion_data = motion_analyzer.update(tracks, flow_points)
        stats = motion_analyzer.get_summary_statistics()

        annotated = visualizer.render(
            frame=static_frame,
            detections=detections,
            tracked_objects=tracks,
            motion_data=motion_data,
            optical_flow_points=flow_points,
            fps=25.0,
            stats=stats,
        )

        assert len(detections) == 0
        assert len(tracks) == 0
        assert len(motion_data) == 0
        assert annotated.shape == static_frame.shape


# -------------------------------------------------------------------------- #
# 4. No Optical Flow Points Does Not Crash Pipeline                          #
# -------------------------------------------------------------------------- #

def test_no_optical_flow_points_does_not_crash_pipeline():
    """Verify that completely flat images with zero trackable corners do not cause crashes."""
    optical_flow = OpticalFlowAnalyzer()
    visualizer = Visualizer()

    # Perfectly flat image has 0 corners
    flat_frame = np.zeros((100, 100), dtype=np.uint8)
    flow_points = optical_flow.update(flat_frame)
    assert len(flow_points) == 0

    # Test visualizer with empty flow list
    bgr_frame = np.zeros((100, 100, 3), dtype=np.uint8)
    annotated = visualizer.draw_optical_flow(bgr_frame, flow_points)
    assert annotated.shape == bgr_frame.shape


# -------------------------------------------------------------------------- #
# 5. Reset Functionality Works Across Subsystems                             #
# -------------------------------------------------------------------------- #

def test_reset_functionality_works():
    """Verify that tracker, motion analyzer, and optical flow analyzer reset state cleanly."""
    tracker = ObjectTracker()
    motion_analyzer = MotionAnalyzer()
    optical_flow = OpticalFlowAnalyzer()

    # Populate tracker with a dummy detection
    det = Detection(
        bbox=BoundingBox(x=10, y=10, width=20, height=20),
        centroid=(20, 20),
        confidence=1.0,
        area=400.0,
    )
    tracks = tracker.update([det])
    assert len(tracks) == 1
    assert tracker.next_object_id == 2

    # Populate motion analyzer
    motion_data = motion_analyzer.update(tracks)
    assert len(motion_data) == 1
    assert len(motion_analyzer.last_active_objects) == 1

    # Populate optical flow analyzer
    textured_frame = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    _ = optical_flow.initialize(textured_frame)
    assert optical_flow.prev_gray is not None

    # Perform Reset
    tracker.reset()
    motion_analyzer.reset()
    optical_flow.reset()

    # Validate Tracker reset
    assert tracker.active_count == 0
    assert tracker.next_object_id == 1
    assert len(tracker.objects) == 0

    # Validate MotionAnalyzer reset
    assert len(motion_analyzer.motion_history) == 0
    assert len(motion_analyzer.recent_velocities) == 0
    assert len(motion_analyzer.recent_displacements) == 0
    assert len(motion_analyzer.last_active_objects) == 0

    # Validate OpticalFlow reset
    assert optical_flow.prev_gray is None
    assert optical_flow.prev_points is None
    assert len(optical_flow.flow_points) == 0


# -------------------------------------------------------------------------- #
# 6. Configuration Loads and Propagates Correctly                            #
# -------------------------------------------------------------------------- #

def test_configuration_loads_and_propagates():
    """Verify that configuration classes initialize properly with customized parameters."""
    cfg = AppConfig(
        video=VideoConfig(target_width=1280, target_height=720, fps_limit=60.0),
        detector=DetectorConfig(var_threshold=32.0, min_contour_area=150.0),
        tracker=TrackerConfig(max_distance=75.0, max_disappeared=15),
        optical_flow=OpticalFlowConfig(max_corners=200, quality_level=0.02),
        motion=MotionAnalysisConfig(smoothing_window=7, stationary_threshold=2.0),
        visualization=VisualizationConfig(line_thickness=3),
        output=OutputConfig(save_video=True, codec="MJPG", fps=30.0),
    )

    assert cfg.video.target_width == 1280
    assert cfg.video.fps_limit == 60.0
    assert cfg.detector.var_threshold == 32.0
    assert cfg.tracker.max_distance == 75.0
    assert cfg.optical_flow.max_corners == 200
    assert cfg.motion.smoothing_window == 7
    assert cfg.motion.stationary_threshold == 2.0
    assert cfg.visualization.line_thickness == 3
    assert cfg.output.save_video is True
    assert cfg.output.codec == "MJPG"


# -------------------------------------------------------------------------- #
# 7. VideoProcessor Handling of Synthetic Videos and Missing Sources         #
# -------------------------------------------------------------------------- #

def test_video_processor_file_and_missing_source():
    """Verify VideoProcessor opens synthetic video files and handles missing files gracefully."""
    # 1. Missing file raises FileNotFoundError
    with pytest.raises(FileNotFoundError):
        vp_missing = VideoProcessor(source="non_existent_file_path_xyz.mp4")
        vp_missing.open()

    # 2. Synthetic video write and read
    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_video_path = os.path.join(tmp_dir, "synthetic_test.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(temp_video_path, fourcc, 20.0, (160, 120))

        # Write 5 frames
        for i in range(5):
            f = np.full((120, 160, 3), i * 40, dtype=np.uint8)
            writer.write(f)
        writer.release()

        # Read back using VideoProcessor context manager
        with VideoProcessor(source=temp_video_path) as vp:
            assert vp.is_opened()
            props = vp.get_properties()
            assert props["is_opened"] is True
            assert vp.width == 160
            assert vp.height == 120

            frames_read = 0
            while True:
                ret, frame = vp.read_frame()
                if not ret:
                    break
                frames_read += 1
                assert frame.shape == (120, 160, 3)

            assert frames_read == 5
            # Read after EOF returns False, None
            ret_eof, frame_eof = vp.read_frame()
            assert ret_eof is False
            assert frame_eof is None

        # After context exit, resources are released
        assert not vp.is_opened()


# -------------------------------------------------------------------------- #
# 8. Visualizer Components Render Without Exceptions                         #
# -------------------------------------------------------------------------- #

def test_visualizer_components():
    """Verify each individual visualizer method executes and decorates the canvas."""
    vis = Visualizer()
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)

    # 1. draw_detections
    det = Detection(bbox=BoundingBox(10, 10, 30, 30), centroid=(25, 25), confidence=0.9, area=900.0)
    out1 = vis.draw_detections(canvas, [det])
    assert out1.shape == canvas.shape

    # 2. draw_tracks
    track = TrackedObject(
        object_id=1,
        current_position=(50, 50),
        previous_position=(45, 45),
        trajectory=[(40, 40), (45, 45), (50, 50)],
        bbox=BoundingBox(35, 35, 30, 30),
    )
    out2 = vis.draw_tracks(canvas, [track])
    assert out2.shape == canvas.shape

    # 3. draw_trajectories
    out3 = vis.draw_trajectories(canvas, [track])
    assert out3.shape == canvas.shape

    # 4. draw_optical_flow
    flow_pt = OpticalFlowPoint(previous_x=30.0, previous_y=30.0, current_x=35.0, current_y=35.0, dx=5.0, dy=5.0)
    out4 = vis.draw_optical_flow(canvas, [flow_pt])
    assert out4.shape == canvas.shape

    # 5. draw_motion_info
    m_data = MotionData(
        object_id=1,
        dx=5.0,
        dy=5.0,
        displacement=7.07,
        direction="DOWN-RIGHT",
        angle=45.0,
        velocity=7.07,
        path_length=14.14,
        net_displacement=14.14,
    )
    out5 = vis.draw_motion_info(canvas, [track], {1: m_data})
    assert out5.shape == canvas.shape

    # 6. draw_statistics
    stats = {
        "total_active_objects": 1,
        "moving_objects": 1,
        "stationary_objects": 0,
        "average_velocity_px_s": 212.1,
        "average_displacement_px": 7.07,
    }
    out6 = vis.draw_statistics(canvas, stats, fps=30.0)
    assert out6.shape == canvas.shape
