"""
Unit tests for the Evaluation module (src/evaluation.py).

Academic Coursework: CSE3010 Computer Vision - Step 9
Verifies:
1. Evaluator initializes correctly.
2. High-resolution frame timing and FPS calculations.
3. Detection metric aggregation (counts, min, max, zero-detection frames).
4. Tracking metric aggregation (lifetime, unique IDs, active counts).
5. Optical flow metric aggregation (valid points, reinitialization events).
6. Motion analysis metric aggregation (displacement, velocity, moving/stationary).
7. Empty-result handling (zero frames evaluated without ZeroDivisionError).
8. CSV export format and schema validation.
9. JSON export structured schema and frame history serialization.
10. Reset functionality restores clean initial state.
11. Terminal summary reporting execution.
"""

import json
import os
import tempfile
import time
import pytest

from src.data_models import BoundingBox, Detection, MotionData, OpticalFlowPoint, TrackedObject
from src.evaluation import EvaluationManager, EvaluationSummary, Evaluator, FrameMetrics


# -------------------------------------------------------------------------- #
# 1. Initialization Test                                                     #
# -------------------------------------------------------------------------- #

def test_evaluator_initialization():
    """Verify Evaluator initializes with clean initial state and default/custom tags."""
    evaluator = Evaluator()
    assert evaluator.experiment_name == "default_experiment"
    assert len(evaluator.frame_history) == 0
    assert evaluator._current_frame_idx == 0

    custom_eval = Evaluator(experiment_name="custom_trial_1")
    assert custom_eval.experiment_name == "custom_trial_1"

    # Verify alias EvaluationManager
    alias_eval = EvaluationManager(experiment_name="alias_trial")
    assert isinstance(alias_eval, Evaluator)


# -------------------------------------------------------------------------- #
# 2. Timing and FPS Calculation Test                                         #
# -------------------------------------------------------------------------- #

def test_frame_timing_and_fps_calculation():
    """Verify time.perf_counter integration and positive FPS calculation."""
    evaluator = Evaluator()

    evaluator.start_frame()
    time.sleep(0.005)  # simulate brief frame computation
    frame_m = evaluator.record_frame()

    assert frame_m.frame_index == 1
    assert frame_m.processing_time_sec > 0.001
    assert frame_m.fps > 0.0
    assert len(evaluator.frame_history) == 1

    summary = evaluator.get_summary()
    assert summary.total_frames == 1
    assert summary.total_processing_time_sec > 0.001
    assert summary.average_processing_time_ms > 1.0
    assert summary.average_fps > 0.0


# -------------------------------------------------------------------------- #
# 3. Detection Metric Aggregation                                            #
# -------------------------------------------------------------------------- #

def test_detection_metrics_aggregation():
    """Verify detection counts, max, min, and zero-detection frames tracking."""
    evaluator = Evaluator()

    # Frame 1: 2 detections
    det1 = Detection(bbox=BoundingBox(10, 10, 20, 20), centroid=(20, 20), confidence=1.0, area=400.0)
    det2 = Detection(bbox=BoundingBox(50, 50, 20, 20), centroid=(60, 60), confidence=1.0, area=400.0)
    evaluator.start_frame()
    evaluator.record_frame(detections=[det1, det2])

    # Frame 2: 0 detections
    evaluator.start_frame()
    evaluator.record_frame(detections=[])

    # Frame 3: 1 detection
    evaluator.start_frame()
    evaluator.record_frame(detections=[det1])

    summary = evaluator.get_summary()
    assert summary.total_frames == 3
    assert summary.total_detections == 3
    assert pytest.approx(summary.average_detections_per_frame) == 1.0
    assert summary.max_detections_per_frame == 2
    assert summary.min_detections_per_frame == 0
    assert summary.frames_with_zero_detections == 1


# -------------------------------------------------------------------------- #
# 4. Tracking Metrics and Lifetime Aggregation                              #
# -------------------------------------------------------------------------- #

def test_tracking_metrics_and_lifetime():
    """Verify unique track registration and lifetime metrics."""
    evaluator = Evaluator()

    track_1 = TrackedObject(
        object_id=1,
        current_position=(20, 20),
        trajectory=[(10, 10), (15, 15), (20, 20)],
    )
    track_2 = TrackedObject(
        object_id=2,
        current_position=(50, 50),
        trajectory=[(50, 50)],
    )

    # Frame 1: track 1 and track 2 active
    evaluator.start_frame()
    evaluator.record_frame(tracked_objects=[track_1, track_2])

    # Frame 2: track 1 only (track 2 deregistered)
    track_1.trajectory.append((25, 25))
    evaluator.start_frame()
    evaluator.record_frame(tracked_objects=[track_1])

    summary = evaluator.get_summary()
    assert summary.total_registered_tracks == 2
    assert summary.deregistered_tracks_count == 1
    assert pytest.approx(summary.average_active_tracks_per_frame) == 1.5
    assert summary.max_active_tracks_per_frame == 2
    assert summary.max_track_lifetime_frames == 4  # track 1 reached 4 points
    assert summary.average_track_lifetime_frames == 2.5  # (4 + 1) / 2


# -------------------------------------------------------------------------- #
# 5. Optical Flow Metrics Aggregation                                       #
# -------------------------------------------------------------------------- #

def test_optical_flow_metrics():
    """Verify valid flow points tracking and reinitialization event recording."""
    evaluator = Evaluator()

    pt1 = OpticalFlowPoint(10.0, 10.0, 12.0, 12.0, 2.0, 2.0)
    pt2 = OpticalFlowPoint(30.0, 30.0, 31.0, 31.0, 1.0, 1.0)

    # Frame 1: 2 flow points, no reinit
    evaluator.start_frame()
    evaluator.record_frame(optical_flow_points=[pt1, pt2], reinitialized_flow=False)

    # Frame 2: 0 flow points, reinit triggered
    evaluator.start_frame()
    evaluator.record_frame(optical_flow_points=[], reinitialized_flow=True)

    summary = evaluator.get_summary()
    assert pytest.approx(summary.average_valid_flow_points) == 1.0
    assert summary.min_valid_flow_points == 0
    assert summary.max_valid_flow_points == 2
    assert summary.flow_reinitialization_events == 1


# -------------------------------------------------------------------------- #
# 6. Motion Metrics Aggregation                                              #
# -------------------------------------------------------------------------- #

def test_motion_analysis_metrics():
    """Verify displacement, velocity, and moving/stationary observation counts."""
    evaluator = Evaluator()

    m_moving = MotionData(
        object_id=1,
        dx=3.0,
        dy=4.0,
        displacement=5.0,
        direction="DOWN-RIGHT",
        angle=53.13,
        velocity=5.0,
        path_length=5.0,
        net_displacement=5.0,
        approximate_velocity=150.0,
    )
    m_stationary = MotionData(
        object_id=2,
        dx=0.0,
        dy=0.0,
        displacement=0.0,
        direction="STATIONARY",
        angle=0.0,
        velocity=0.0,
        path_length=0.0,
        net_displacement=0.0,
        approximate_velocity=0.0,
    )

    evaluator.start_frame()
    evaluator.record_frame(motion_data={1: m_moving, 2: m_stationary})

    summary = evaluator.get_summary()
    assert summary.total_moving_observations == 1
    assert summary.total_stationary_observations == 1
    assert pytest.approx(summary.average_displacement_px_frame) == 2.5
    assert pytest.approx(summary.average_velocity_px_s) == 75.0


# -------------------------------------------------------------------------- #
# 7. Empty Evaluator Summary Handling                                       #
# -------------------------------------------------------------------------- #

def test_empty_evaluator_summary():
    """Verify get_summary returns clean zeroed metrics when no frames were recorded."""
    evaluator = Evaluator()
    summary = evaluator.get_summary()

    assert summary.total_frames == 0
    assert summary.total_processing_time_sec == 0.0
    assert summary.average_fps == 0.0
    assert summary.total_detections == 0
    assert summary.total_registered_tracks == 0


# -------------------------------------------------------------------------- #
# 8. CSV Export Test                                                         #
# -------------------------------------------------------------------------- #

def test_csv_export():
    """Verify save_csv produces valid CSV headers and rows."""
    evaluator = Evaluator(experiment_name="csv_unit_test")
    evaluator.start_frame()
    evaluator.record_frame()

    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = os.path.join(tmp_dir, "results.csv")
        evaluator.save_csv(csv_path, video_name="synthetic.avi")

        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        assert len(lines) == 2  # header + 1 row
        assert "experiment,video,frames,total_time_s,avg_fps" in lines[0]
        assert "csv_unit_test,synthetic.avi,1" in lines[1]


# -------------------------------------------------------------------------- #
# 9. JSON Export Test                                                        #
# -------------------------------------------------------------------------- #

def test_json_export():
    """Verify save_json exports valid JSON with structured summary and history."""
    evaluator = Evaluator(experiment_name="json_unit_test")
    evaluator.start_frame()
    evaluator.record_frame()

    with tempfile.TemporaryDirectory() as tmp_dir:
        json_path = os.path.join(tmp_dir, "results.json")
        evaluator.save_json(json_path, video_name="demo.avi", include_frame_history=True)

        assert os.path.exists(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["experiment_name"] == "json_unit_test"
        assert data["video_name"] == "demo.avi"
        assert "summary" in data
        assert data["summary"]["total_frames"] == 1
        assert len(data["frame_history"]) == 1


# -------------------------------------------------------------------------- #
# 10. Evaluator Reset Test                                                   #
# -------------------------------------------------------------------------- #

def test_evaluator_reset():
    """Verify reset clears all recorded history, lifetimes, and counters."""
    evaluator = Evaluator()
    evaluator.start_frame()
    evaluator.record_frame(
        detections=[Detection(BoundingBox(0, 0, 10, 10), (5, 5), 1.0, 100.0)],
        tracked_objects=[TrackedObject(1, (5, 5))],
        optical_flow_points=[OpticalFlowPoint(0.0, 0.0, 1.0, 1.0, 1.0, 1.0)],
        reinitialized_flow=True,
    )

    assert len(evaluator.frame_history) == 1
    assert evaluator._flow_reinit_count == 1

    evaluator.reset()
    assert len(evaluator.frame_history) == 0
    assert evaluator._current_frame_idx == 0
    assert len(evaluator._track_lifetimes) == 0
    assert len(evaluator._all_registered_ids) == 0
    assert evaluator._flow_reinit_count == 0


# -------------------------------------------------------------------------- #
# 11. Terminal Summary Output Test                                           #
# -------------------------------------------------------------------------- #

def test_terminal_summary_output(capsys):
    """Verify print_terminal_summary executes cleanly and outputs formatted text."""
    evaluator = Evaluator(experiment_name="cli_summary_test")
    evaluator.start_frame()
    evaluator.record_frame()
    evaluator.print_terminal_summary()

    captured = capsys.readouterr()
    assert "CV-MotionTrack Evaluation: cli_summary_test" in captured.out
    assert "Frames processed" in captured.out
    assert "Average pipeline FPS" in captured.out
