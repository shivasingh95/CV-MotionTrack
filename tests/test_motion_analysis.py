"""
Unit tests for the motion analysis module.

Verifies:
- MotionAnalyzer initialization.
- Euclidean displacement calculation.
- Heading direction estimation.
- Velocity and instantaneous speed calculations.
- Historical tracking and summary statistics reporting.
"""

import pytest

from src.config import MotionAnalysisConfig
from src.data_models import MotionData, TrackedObject
from src.motion_analysis import MotionAnalyzer


def test_motion_analyzer_initialization():
    """Verify MotionAnalyzer initializes with default configuration and empty state."""
    analyzer = MotionAnalyzer()
    assert analyzer.config is not None
    assert len(analyzer.motion_history) == 0


def test_displacement_calculation():
    """Verify Euclidean spatial displacement calculation."""
    analyzer = MotionAnalyzer()

    # 3-4-5 right triangle
    disp = analyzer.calculate_displacement((0, 0), (3, 4))
    assert pytest.approx(disp, rel=1e-3) == 5.0

    # Stationary points
    zero_disp = analyzer.calculate_displacement((10, 20), (10, 20))
    assert zero_disp == 0.0


def test_direction_estimation():
    """Verify heading direction in degrees [0, 360)."""
    analyzer = MotionAnalyzer()

    # Moving right along horizontal axis (+X): 0 deg
    assert pytest.approx(analyzer.estimate_direction((0, 0), (10, 0)), abs=1e-2) == 0.0

    # Moving downward in image space (+Y): 90 deg
    assert pytest.approx(analyzer.estimate_direction((0, 0), (0, 10)), abs=1e-2) == 90.0

    # Moving left (-X): 180 deg
    assert pytest.approx(analyzer.estimate_direction((10, 0), (0, 0)), abs=1e-2) == 180.0

    # Stationary points default to 0 deg
    assert analyzer.estimate_direction((5, 5), (5, 5)) == 0.0


def test_velocity_estimation():
    """Verify velocity computation in pixels per second."""
    cfg = MotionAnalysisConfig(fps=30.0)
    analyzer = MotionAnalyzer(config=cfg)

    # 15 pixels displaced in 1 frame (1/30 s) -> 450 px/s
    dt = 1.0 / 30.0
    velocity = analyzer.estimate_velocity(15.0, dt)
    assert pytest.approx(velocity, rel=1e-3) == 450.0


def test_motion_update_pipeline():
    """Verify motion data generation from tracked object states."""
    cfg = MotionAnalysisConfig(fps=30.0, min_displacement_threshold=1.0)
    analyzer = MotionAnalyzer(config=cfg)

    # First observation (no previous position)
    track = TrackedObject(
        object_id=1,
        current_position=(100, 100),
        previous_position=None,
        trajectory=[(100, 100)],
    )
    motion = analyzer.update([track])
    assert 1 in motion
    assert motion[1].displacement == 0.0
    assert motion[1].approximate_velocity == 0.0

    # Second observation (moved to (103, 104) -> displacement = 5 px)
    track.previous_position = (100, 100)
    track.current_position = (103, 104)
    track.trajectory.append((103, 104))

    motion_f2 = analyzer.update([track])
    assert 1 in motion_f2
    m_data = motion_f2[1]
    assert isinstance(m_data, MotionData)
    assert pytest.approx(m_data.displacement, rel=1e-2) == 5.0
    assert pytest.approx(m_data.approximate_velocity, rel=1e-2) == 150.0

    # Verify summary statistics
    stats = analyzer.get_summary_statistics()
    assert stats["total_objects_tracked"] == 1
    assert stats["average_velocity_px_s"] > 0.0
