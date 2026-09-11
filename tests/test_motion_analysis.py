"""
Unit tests for the Motion Analysis module (src/motion_analysis.py).

Academic Coursework: CSE3010 Computer Vision - Step 7

Covers all 16 required evaluation criteria and edge cases:
1. MotionAnalyzer initializes correctly.
2. Zero movement returns zero displacement.
3. Correct dx and dy are calculated.
4. Euclidean displacement is correct.
5. RIGHT direction is detected.
6. LEFT direction is detected.
7. UP direction is detected.
8. DOWN direction is detected.
9. Diagonal directions are detected (UP-RIGHT, UP-LEFT, DOWN-RIGHT, DOWN-LEFT).
10. Very small movement is classified as STATIONARY.
11. Angle calculation is correct.
12. Velocity calculation is correct.
13. Path length is different from net displacement when the path bends.
14. Multiple observations are accumulated correctly.
15. Motion history is bounded.
16. Invalid input is handled clearly.

Edge cases verified:
- None positions
- Malformed positions
- Zero displacement
- Diagonal motion
- Very small movement
- Empty trajectory
- Single-point trajectory
- Multiple-point trajectory
"""

import math
import pytest

from src.config import MotionAnalysisConfig
from src.data_models import BoundingBox, MotionData, TrackedObject
from src.motion_analysis import Displacement, MotionAnalyzer


def _create_track(
    object_id: int,
    current_pos: tuple,
    previous_pos: tuple = None,
    trajectory: list = None,
) -> TrackedObject:
    """Helper to construct synthetic TrackedObject instances."""
    if trajectory is None:
        trajectory = [previous_pos, current_pos] if previous_pos else [current_pos]
    bbox = BoundingBox(x=current_pos[0] - 10, y=current_pos[1] - 10, width=20, height=20)
    return TrackedObject(
        object_id=object_id,
        current_position=current_pos,
        previous_position=previous_pos,
        trajectory=trajectory,
        bbox=bbox,
    )


# ---------------------------------------------------------------------- #
# 1. Initialization Test                                                 #
# ---------------------------------------------------------------------- #

def test_motion_analyzer_initialization():
    """Verify MotionAnalyzer initializes with default and custom settings."""
    analyzer = MotionAnalyzer()
    assert analyzer.smoothing_window == 5
    assert analyzer.stationary_threshold == 1.0
    assert analyzer.history_length == 30
    assert len(analyzer.motion_history) == 0

    # Custom parameter initialization
    custom_analyzer = MotionAnalyzer(
        smoothing_window=10,
        stationary_threshold=2.5,
        history_length=50,
    )
    assert custom_analyzer.smoothing_window == 10
    assert custom_analyzer.stationary_threshold == 2.5
    assert custom_analyzer.history_length == 50

    # Config object initialization
    cfg = MotionAnalysisConfig(
        smoothing_window=7,
        stationary_threshold=1.5,
        history_length=40,
        fps=25.0,
    )
    cfg_analyzer = MotionAnalyzer(config=cfg)
    assert cfg_analyzer.smoothing_window == 7
    assert cfg_analyzer.stationary_threshold == 1.5
    assert cfg_analyzer.history_length == 40


# ---------------------------------------------------------------------- #
# 2. Zero Movement Returns Zero Displacement                             #
# ---------------------------------------------------------------------- #

def test_zero_movement_returns_zero_displacement():
    """Verify identical positions yield exactly zero dx, dy, and displacement."""
    analyzer = MotionAnalyzer()
    p1 = (25, 40)
    p2 = (25, 40)

    dx, dy, disp = analyzer.calculate_displacement(p1, p2)
    assert dx == 0.0
    assert dy == 0.0
    assert disp == 0.0

    # Float compute_displacement compatibility
    disp_float = analyzer.compute_displacement(p1, p2)
    assert disp_float == 0.0
    assert isinstance(disp_float, float)


# ---------------------------------------------------------------------- #
# 3. Correct dx and dy Calculated                                        #
# ---------------------------------------------------------------------- #

def test_correct_dx_and_dy_calculated():
    """Verify coordinate shifts dx = x2 - x1, dy = y2 - y1."""
    analyzer = MotionAnalyzer()

    # Positive shifts: (10, 10) -> (13, 14)
    dx, dy, disp = analyzer.calculate_displacement((10, 10), (13, 14))
    assert dx == 3.0
    assert dy == 4.0
    assert disp == 5.0

    # Negative shifts: (50, 60) -> (42, 53)
    dx_neg, dy_neg, disp_neg = analyzer.calculate_displacement((50, 60), (42, 53))
    assert dx_neg == -8.0
    assert dy_neg == -7.0
    assert pytest.approx(disp_neg, rel=1e-3) == math.hypot(-8.0, -7.0)


# ---------------------------------------------------------------------- #
# 4. Euclidean Displacement Is Correct                                   #
# ---------------------------------------------------------------------- #

def test_euclidean_displacement_correct():
    """Verify Euclidean distance d = sqrt(dx^2 + dy^2)."""
    analyzer = MotionAnalyzer()

    # 3-4-5 right triangle
    disp_5 = analyzer.calculate_displacement((0, 0), (3, 4))
    assert pytest.approx(disp_5.displacement) == 5.0
    assert disp_5.displacement == 5.0
    assert disp_5.dx == 3.0
    assert disp_5.dy == 4.0

    # 8-15-17 right triangle
    disp_17 = analyzer.calculate_displacement((10, 20), (18, 35))
    assert pytest.approx(disp_17.displacement) == 17.0
    assert disp_17.dx == 8.0
    assert disp_17.dy == 15.0


# ---------------------------------------------------------------------- #
# 5. RIGHT Direction Detected                                            #
# ---------------------------------------------------------------------- #

def test_right_direction_detected():
    """Verify horizontal motion toward increasing x is classified as RIGHT."""
    analyzer = MotionAnalyzer()
    direction = analyzer.calculate_direction((10, 10), (25, 10))
    assert direction == "RIGHT"

    # Also test via dx, dy directly
    assert analyzer.calculate_direction(15.0, 0.0) == "RIGHT"


# ---------------------------------------------------------------------- #
# 6. LEFT Direction Detected                                             #
# ---------------------------------------------------------------------- #

def test_left_direction_detected():
    """Verify horizontal motion toward decreasing x is classified as LEFT."""
    analyzer = MotionAnalyzer()
    direction = analyzer.calculate_direction((25, 10), (10, 10))
    assert direction == "LEFT"

    assert analyzer.calculate_direction(-15.0, 0.0) == "LEFT"


# ---------------------------------------------------------------------- #
# 7. UP Direction Detected                                               #
# ---------------------------------------------------------------------- #

def test_up_direction_detected():
    """Verify vertical motion toward decreasing y is classified as UP."""
    analyzer = MotionAnalyzer()
    # In image coordinates, y increases downward, so decreasing y is UP
    direction = analyzer.calculate_direction((10, 30), (10, 15))
    assert direction == "UP"

    assert analyzer.calculate_direction(0.0, -15.0) == "UP"


# ---------------------------------------------------------------------- #
# 8. DOWN Direction Detected                                             #
# ---------------------------------------------------------------------- #

def test_down_direction_detected():
    """Verify vertical motion toward increasing y is classified as DOWN."""
    analyzer = MotionAnalyzer()
    # In image coordinates, increasing y is DOWN
    direction = analyzer.calculate_direction((10, 15), (10, 30))
    assert direction == "DOWN"

    assert analyzer.calculate_direction(0.0, 15.0) == "DOWN"


# ---------------------------------------------------------------------- #
# 9. Diagonal Directions Detected                                        #
# ---------------------------------------------------------------------- #

def test_diagonal_directions_detected():
    """Verify all four diagonal quadrant directions in image space."""
    analyzer = MotionAnalyzer()

    # UP-RIGHT: +x, -y
    assert analyzer.calculate_direction((10, 30), (20, 20)) == "UP-RIGHT"
    assert analyzer.calculate_direction(10.0, -10.0) == "UP-RIGHT"

    # UP-LEFT: -x, -y
    assert analyzer.calculate_direction((30, 30), (20, 20)) == "UP-LEFT"
    assert analyzer.calculate_direction(-10.0, -10.0) == "UP-LEFT"

    # DOWN-RIGHT: +x, +y
    assert analyzer.calculate_direction((10, 10), (20, 20)) == "DOWN-RIGHT"
    assert analyzer.calculate_direction(10.0, 10.0) == "DOWN-RIGHT"

    # DOWN-LEFT: -x, +y
    assert analyzer.calculate_direction((30, 10), (20, 20)) == "DOWN-LEFT"
    assert analyzer.calculate_direction(-10.0, 10.0) == "DOWN-LEFT"


# ---------------------------------------------------------------------- #
# 10. Small Movement Classified as STATIONARY                            #
# ---------------------------------------------------------------------- #

def test_small_movement_classified_as_stationary():
    """Verify movements below stationary_threshold are marked as STATIONARY."""
    analyzer = MotionAnalyzer(stationary_threshold=1.5)

    # Sub-threshold shift: dx = 0.6, dy = 0.8 -> disp = 1.0 < 1.5
    direction = analyzer.calculate_direction((10, 10), (10.6, 10.8))
    assert direction == "STATIONARY"

    # Barely above threshold: dx = 1.2, dy = 1.6 -> disp = 2.0 >= 1.5
    moving_dir = analyzer.calculate_direction((10, 10), (11.2, 11.6))
    assert moving_dir == "DOWN-RIGHT"


# ---------------------------------------------------------------------- #
# 11. Angle Calculation Correct                                          #
# ---------------------------------------------------------------------- #

def test_angle_calculation_correct():
    """Verify heading angle in degrees [0, 360) normalized relative to +X axis."""
    analyzer = MotionAnalyzer()

    # RIGHT: 0 deg
    assert pytest.approx(analyzer.calculate_angle((0, 0), (10, 0))) == 0.0
    # DOWN (+Y in image coordinates): 90 deg
    assert pytest.approx(analyzer.calculate_angle((0, 0), (0, 10))) == 90.0
    # LEFT (-X): 180 deg
    assert pytest.approx(analyzer.calculate_angle((10, 0), (0, 0))) == 180.0
    # UP (-Y in image coordinates): 270 deg
    assert pytest.approx(analyzer.calculate_angle((0, 10), (0, 0))) == 270.0
    # DOWN-RIGHT: 45 deg
    assert pytest.approx(analyzer.calculate_angle((0, 0), (10, 10))) == 45.0
    # UP-RIGHT: 315 deg
    assert pytest.approx(analyzer.calculate_angle((0, 10), (10, 0))) == 315.0

    # Stationary points default to 0.0
    assert analyzer.calculate_angle((5, 5), (5, 5)) == 0.0


# ---------------------------------------------------------------------- #
# 12. Velocity Calculation Correct                                       #
# ---------------------------------------------------------------------- #

def test_velocity_calculation_correct():
    """Verify image-space velocity v = d / dt in pixels/frame and pixels/second."""
    analyzer = MotionAnalyzer()

    # Default frame interval: 1 frame -> 15.0 pixels/frame
    v_frame = analyzer.calculate_velocity(15.0, frame_interval=1.0)
    assert pytest.approx(v_frame) == 15.0

    # Scaled with FPS = 30 (dt = 1/30 s) -> 450.0 pixels/second
    dt = 1.0 / 30.0
    v_sec = analyzer.calculate_velocity(15.0, frame_interval=dt)
    assert pytest.approx(v_sec) == 450.0

    # Negative or zero frame interval raises ValueError
    with pytest.raises(ValueError):
        analyzer.calculate_velocity(10.0, frame_interval=0.0)

    with pytest.raises(ValueError):
        analyzer.calculate_velocity(10.0, frame_interval=-1.0)


# ---------------------------------------------------------------------- #
# 13. Path Length Differs from Net Displacement on Bending Trajectories  #
# ---------------------------------------------------------------------- #

def test_path_length_differs_from_net_displacement():
    """Verify total path length exceeds net displacement on bent trajectories."""
    analyzer = MotionAnalyzer()

    # Bending path: (0, 0) -> (0, 10) -> (10, 10)
    # Segment 1: length 10.0
    # Segment 2: length 10.0
    # Total path length = 20.0
    # Net displacement = distance((0,0), (10,10)) = sqrt(200) ≈ 14.142
    trajectory = [(0, 0), (0, 10), (10, 10)]

    path_length = analyzer.calculate_path_length(trajectory)
    net_disp = analyzer.calculate_net_displacement(trajectory)

    assert pytest.approx(path_length) == 20.0
    assert pytest.approx(net_disp) == math.hypot(10, 10)
    assert path_length > net_disp

    # Straight path: Path length equals net displacement
    straight_traj = [(0, 0), (5, 0), (10, 0)]
    assert pytest.approx(analyzer.calculate_path_length(straight_traj)) == 10.0
    assert pytest.approx(analyzer.calculate_net_displacement(straight_traj)) == 10.0


# ---------------------------------------------------------------------- #
# 14. Multiple Observations Accumulated Correctly                        #
# ---------------------------------------------------------------------- #

def test_multiple_observations_accumulated():
    """Verify sequential updates accumulate history and update active statistics."""
    analyzer = MotionAnalyzer(smoothing_window=5)

    # Frame 1: Birth of object 1 at (100, 100)
    track1 = _create_track(1, (100, 100), None, [(100, 100)])
    m1 = analyzer.update([track1])[1]
    assert m1.displacement == 0.0
    assert m1.direction == "STATIONARY"
    assert len(analyzer.motion_history[1]) == 1

    # Frame 2: Object 1 moves to (103, 104) -> displacement = 5.0
    track2 = _create_track(1, (103, 104), (100, 100), [(100, 100), (103, 104)])
    m2 = analyzer.update([track2])[1]
    assert pytest.approx(m2.displacement) == 5.0
    assert m2.direction == "DOWN-RIGHT"
    assert len(analyzer.motion_history[1]) == 2

    # Frame 3: Object 1 moves to (106, 108) -> displacement = 5.0
    track3 = _create_track(
        1, (106, 108), (103, 104), [(100, 100), (103, 104), (106, 108)]
    )
    m3 = analyzer.update([track3])[1]
    assert pytest.approx(m3.displacement) == 5.0
    assert len(analyzer.motion_history[1]) == 3

    # Statistics reporting
    stats = analyzer.get_statistics()
    assert stats["total_active_objects"] == 1
    assert stats["moving_objects"] == 1
    assert stats["stationary_objects"] == 0
    assert stats["objects"][1]["number_of_observations"] == 3


# ---------------------------------------------------------------------- #
# 15. Motion History Is Bounded                                          #
# ---------------------------------------------------------------------- #

def test_motion_history_is_bounded():
    """Verify history length is capped at history_length without unbounded growth."""
    max_hist = 5
    analyzer = MotionAnalyzer(history_length=max_hist)

    # Update 12 times
    curr = (10, 10)
    for i in range(12):
        nxt = (curr[0] + 5, curr[1] + 5)
        track = _create_track(1, nxt, curr)
        analyzer.update([track])
        curr = nxt

    # Registry must be capped at max_hist = 5
    assert len(analyzer.motion_history[1]) == max_hist


# ---------------------------------------------------------------------- #
# 16. Invalid Input Handled Clearly                                      #
# ---------------------------------------------------------------------- #

def test_invalid_input_handled_clearly():
    """Verify clear ValueError exceptions on invalid or malformed arguments."""
    analyzer = MotionAnalyzer()

    # None coordinates
    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.calculate_displacement(None, (10, 10))

    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.calculate_displacement((10, 10), None)

    # Malformed length coordinates
    with pytest.raises(ValueError, match="must be a 2D coordinate"):
        analyzer.calculate_displacement((10,), (10, 20))

    with pytest.raises(ValueError, match="must be a 2D coordinate"):
        analyzer.calculate_displacement((10, 20, 30), (10, 20))

    # Non-numeric coordinate values
    with pytest.raises(ValueError, match="must be numeric"):
        analyzer.calculate_displacement(("a", 10), (10, 20))

    # NaN coordinates
    with pytest.raises(ValueError, match="cannot be NaN"):
        analyzer.calculate_displacement((float("nan"), 10), (10, 20))

    # None track
    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.analyze_track(None)

    # None update list
    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.update(None)

    # Trajectory edge cases
    with pytest.raises(ValueError, match="cannot be None"):
        analyzer.calculate_path_length(None)

    # Empty trajectory safely returns 0.0
    assert analyzer.calculate_path_length([]) == 0.0
    assert analyzer.calculate_net_displacement([]) == 0.0

    # Single-point trajectory safely returns 0.0
    assert analyzer.calculate_path_length([(50, 50)]) == 0.0
    assert analyzer.calculate_net_displacement([(50, 50)]) == 0.0
