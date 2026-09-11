"""
Motion Analysis Module for CV-MotionTrack.

Academic Coursework: CSE3010 Computer Vision - Step 7

Academic Purpose & Theoretical Foundations:
============================================
1. Spatio-Temporal Motion Analysis:
   Motion cannot be observed from a static spatial image; it inherently requires observing
   changes across both SPACE (2D pixel coordinates (x, y)) and TIME (discrete frame sequences).
   An object trajectory is a discrete spatio-temporal path:
       T = [(x1, y1, t1), (x2, y2, t2), ..., (xn, yn, tn)]
   where each coordinate represents the object's spatial centroid at temporal epoch ti.

2. Frame-to-Frame Displacement:
   For consecutive observation points (x1, y1) and (x2, y2):
       dx = x2 - x1
       dy = y2 - y1
   The Euclidean displacement magnitude (in pixels) is:
       d = sqrt(dx^2 + dy^2)

3. Image-Space Velocity vs. Physical-World Velocity:
   In the absence of calibrated extrinsic/intrinsic camera parameters and a physical scene scale
   (e.g., mm/pixel or world homography), physical velocity (m/s, km/h) CANNOT be determined.
   Therefore, motion is rigorously evaluated in IMAGE SPACE:
       v = d / Δt
   By default, with frame interval Δt = 1 frame:
       v is measured in pixels/frame.
   When capture frame rate (FPS) is available (Δt = 1 / FPS):
       v can be expressed in pixels/second.

4. Image-Space Direction Classification:
   In digital image coordinates:
       - X increases towards the RIGHT (+x).
       - Y increases towards the BOTTOM (+y).
   Consequently:
       - UP corresponds to negative dy (-y).
       - DOWN corresponds to positive dy (+y).
       - RIGHT corresponds to positive dx (+x).
       - LEFT corresponds to negative dx (-x).
   To prevent false directional classifications due to sensor noise or contour centroid jitter,
   displacements below `stationary_threshold` are classified as STATIONARY.
   For moving objects, 8 continuous 45-degree sectors define the heading:
       RIGHT, DOWN-RIGHT, DOWN, DOWN-LEFT, LEFT, UP-LEFT, UP, UP-RIGHT.

5. Trajectory Metrics: Path Length vs. Net Displacement:
   - Total Path Length (L): Cumulative Euclidean distance traversed along the trajectory:
       L = ∑_{i=1}^{n-1} sqrt((x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2)
   - Net Displacement (D): Direct straight-line Euclidean distance from start to finish:
       D = sqrt((xn - x1)^2 + (yn - y1)^2)
   For curved or oscillatory paths, L > D. Only for strictly rectilinear motion does L = D.

6. Rolling Moving-Average Smoothing:
   Raw object centroid detections may exhibit minor sub-pixel contour vibrations. A sliding
   moving-average window across the last K observations dampens high-frequency jitter without
   attenuating authentic macro-scale motion trajectories.
"""

import math
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np

from src.config import (
    MOTION_HISTORY_LENGTH,
    MOTION_SMOOTHING_WINDOW,
    STATIONARY_THRESHOLD,
    MotionAnalysisConfig,
)
from src.data_models import MotionData, OpticalFlowPoint, TrackedObject


class Displacement(NamedTuple):
    """
    Represents Euclidean spatial displacement and coordinate shifts between two points.

    Fields:
        dx: Horizontal delta in pixels (x2 - x1).
        dy: Vertical delta in pixels (y2 - y1).
        displacement: Euclidean distance magnitude in pixels sqrt(dx^2 + dy^2).
    """
    dx: float
    dy: float
    displacement: float


class MotionAnalyzer:
    """
    Estimates image-space motion kinematics (displacement, direction, velocity, trajectory)
    for tracked objects across discrete video frames.

    Attributes:
        smoothing_window: Number of historical frames for rolling velocity smoothing.
        stationary_threshold: Minimum pixel displacement required to classify motion as non-stationary.
        history_length: Maximum motion history states retained per object ID.
        config: MotionAnalysisConfig holding hyperparameters.
        motion_history: Historical record of MotionData per object ID.
    """

    def __init__(
        self,
        smoothing_window: Union[int, MotionAnalysisConfig] = MOTION_SMOOTHING_WINDOW,
        stationary_threshold: float = STATIONARY_THRESHOLD,
        history_length: int = MOTION_HISTORY_LENGTH,
        config: Optional[MotionAnalysisConfig] = None,
    ) -> None:
        """
        Initialize the MotionAnalyzer.

        Supports configuration via individual keyword arguments or by passing
        a MotionAnalysisConfig dataclass instance.
        """
        if isinstance(smoothing_window, MotionAnalysisConfig):
            config = smoothing_window
            smoothing_window = MOTION_SMOOTHING_WINDOW

        if config is not None:
            smoothing_window = getattr(config, "smoothing_window", config.speed_smoothing_window)
            stationary_threshold = getattr(config, "stationary_threshold", config.min_displacement_threshold)
            history_length = getattr(config, "history_length", MOTION_HISTORY_LENGTH)

        self.smoothing_window: int = max(1, int(smoothing_window))
        self.stationary_threshold: float = float(stationary_threshold)
        self.history_length: int = max(1, int(history_length))
        self.config: MotionAnalysisConfig = config or MotionAnalysisConfig(
            smoothing_window=self.smoothing_window,
            stationary_threshold=self.stationary_threshold,
            history_length=self.history_length,
        )

        # Object tracking history caches
        self.motion_history: Dict[int, List[MotionData]] = {}
        self.recent_velocities: Dict[int, List[float]] = {}
        self.recent_displacements: Dict[int, List[float]] = {}
        self.last_active_objects: Dict[int, MotionData] = {}

    # ------------------------------------------------------------------ #
    # Input Validation Helpers                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_position(pos: Any, name: str = "Position") -> Tuple[float, float]:
        """
        Validate that an input coordinate is a valid non-null 2D numeric coordinate.

        Raises:
            ValueError: If position is None, not a 2D sequence, or contains non-numeric values.
        """
        if pos is None:
            raise ValueError(f"{name} cannot be None.")
        if not isinstance(pos, (tuple, list, np.ndarray)) or len(pos) != 2:
            raise ValueError(f"{name} must be a 2D coordinate sequence of length 2 (got {pos}).")
        try:
            x = float(pos[0])
            y = float(pos[1])
            if math.isnan(x) or math.isnan(y) or math.isinf(x) or math.isinf(y):
                raise ValueError(f"{name} coordinates cannot be NaN or Infinite.")
            return (x, y)
        except (TypeError, ValueError) as e:
            raise ValueError(f"{name} coordinates must be numeric: {e}")

    # ------------------------------------------------------------------ #
    # Core Kinematic Calculations                                        #
    # ------------------------------------------------------------------ #

    def calculate_displacement(
        self,
        previous_position: Any,
        current_position: Any,
    ) -> Displacement:
        """
        Calculate Euclidean displacement and coordinate shifts between two points:
            dx = x2 - x1
            dy = y2 - y1
            displacement = sqrt(dx^2 + dy^2)

        Returns a Displacement object that acts as a float (value = displacement)
        and unpacks as a 3-tuple (dx, dy, displacement).

        Args:
            previous_position: Starting coordinate (x1, y1).
            current_position: Ending coordinate (x2, y2).

        Returns:
            Displacement: Float-compatible object unpacking as (dx, dy, displacement).
        """
        p1 = self._validate_position(previous_position, "previous_position")
        p2 = self._validate_position(current_position, "current_position")

        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        disp = float(math.hypot(dx, dy))

        return Displacement(dx, dy, disp)

    def compute_displacement(self, p1: Any, p2: Any) -> float:
        """
        Calculate scalar Euclidean displacement magnitude between two points.

        Args:
            p1: Starting coordinate (x1, y1).
            p2: Ending coordinate (x2, y2).

        Returns:
            float: Euclidean distance in pixels.
        """
        return self.calculate_displacement(p1, p2).displacement

    def calculate_direction(
        self,
        previous_position_or_dx: Any,
        current_position_or_dy: Any = None,
    ) -> str:
        """
        Determine the qualitative image-space direction of motion.

        Possible directions:
            'RIGHT', 'LEFT', 'UP', 'DOWN',
            'UP-RIGHT', 'UP-LEFT', 'DOWN-RIGHT', 'DOWN-LEFT',
            'STATIONARY'

        Direction Sector Mapping (in image coordinates where +y is DOWN):
            - displacement < stationary_threshold: 'STATIONARY'
            - [337.5°, 360°) ∪ [0°, 22.5°): 'RIGHT'
            - [22.5°, 67.5°): 'DOWN-RIGHT'
            - [67.5°, 112.5°): 'DOWN'
            - [112.5°, 157.5°): 'DOWN-LEFT'
            - [157.5°, 202.5°): 'LEFT'
            - [202.5°, 247.5°): 'UP-LEFT'
            - [247.5°, 292.5°): 'UP'
            - [292.5°, 337.5°): 'UP-RIGHT'

        Args:
            previous_position_or_dx: (x1, y1) tuple or dx float.
            current_position_or_dy: (x2, y2) tuple or dy float.

        Returns:
            str: Direction name.
        """
        if isinstance(previous_position_or_dx, (int, float)) and isinstance(
            current_position_or_dy, (int, float)
        ):
            dx = float(previous_position_or_dx)
            dy = float(current_position_or_dy)
            disp = math.hypot(dx, dy)
        else:
            dx, dy, disp = self.calculate_displacement(
                previous_position_or_dx, current_position_or_dy
            )

        if disp < self.stationary_threshold:
            return "STATIONARY"

        # Calculate angle in degrees [0, 360)
        angle = (math.degrees(math.atan2(dy, dx))) % 360.0

        if angle >= 337.5 or angle < 22.5:
            return "RIGHT"
        elif 22.5 <= angle < 67.5:
            return "DOWN-RIGHT"
        elif 67.5 <= angle < 112.5:
            return "DOWN"
        elif 112.5 <= angle < 157.5:
            return "DOWN-LEFT"
        elif 157.5 <= angle < 202.5:
            return "LEFT"
        elif 202.5 <= angle < 247.5:
            return "UP-LEFT"
        elif 247.5 <= angle < 292.5:
            return "UP"
        else:  # 292.5 <= angle < 337.5
            return "UP-RIGHT"

    def calculate_angle(
        self,
        previous_position_or_dx: Any,
        current_position_or_dy: Any = None,
    ) -> float:
        """
        Calculate motion heading angle in degrees [0.0, 360.0) relative to positive horizontal axis.

        In image coordinates:
            - 0° points along +X (RIGHT).
            - 90° points along +Y (DOWN).
            - 180° points along -X (LEFT).
            - 270° points along -Y (UP).

        Args:
            previous_position_or_dx: (x1, y1) tuple or dx float.
            current_position_or_dy: (x2, y2) tuple or dy float.

        Returns:
            float: Angle in degrees in range [0.0, 360.0).
        """
        if isinstance(previous_position_or_dx, (int, float)) and isinstance(
            current_position_or_dy, (int, float)
        ):
            dx = float(previous_position_or_dx)
            dy = float(current_position_or_dy)
        else:
            p1 = self._validate_position(previous_position_or_dx, "previous_position")
            p2 = self._validate_position(current_position_or_dy, "current_position")
            dx = float(p2[0] - p1[0])
            dy = float(p2[1] - p1[1])

        if dx == 0.0 and dy == 0.0:
            return 0.0

        angle_deg = math.degrees(math.atan2(dy, dx))
        return float(angle_deg % 360.0)

    def estimate_direction(self, p1: Any, p2: Any) -> float:
        """
        Backward-compatibility method returning heading angle in degrees [0.0, 360.0).

        Args:
            p1: Starting coordinate (x1, y1).
            p2: Ending coordinate (x2, y2).

        Returns:
            float: Angle in degrees.
        """
        return self.calculate_angle(p1, p2)

    def calculate_velocity(
        self,
        displacement: float,
        frame_interval: float = 1.0,
    ) -> float:
        """
        Calculate scalar image-space velocity:
            v = displacement / Δt

        Default frame_interval is 1.0 frame, yielding velocity in pixels/frame.
        If frame_interval = 1.0 / fps (or dt in seconds), velocity is in pixels/second.

        Args:
            displacement: Spatial displacement magnitude in pixels.
            frame_interval: Elapsed time or frame step (must be > 0).

        Returns:
            float: Velocity in pixels/frame or pixels/sec.
        """
        if frame_interval <= 0:
            raise ValueError("frame_interval must be strictly positive.")
        return float(displacement) / float(frame_interval)

    def estimate_velocity(
        self, displacement: float, dt: Optional[float] = None
    ) -> float:
        """
        Backward-compatibility method returning velocity in pixels/second.

        Args:
            displacement: Displacement magnitude in pixels.
            dt: Optional time delta in seconds. If None, derived from config.fps.

        Returns:
            float: Velocity in pixels/second.
        """
        time_step = dt if (dt is not None and dt > 0) else (1.0 / max(self.config.fps, 1.0))
        return self.calculate_velocity(displacement, time_step)

    # ------------------------------------------------------------------ #
    # Trajectory Analysis                                                #
    # ------------------------------------------------------------------ #

    def calculate_path_length(self, trajectory: List[Tuple[int, int]]) -> float:
        """
        Compute total accumulated path length along an object's trajectory:
            L = ∑_{i=1}^{n-1} sqrt((x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2)

        Args:
            trajectory: Ordered list of historical centroid coordinates.

        Returns:
            float: Total distance traversed in pixels.
        """
        if trajectory is None:
            raise ValueError("Trajectory cannot be None.")
        if len(trajectory) < 2:
            return 0.0

        total_length = 0.0
        for i in range(len(trajectory) - 1):
            p1 = self._validate_position(trajectory[i], f"trajectory[{i}]")
            p2 = self._validate_position(trajectory[i + 1], f"trajectory[{i+1}]")
            total_length += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        return float(total_length)

    def calculate_net_displacement(self, trajectory: List[Tuple[int, int]]) -> float:
        """
        Compute net displacement (straight-line distance between start and end):
            D = sqrt((xn - x1)^2 + (yn - y1)^2)

        Args:
            trajectory: Ordered list of historical centroid coordinates.

        Returns:
            float: Direct distance from origin to destination in pixels.
        """
        if trajectory is None:
            raise ValueError("Trajectory cannot be None.")
        if len(trajectory) < 2:
            return 0.0

        p_start = self._validate_position(trajectory[0], "trajectory[0]")
        p_end = self._validate_position(trajectory[-1], "trajectory[-1]")

        return float(math.hypot(p_end[0] - p_start[0], p_end[1] - p_start[1]))

    # ------------------------------------------------------------------ #
    # Track Analysis & Pipeline Update                                   #
    # ------------------------------------------------------------------ #

    def analyze_track(
        self,
        tracked_object: TrackedObject,
        optical_flow_points: Optional[List[OpticalFlowPoint]] = None,
    ) -> MotionData:
        """
        Compute all motion metrics for an individual tracked object.

        Args:
            tracked_object: Active TrackedObject instance.
            optical_flow_points: Optional optical flow features in frame.

        Returns:
            MotionData: Computed kinematics record.
        """
        if tracked_object is None:
            raise ValueError("tracked_object cannot be None.")
        if not isinstance(tracked_object, TrackedObject):
            raise ValueError(
                f"tracked_object must be an instance of TrackedObject (got {type(tracked_object).__name__})."
            )

        obj_id = tracked_object.object_id
        prev_pos = tracked_object.previous_position
        curr_pos = tracked_object.current_position
        trajectory = tracked_object.trajectory or [curr_pos]

        if prev_pos is None:
            # First observation of this object
            dx = 0.0
            dy = 0.0
            displacement = 0.0
            direction = "STATIONARY"
            angle = 0.0
            raw_velocity = 0.0
            approx_velocity = 0.0
        else:
            dx, dy, displacement = self.calculate_displacement(prev_pos, curr_pos)
            if displacement < self.stationary_threshold:
                direction = "STATIONARY"
                angle = 0.0
                raw_velocity = 0.0
                approx_velocity = 0.0
            else:
                direction = self.calculate_direction(prev_pos, curr_pos)
                angle = self.calculate_angle(dx, dy)
                raw_velocity = self.calculate_velocity(displacement, frame_interval=1.0)
                dt = 1.0 / max(self.config.fps, 1.0)
                approx_velocity = self.calculate_velocity(displacement, frame_interval=dt)

        # Maintain smoothing rolling window
        if obj_id not in self.recent_velocities:
            self.recent_velocities[obj_id] = []
            self.recent_displacements[obj_id] = []

        self.recent_velocities[obj_id].append(raw_velocity)
        self.recent_displacements[obj_id].append(displacement)

        if len(self.recent_velocities[obj_id]) > self.smoothing_window:
            self.recent_velocities[obj_id].pop(0)
            self.recent_displacements[obj_id].pop(0)

        smoothed_velocity = float(
            sum(self.recent_velocities[obj_id]) / len(self.recent_velocities[obj_id])
        )

        path_length = self.calculate_path_length(trajectory)
        net_disp = self.calculate_net_displacement(trajectory)

        m_data = MotionData(
            object_id=obj_id,
            dx=dx,
            dy=dy,
            displacement=displacement,
            direction=direction,
            angle=angle,
            velocity=smoothed_velocity,
            path_length=path_length,
            net_displacement=net_disp,
            approximate_velocity=approx_velocity,
            instantaneous_speed=displacement,
        )

        # Bounded historical accumulation
        if obj_id not in self.motion_history:
            self.motion_history[obj_id] = []
        self.motion_history[obj_id].append(m_data)

        if len(self.motion_history[obj_id]) > self.history_length:
            self.motion_history[obj_id].pop(0)

        return m_data

    def update(
        self,
        tracked_objects: List[TrackedObject],
        optical_flow_points: Optional[List[Any]] = None,
    ) -> Dict[int, MotionData]:
        """
        Analyze motion for all currently active tracked objects.

        Args:
            tracked_objects: List of TrackedObject instances from tracker.
            optical_flow_points: Optional optical flow features from flow analyzer.

        Returns:
            Dict[int, MotionData]: Mapping of object_id -> MotionData.
        """
        if tracked_objects is None:
            raise ValueError("tracked_objects cannot be None.")

        current_motion: Dict[int, MotionData] = {}
        for track in tracked_objects:
            m_data = self.analyze_track(track, optical_flow_points)
            current_motion[track.object_id] = m_data

        self.last_active_objects = current_motion
        return current_motion

    def reset(self) -> None:
        """
        Reset motion analysis state.

        Clears historical motion data, rolling smoothing buffers, and active object statistics.
        """
        self.motion_history.clear()
        self.recent_velocities.clear()
        self.recent_displacements.clear()
        self.last_active_objects.clear()

    # ------------------------------------------------------------------ #
    # Statistical Telemetry                                              #
    # ------------------------------------------------------------------ #

    def get_statistics(self) -> Dict[str, Any]:
        """
        Generate aggregate motion statistics for actively tracked objects.

        Returns:
            Dict[str, Any]: Metrics including active object counts, velocities, and per-object summaries.
        """
        active_objects = list(self.last_active_objects.values())
        total_active = len(active_objects)

        if total_active > 0:
            avg_disp = float(sum(m.displacement for m in active_objects) / total_active)
            avg_vel = float(sum(m.velocity for m in active_objects) / total_active)
            moving_count = sum(1 for m in active_objects if m.direction != "STATIONARY")
            stationary_count = sum(1 for m in active_objects if m.direction == "STATIONARY")
        else:
            avg_disp = 0.0
            avg_vel = 0.0
            moving_count = 0
            stationary_count = 0

        # Per-object statistics dictionary
        object_stats = {}
        for obj_id, m in self.last_active_objects.items():
            obs_count = len(self.motion_history.get(obj_id, []))
            object_stats[obj_id] = {
                "object_id": obj_id,
                "current_displacement": m.displacement,
                "current_direction": m.direction,
                "current_velocity": m.velocity,
                "current_angle": m.angle,
                "total_path_length": m.path_length,
                "net_displacement": m.net_displacement,
                "number_of_observations": obs_count,
            }

        return {
            "total_active_objects": total_active,
            "average_displacement": avg_disp,
            "average_velocity": avg_vel,
            "number_of_moving_objects": moving_count,
            "number_of_stationary_objects": stationary_count,
            "moving_objects": moving_count,
            "stationary_objects": stationary_count,
            "objects": object_stats,
            # Backward-compatibility keys
            "total_objects_tracked": len(self.motion_history),
            "average_velocity_px_s": avg_vel * self.config.fps,
            "max_velocity_px_s": max(
                (m.approximate_velocity for h in self.motion_history.values() for m in h),
                default=0.0,
            ),
            "total_displacement_px": sum(
                m.displacement for h in self.motion_history.values() for m in h
            ),
        }

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Backward-compatibility alias for get_statistics()."""
        return self.get_statistics()

    def reset(self) -> None:
        """Clear all historical motion tracking state."""
        self.motion_history.clear()
        self.recent_velocities.clear()
        self.recent_displacements.clear()
        self.last_active_objects.clear()
