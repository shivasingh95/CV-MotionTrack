"""
Motion analysis module for CV-MotionTrack.

Computes spatio-temporal kinematic parameters: displacement magnitude,
heading direction, instantaneous and average velocities, and cumulative motion statistics.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from src.config import MotionAnalysisConfig
from src.data_models import MotionData, TrackedObject


class MotionAnalyzer:
    """
    Estimates motion kinematics (displacement, direction, velocity)
    for tracked foreground entities over discrete time steps.

    Attributes:
        config: MotionAnalysisConfig holding sampling rates and smoothing thresholds.
        motion_history: Historical record of motion data keyed by object_id.
    """

    def __init__(self, config: Optional[MotionAnalysisConfig] = None) -> None:
        """
        Initialize the MotionAnalyzer.

        Args:
            config: Motion analysis configuration. If None, default settings are used.
        """
        self.config = config or MotionAnalysisConfig()
        self.motion_history: Dict[int, List[MotionData]] = {}

    def calculate_displacement(
        self, p1: Tuple[int, int], p2: Tuple[int, int]
    ) -> float:
        """
        Calculate Euclidean spatial displacement between two 2D points.

        Args:
            p1: Initial (x, y) coordinates.
            p2: Subsequent (x, y) coordinates.

        Returns:
            float: Spatial distance in pixels.
        """
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])
        return math.hypot(dx, dy)

    def estimate_direction(
        self, p1: Tuple[int, int], p2: Tuple[int, int]
    ) -> float:
        """
        Estimate heading direction angle in degrees from p1 to p2.
        Orientation is measured counterclockwise from the positive horizontal x-axis: [0, 360).

        Args:
            p1: Starting coordinate (x1, y1).
            p2: Ending coordinate (x2, y2).

        Returns:
            float: Direction angle in degrees [0.0, 360.0).
        """
        dx = float(p2[0] - p1[0])
        dy = float(p2[1] - p1[1])  # In image coordinates, y increases downward

        if dx == 0.0 and dy == 0.0:
            return 0.0

        angle_rad = math.atan2(dy, dx)
        angle_deg = math.degrees(angle_rad)
        return angle_deg % 360.0

    def estimate_velocity(self, displacement: float, dt: Optional[float] = None) -> float:
        """
        Compute scalar velocity in pixels per second.

        Args:
            displacement: Displacement magnitude in pixels.
            dt: Time elapsed between frames in seconds. If None, derived from config.fps.

        Returns:
            float: Velocity in pixels/sec.
        """
        time_step = dt if (dt is not None and dt > 0) else (1.0 / max(self.config.fps, 1.0))
        return displacement / time_step

    def update(
        self, tracked_objects: List[TrackedObject]
    ) -> Dict[int, MotionData]:
        """
        Update motion estimates for all actively tracked objects in the current frame.

        Args:
            tracked_objects: List of TrackedObject instances from tracker.

        Returns:
            Dict[int, MotionData]: Mapping of object_id -> estimated MotionData.
        """
        current_motion: Dict[int, MotionData] = {}
        dt = 1.0 / max(self.config.fps, 1.0)

        for track in tracked_objects:
            obj_id = track.object_id

            if track.previous_position is None:
                # First frame of observation for this object
                m_data = MotionData(
                    object_id=obj_id,
                    displacement=0.0,
                    direction=0.0,
                    approximate_velocity=0.0,
                    instantaneous_speed=0.0,
                )
            else:
                disp = self.calculate_displacement(
                    track.previous_position, track.current_position
                )

                if disp < self.config.min_displacement_threshold:
                    # Sub-threshold jitter suppression
                    direction = 0.0
                    velocity = 0.0
                    inst_speed = 0.0
                else:
                    direction = self.estimate_direction(
                        track.previous_position, track.current_position
                    )
                    velocity = self.estimate_velocity(disp, dt)
                    inst_speed = disp

                m_data = MotionData(
                    object_id=obj_id,
                    displacement=disp,
                    direction=direction,
                    approximate_velocity=velocity,
                    instantaneous_speed=inst_speed,
                )

            # Record into object motion history
            if obj_id not in self.motion_history:
                self.motion_history[obj_id] = []
            self.motion_history[obj_id].append(m_data)

            current_motion[obj_id] = m_data

        return current_motion

    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Generate aggregate statistical summary of motion across all tracked objects.

        Returns:
            Dict[str, Any]: Statistical metrics including total tracks, average velocities.
        """
        total_objects = len(self.motion_history)
        all_velocities: List[float] = []
        all_displacements: List[float] = []

        for history in self.motion_history.values():
            for m in history:
                if m.approximate_velocity > 0:
                    all_velocities.append(m.approximate_velocity)
                all_displacements.append(m.displacement)

        avg_velocity = (
            float(sum(all_velocities) / len(all_velocities)) if all_velocities else 0.0
        )
        max_velocity = float(max(all_velocities)) if all_velocities else 0.0
        total_displacement = float(sum(all_displacements)) if all_displacements else 0.0

        return {
            "total_objects_tracked": total_objects,
            "average_velocity_px_s": avg_velocity,
            "max_velocity_px_s": max_velocity,
            "total_displacement_px": total_displacement,
        }

    def reset(self) -> None:
        """Reset historical motion records."""
        self.motion_history.clear()
