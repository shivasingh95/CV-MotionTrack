"""
Visualization module for CV-MotionTrack.

Academic Computer Vision project for CSE3010.
Renders:
- Detections and bounding boxes
- Persistent object identities (ID: <num>)
- Historical trajectory trails
- Lucas-Kanade optical flow motion vectors
- Kinematic motion info badges (direction, velocity, displacement)
- Telemetry HUD statistics panel
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import VisualizationConfig
from src.data_models import Detection, MotionData, OpticalFlowPoint, TrackedObject


class Visualizer:
    """
    Renders computer vision annotations, feature tracking vectors,
    and motion diagnostics onto video frames using OpenCV.

    Attributes:
        config: VisualizationConfig defining styling, color palette, and display flags.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        """
        Initialize the Visualizer.

        Args:
            config: Visualization configuration instance. If None, default settings are used.
        """
        self.config = config or VisualizationConfig()

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw candidate contour bounding boxes and detection centroids.

        Args:
            frame: Input video frame (BGR).
            detections: List of candidate Detection instances from ObjectDetector.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not self.config.show_bboxes or not detections:
            return frame

        canvas = frame.copy()
        for det in detections:
            bbox = det.bbox
            cv2.rectangle(
                canvas,
                bbox.top_left,
                bbox.bottom_right,
                self.config.bbox_color,
                self.config.line_thickness,
            )
            if self.config.show_centroids:
                cv2.circle(
                    canvas,
                    det.centroid,
                    3,
                    self.config.centroid_color,
                    -1,
                )
        return canvas

    def draw_tracks(
        self, frame: np.ndarray, tracked_objects: List[TrackedObject]
    ) -> np.ndarray:
        """
        Draw tracked object bounding boxes and prominent object ID labels.

        For each active object:
        - Bounding rectangle
        - Object ID tag: "ID: <object_id>"

        Args:
            frame: Video frame (BGR).
            tracked_objects: List of actively tracked objects.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not tracked_objects:
            return frame

        canvas = frame.copy()
        for track in tracked_objects:
            cx, cy = track.current_position

            # Draw bounding box if available
            if track.bbox is not None and self.config.show_bboxes:
                bx, by, bw, bh = track.bbox.x, track.bbox.y, track.bbox.width, track.bbox.height
                cv2.rectangle(
                    canvas,
                    (bx, by),
                    (bx + bw, by + bh),
                    self.config.bbox_color,
                    self.config.line_thickness,
                )
                tag_x, tag_y = bx, max(18, by - 6)
            else:
                tag_x, tag_y = cx, max(18, cy - 10)

            # Draw centroid point
            if self.config.show_centroids:
                cv2.circle(canvas, (cx, cy), 4, self.config.centroid_color, -1)

            # Draw Object ID label badge
            if self.config.show_ids:
                label = f"ID: {track.object_id}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                scale = self.config.font_scale
                (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)

                # Background badge behind label for legibility
                bg_p1 = (tag_x, tag_y - th - 4)
                bg_p2 = (tag_x + tw + 6, tag_y + 2)
                cv2.rectangle(canvas, bg_p1, bg_p2, (0, 0, 0), -1)
                cv2.putText(
                    canvas,
                    label,
                    (tag_x + 3, tag_y - 2),
                    font,
                    scale,
                    self.config.text_color,
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def draw_trajectories(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        max_history: Optional[int] = None,
    ) -> np.ndarray:
        """
        Draw continuous polyline trajectory trails connecting historical centroids.

        Demonstrates spatio-temporal tracking:
        point1 -> point2 -> ... -> current_point

        Args:
            frame: Video frame (BGR).
            tracked_objects: Active tracked objects.
            max_history: Max historical points to render (defaults to config length).

        Returns:
            np.ndarray: Annotated frame.
        """
        if not self.config.show_trajectories or not tracked_objects:
            return frame

        canvas = frame.copy()
        for track in tracked_objects:
            traj = track.trajectory
            if not traj or len(traj) < 2:
                continue

            # Limit to recent history window
            if max_history is not None and len(traj) > max_history:
                points = traj[-max_history:]
            else:
                points = traj

            pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(
                canvas,
                [pts],
                isClosed=False,
                color=self.config.trajectory_color,
                thickness=max(1, self.config.line_thickness - 1),
                lineType=cv2.LINE_AA,
            )

        return canvas

    def draw_optical_flow(
        self,
        frame: np.ndarray,
        optical_flow_points: List[OpticalFlowPoint],
        max_points: Optional[int] = None,
    ) -> np.ndarray:
        """
        Render Lucas-Kanade optical flow motion vectors.

        For each valid point:
        previous point ------> current point

        Visual vectors are styled distinctly (Cyan) from object trajectories (Yellow).

        Args:
            frame: Video frame (BGR).
            optical_flow_points: List of valid OpticalFlowPoint instances.
            max_points: Optional cap on rendered points to prevent screen clutter.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not getattr(self.config, "show_optical_flow", True) or not optical_flow_points:
            return frame

        canvas = frame.copy()
        color = getattr(self.config, "optical_flow_color", (255, 255, 0))
        limit = max_points or getattr(self.config, "max_displayed_flow_points", 150)

        points_to_draw = optical_flow_points[:limit]
        for pt in points_to_draw:
            p0 = (int(round(pt.previous_x)), int(round(pt.previous_y)))
            p1 = (int(round(pt.current_x)), int(round(pt.current_y)))

            # Only draw arrows if there is perceptible displacement
            if pt.magnitude >= 0.5:
                cv2.arrowedLine(
                    canvas,
                    p0,
                    p1,
                    color,
                    1,
                    tipLength=0.3,
                    line_type=cv2.LINE_AA,
                )
            else:
                cv2.circle(canvas, p1, 2, color, -1)

        return canvas

    def draw_motion_info(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        motion_data: Dict[int, MotionData],
    ) -> np.ndarray:
        """
        Display compact kinematic telemetry badges adjacent to each tracked object.

        Displays:
        - ID: <id>
        - Dir: <direction>
        - Vel: <velocity> px/s (or px/frame)
        - Disp: <displacement> px

        Safe fallback placeholders are displayed if motion data is not yet computed.

        Args:
            frame: Video frame (BGR).
            tracked_objects: List of active tracks.
            motion_data: Mapping of object_id -> MotionData.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not getattr(self.config, "show_motion_info", True) or not tracked_objects:
            return frame

        canvas = frame.copy()
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.38
        line_height = 14

        for track in tracked_objects:
            obj_id = track.object_id
            m = motion_data.get(obj_id)

            # Determine anchor position (below bounding box or to the right)
            if track.bbox is not None:
                x = track.bbox.x
                y = track.bbox.y + track.bbox.height + 14
            else:
                x = track.current_position[0] + 10
                y = track.current_position[1] + 14

            # Ensure coordinates stay within frame bounds
            x = max(5, min(x, canvas.shape[1] - 150))
            y = max(20, min(y, canvas.shape[0] - 50))

            # Format information strings with graceful fallbacks
            if m is not None:
                dir_str = m.direction
                disp_str = f"{m.displacement:.1f} px"
                # Use approximate_velocity (px/s) if available, else velocity (px/frame)
                vel_val = m.approximate_velocity if m.approximate_velocity > 0 else m.velocity
                vel_unit = "px/s" if m.approximate_velocity > 0 else "px/f"
                vel_str = f"{vel_val:.1f} {vel_unit}"
            else:
                dir_str = "STATIONARY"
                disp_str = "0.0 px"
                vel_str = "0.0 px/f"

            lines = [
                f"Dir: {dir_str}",
                f"Vel: {vel_str}",
                f"Dsp: {disp_str}",
            ]

            # Draw compact semi-transparent background box
            box_w = 115
            box_h = len(lines) * line_height + 6
            overlay = canvas.copy()
            cv2.rectangle(
                overlay,
                (x - 2, y - 10),
                (x + box_w, y - 10 + box_h),
                (15, 15, 15),
                -1,
            )
            cv2.addWeighted(overlay, 0.65, canvas, 0.35, 0, canvas)

            # Draw telemetry lines
            for i, line in enumerate(lines):
                cv2.putText(
                    canvas,
                    line,
                    (x + 2, y + i * line_height),
                    font,
                    font_scale,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    def draw_statistics(
        self,
        frame: np.ndarray,
        statistics: Optional[Dict[str, Any]] = None,
        fps: float = 0.0,
        flow_points_count: int = 0,
        tracked_objects: Optional[List[TrackedObject]] = None,
        motion_data: Optional[Dict[int, MotionData]] = None,
    ) -> np.ndarray:
        """
        Render a clean HUD statistics panel on the frame.

        Metrics displayed:
        - Active Objects
        - Moving Objects
        - Stationary Objects
        - Average Velocity
        - Average Displacement
        - Optical Flow Points
        - Instantaneous FPS

        Args:
            frame: Video frame (BGR).
            statistics: Summary statistics dictionary from MotionAnalyzer.
            fps: Processing throughput in frames per second.
            flow_points_count: Number of active optical flow points.
            tracked_objects: Optional active tracks.
            motion_data: Optional motion estimates.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not self.config.show_dashboard:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Extract telemetry parameters with safe fallbacks
        stats = statistics or {}
        active = stats.get("total_active_objects", len(tracked_objects) if tracked_objects else 0)
        moving = stats.get("moving_objects", 0)
        stationary = stats.get("stationary_objects", 0)
        avg_vel = stats.get("average_velocity_px_s", stats.get("average_velocity_px_frame", 0.0))
        avg_disp = stats.get("average_displacement_px", 0.0)

        # Draw semi-transparent header bar across top
        overlay = canvas.copy()
        panel_height = 36
        cv2.rectangle(overlay, (0, 0), (w, panel_height), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        # Construct primary HUD line
        flow_str = f" | Flow: {flow_points_count}" if flow_points_count > 0 else ""
        hud_text = (
            f"CV-MotionTrack | FPS: {fps:.1f} | "
            f"Objects: {active} (Mov: {moving}, Stat: {stationary}){flow_str} | "
            f"Avg Vel: {avg_vel:.1f} px/s"
        )

        cv2.putText(
            canvas,
            hud_text,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def draw_motion_vectors(
        self,
        frame: np.ndarray,
        tracked_objects: List[TrackedObject],
        motion_data: Dict[int, MotionData],
        vector_scale: float = 3.0,
    ) -> np.ndarray:
        """
        Legacy compatibility helper for drawing object motion arrows from centroid delta.

        Args:
            frame: Video frame (BGR).
            tracked_objects: List of TrackedObject instances.
            motion_data: Mapping of object_id -> MotionData.
            vector_scale: Multiplier for vector length visualization.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not self.config.show_motion_vectors or not tracked_objects:
            return frame

        canvas = frame.copy()
        for track in tracked_objects:
            obj_id = track.object_id
            m = motion_data.get(obj_id)
            if m is None or m.displacement < 1.0:
                continue

            cx, cy = track.current_position
            if track.previous_position is not None:
                dx = cx - track.previous_position[0]
                dy = cy - track.previous_position[1]
                target_x = int(round(cx + dx * vector_scale))
                target_y = int(round(cy + dy * vector_scale))

                cv2.arrowedLine(
                    canvas,
                    (cx, cy),
                    (target_x, target_y),
                    self.config.vector_color,
                    2,
                    tipLength=0.3,
                )

        return canvas

    def draw_dashboard(
        self,
        frame: np.ndarray,
        active_count: int = 0,
        fps: float = 0.0,
        stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """Legacy compatibility alias for draw_statistics."""
        return self.draw_statistics(frame, statistics=stats, fps=fps)

    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracked_objects: List[TrackedObject],
        motion_data: Dict[int, MotionData],
        optical_flow_points: Optional[List[OpticalFlowPoint]] = None,
        active_count: Optional[int] = None,
        fps: float = 0.0,
        stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Coordinate complete composite visualization pipeline on a single frame.

        Visual layer hierarchy:
        1. Base frame
        2. Detection boxes (if enabled)
        3. Trajectory trails
        4. Optical flow feature motion vectors
        5. Tracked bounding boxes and Object IDs
        6. Centroid directional vectors
        7. Kinematic motion info badges (direction, velocity, displacement)
        8. Top telemetry HUD statistics panel

        Args:
            frame: Raw BGR video frame.
            detections: Current detections from ObjectDetector.
            tracked_objects: Active tracks from ObjectTracker.
            motion_data: Motion kinematic estimates from MotionAnalyzer.
            optical_flow_points: Optional feature points from OpticalFlowAnalyzer.
            active_count: Optional active track count.
            fps: Current estimated processing frame rate.
            stats: Summary metrics dictionary from MotionAnalyzer.

        Returns:
            np.ndarray: Fully annotated composite frame.
        """
        out = frame.copy()

        # 1. Trajectories (drawn under objects)
        out = self.draw_trajectories(out, tracked_objects)

        # 2. Optical flow vectors
        if optical_flow_points:
            out = self.draw_optical_flow(out, optical_flow_points)

        # 3. Tracked objects (BBoxes + IDs)
        out = self.draw_tracks(out, tracked_objects)

        # 4. Motion vectors
        out = self.draw_motion_vectors(out, tracked_objects, motion_data)

        # 5. Kinematic motion labels (ID, Dir, Vel, Disp)
        out = self.draw_motion_info(out, tracked_objects, motion_data)

        # 6. Statistics HUD panel
        flow_pts_count = len(optical_flow_points) if optical_flow_points else 0
        out = self.draw_statistics(
            out,
            statistics=stats,
            fps=fps,
            flow_points_count=flow_pts_count,
            tracked_objects=tracked_objects,
            motion_data=motion_data,
        )

        return out
