"""
Visualization module for CV-MotionTrack.

Renders bounding boxes, unique object IDs, historical trajectory curves,
directional motion vectors, and analytical telemetry HUD on video frames.
"""

from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from src.config import VisualizationConfig
from src.data_models import Detection, MotionData, TrackedObject


class Visualizer:
    """
    Renders computer vision annotations and motion diagnostics onto video frames.

    Attributes:
        config: VisualizationConfig defining colors, line widths, and display flags.
    """

    def __init__(self, config: Optional[VisualizationConfig] = None) -> None:
        """
        Initialize the Visualizer.

        Args:
            config: Visualization configuration. If None, default settings are used.
        """
        self.config = config or VisualizationConfig()

    def draw_detections(
        self, frame: np.ndarray, detections: List[Detection]
    ) -> np.ndarray:
        """
        Draw candidate bounding boxes and detection centroids.

        Args:
            frame: Input video frame (BGR).
            detections: List of Detection instances.

        Returns:
            np.ndarray: Annotated frame.
        """
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
                    4,
                    self.config.centroid_color,
                    -1,
                )
        return canvas

    def draw_tracks(
        self, frame: np.ndarray, tracked_objects: List[TrackedObject]
    ) -> np.ndarray:
        """
        Draw object identities, current centroids, and historical trajectory curves.

        Args:
            frame: Video frame (BGR).
            tracked_objects: List of TrackedObject instances.

        Returns:
            np.ndarray: Annotated frame.
        """
        canvas = frame.copy()
        for track in tracked_objects:
            cx, cy = track.current_position

            # Draw trajectory trail
            if self.config.show_trajectories and len(track.trajectory) > 1:
                pts = np.array(track.trajectory, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(
                    canvas,
                    [pts],
                    isClosed=False,
                    color=self.config.trajectory_color,
                    thickness=max(1, self.config.line_thickness - 1),
                )

            # Draw current centroid
            if self.config.show_centroids:
                cv2.circle(canvas, (cx, cy), 5, self.config.centroid_color, -1)

            # Draw Object ID text label
            if self.config.show_ids:
                label = f"ID: {track.object_id}"
                # Position label above bounding box if available, else above centroid
                if track.bbox is not None:
                    lx, ly = track.bbox.x, max(15, track.bbox.y - 8)
                else:
                    lx, ly = cx, max(15, cy - 10)

                cv2.putText(
                    canvas,
                    label,
                    (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.config.font_scale,
                    self.config.text_color,
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
        Render directional arrows indicating object movement vectors.

        Args:
            frame: Video frame (BGR).
            tracked_objects: List of TrackedObject instances.
            motion_data: Mapping of object_id -> MotionData.
            vector_scale: Scaling multiplier for arrow length visualization.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not self.config.show_motion_vectors:
            return frame

        canvas = frame.copy()
        for track in tracked_objects:
            obj_id = track.object_id
            m = motion_data.get(obj_id)
            if m is None or m.displacement <= 0.5:
                continue

            cx, cy = track.current_position
            if track.previous_position is not None:
                dx = cx - track.previous_position[0]
                dy = cy - track.previous_position[1]
                target_x = int(cx + dx * vector_scale)
                target_y = int(cy + dy * vector_scale)

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
        active_count: int,
        fps: float,
        stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Render a semi-transparent diagnostic telemetry dashboard header on the frame.

        Args:
            frame: Video frame (BGR).
            active_count: Number of active tracks in current frame.
            fps: Current processing throughput in frames per second.
            stats: Optional statistical metrics dictionary.

        Returns:
            np.ndarray: Annotated frame.
        """
        if not self.config.show_dashboard:
            return frame

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # Draw semi-transparent header overlay (top 35 pixels)
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

        hud_text = f"CV-MotionTrack | FPS: {fps:.1f} | Active Objects: {active_count}"
        if stats and "average_velocity_px_s" in stats:
            hud_text += f" | Avg Vel: {stats['average_velocity_px_s']:.1f} px/s"

        cv2.putText(
            canvas,
            hud_text,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        return canvas

    def render(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        tracked_objects: List[TrackedObject],
        motion_data: Dict[int, MotionData],
        active_count: int,
        fps: float = 0.0,
        stats: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Coordinate complete composite visualization pipeline on a single frame.

        Args:
            frame: Raw BGR video frame.
            detections: Current detections.
            tracked_objects: Active tracks.
            motion_data: Motion kinematic estimates.
            active_count: Number of active objects.
            fps: Current frame rate.
            stats: Summary metrics.

        Returns:
            np.ndarray: Fully annotated composite frame.
        """
        out = frame.copy()
        if self.config.show_bboxes:
            out = self.draw_detections(out, detections)
        out = self.draw_tracks(out, tracked_objects)
        out = self.draw_motion_vectors(out, tracked_objects, motion_data)
        out = self.draw_dashboard(out, active_count=active_count, fps=fps, stats=stats)
        return out
