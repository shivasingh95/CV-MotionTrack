"""
Object tracking module for CV-MotionTrack.

Implements classical multi-object tracking via centroid Euclidean distance matching,
maintaining persistent object IDs, state histories, and trajectory curves.
"""

from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from src.config import TrackerConfig
from src.data_models import Detection, TrackedObject


class ObjectTracker:
    """
    Maintains persistent identities and historical trajectories for detected objects
    across consecutive video frames.

    Attributes:
        config: TrackerConfig specifying association limits and memory length.
        next_object_id: Incrementing integer counter for assigning unique object IDs.
        objects: Internal registry mapping object_id -> TrackedObject.
    """

    def __init__(self, config: Optional[TrackerConfig] = None) -> None:
        """
        Initialize the ObjectTracker.

        Args:
            config: Tracker configuration. If None, default settings are used.
        """
        self.config = config or TrackerConfig()
        self.next_object_id: int = 1
        self.objects: Dict[int, TrackedObject] = {}

    def _register(self, detection: Detection) -> None:
        """Register a new object detection with a unique ID."""
        new_track = TrackedObject(
            object_id=self.next_object_id,
            current_position=detection.centroid,
            previous_position=None,
            trajectory=[detection.centroid],
            bbox=detection.bbox,
            disappeared_count=0,
        )
        self.objects[self.next_object_id] = new_track
        self.next_object_id += 1

    def _deregister(self, object_id: int) -> None:
        """Remove a lost object from the active tracking registry."""
        if object_id in self.objects:
            del self.objects[object_id]

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Associate new detections with existing tracked objects using centroid distance.

        Args:
            detections: List of Detection instances discovered in the current frame.

        Returns:
            List[TrackedObject]: Currently active tracked objects.
        """
        # Case 1: No detections present in current frame
        if len(detections) == 0:
            lost_ids: List[int] = []
            for obj_id, track in self.objects.items():
                track.disappeared_count += 1
                if track.disappeared_count > self.config.max_disappeared:
                    lost_ids.append(obj_id)
            for obj_id in lost_ids:
                self._deregister(obj_id)
            return list(self.objects.values())

        # Case 2: No existing tracked objects; register all new detections
        if len(self.objects) == 0:
            for det in detections:
                self._register(det)
            return list(self.objects.values())

        # Case 3: Match existing tracks with current detections using Euclidean distance
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[oid].current_position for oid in object_ids]
        detection_centroids = [det.centroid for det in detections]

        # Construct pairwise Euclidean distance matrix
        obj_pts = np.array(object_centroids)
        det_pts = np.array(detection_centroids)

        # Distances shape: (num_objects, num_detections)
        distances = np.linalg.norm(obj_pts[:, np.newaxis] - det_pts[np.newaxis, :], axis=2)

        # Sort matrix entries to associate greedily by smallest distance
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]

        used_rows: Set[int] = set()
        used_cols: Set[int] = set()

        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # If distance exceeds threshold, do not associate
            if distances[row, col] > self.config.max_distance_threshold:
                continue

            obj_id = object_ids[row]
            det = detections[col]
            track = self.objects[obj_id]

            # Update tracked object state
            track.previous_position = track.current_position
            track.current_position = det.centroid
            track.bbox = det.bbox
            track.trajectory.append(det.centroid)
            if len(track.trajectory) > self.config.max_trajectory_length:
                track.trajectory.pop(0)
            track.disappeared_count = 0

            used_rows.add(row)
            used_cols.add(col)

        # Handle unmatched existing objects (increment disappeared)
        unused_rows = set(range(len(object_ids))) - used_rows
        lost_ids = []
        for row in unused_rows:
            obj_id = object_ids[row]
            track = self.objects[obj_id]
            track.disappeared_count += 1
            if track.disappeared_count > self.config.max_disappeared:
                lost_ids.append(obj_id)
        for obj_id in lost_ids:
            self._deregister(obj_id)

        # Handle unmatched detections (register as newly appearing objects)
        unused_cols = set(range(len(detections))) - used_cols
        for col in unused_cols:
            self._register(detections[col])

        return list(self.objects.values())

    def reset(self) -> None:
        """Clear all active tracks and reset the ID counter."""
        self.objects.clear()
        self.next_object_id = 1
