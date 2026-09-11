"""
Object tracking module for CV-MotionTrack.

Implements classical centroid-based multi-object tracking:

  1. Pairwise Euclidean distance between previous and current centroids.
  2. Greedy nearest-centroid matching within a configurable distance threshold.
  3. Unique ID assignment for newly appearing objects.
  4. Temporary memory for objects that disappear momentarily (occlusion tolerance).
  5. Trajectory recording (historical centroid path per object).

This module is decoupled from the detector and the visualizer.
  - Input  : List[Detection]  (from src/detector.py)
  - Output : List[TrackedObject]  (consumed by src/visualizer.py and src/motion_analysis.py)

CSE3010 concepts demonstrated
──────────────────────────────
• Centroid-based tracking    – represent each object by its spatial centre
• Euclidean distance         – d = sqrt((x2-x1)^2 + (y2-y1)^2)
• Greedy nearest-neighbour   – match each track to the closest unmatched detection
• Disappearance tolerance    – keep tracks alive across brief detection gaps
• Trajectory maintenance     – record movement history for path visualization

Known Limitations of This Approach (important for academic report)
───────────────────────────────────────────────────────────────────
• Path crossings: when two objects cross trajectories the centroid of each can
  jump to the wrong track, causing an ID swap.
• Heavy occlusion: if an object is fully occluded for more than max_disappeared
  frames its ID is lost and a new ID is assigned when it reappears.
• Fast motion: objects that move more than max_distance pixels between consecutive
  frames will not be matched and will instead be registered as new objects.
• Greedy vs. optimal: greedy nearest-neighbour matching is not globally optimal.
  The Hungarian algorithm (scipy.optimize.linear_sum_assignment) would give a
  globally minimum-cost assignment but adds complexity; it can be substituted
  for this greedy approach in a later milestone.
"""

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np

from src.config import (
    MAX_TRAJECTORY_LENGTH,
    TRACKER_MAX_DISAPPEARED,
    TRACKER_MAX_DISTANCE,
    TrackerConfig,
)
from src.data_models import BoundingBox, Detection, TrackedObject


class ObjectTracker:
    """
    Maintains persistent object identities across consecutive video frames
    using a simple centroid-based Euclidean distance matching strategy.

    Each detected object is represented by a :class:`~src.data_models.TrackedObject`
    that stores its unique ID, current and previous centroids, bounding box,
    disappearance counter, and historical trajectory.

    Parameters
    ----------
    max_distance : float
        Maximum allowable Euclidean pixel distance between a track centroid and a
        detection centroid to consider them a match.  Detections further away than
        this threshold are treated as new objects.  Corresponds to ``TRACKER_MAX_DISTANCE``
        in ``config.py``.
    max_disappeared : int
        Maximum number of consecutive frames a track may go unmatched before it
        is permanently deregistered.  Corresponds to ``TRACKER_MAX_DISAPPEARED``.
    max_trajectory_length : int
        Maximum number of historical centroid positions retained per track.
        Older entries are discarded (FIFO) to bound memory usage.
        Corresponds to ``MAX_TRAJECTORY_LENGTH``.
    config : TrackerConfig, optional
        If provided, all keyword arguments above are ignored and values are drawn
        from the config object.  Allows the tracker to be driven from ``AppConfig``.

    Attributes
    ----------
    next_object_id : int
        Monotonically increasing counter; the ID assigned to the next new track.
    objects : Dict[int, TrackedObject]
        Live registry of currently active tracks, keyed by object_id.
    """

    def __init__(
        self,
        max_distance: Union[float, TrackerConfig] = TRACKER_MAX_DISTANCE,
        max_disappeared: int = TRACKER_MAX_DISAPPEARED,
        max_trajectory_length: int = MAX_TRAJECTORY_LENGTH,
        config: Optional[TrackerConfig] = None,
    ) -> None:
        if isinstance(max_distance, TrackerConfig):
            config = max_distance
            max_distance = TRACKER_MAX_DISTANCE

        if config is not None:
            max_distance          = config.max_distance
            max_disappeared       = config.max_disappeared
            max_trajectory_length = config.max_trajectory_length

        self.max_distance: float = max_distance
        self.max_disappeared: int = max_disappeared
        self.max_trajectory_length: int = max_trajectory_length

        self.next_object_id: int = 1
        self.objects: Dict[int, TrackedObject] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def register(self, detection: Detection) -> int:
        """
        Create and register a new tracked object from an unmatched detection.

        Assigns a monotonically increasing unique integer ID.  Stores the
        detection centroid as the object's initial position and first trajectory
        point.  Sets ``disappeared_count`` to 0.

        Parameters
        ----------
        detection : Detection
            Unmatched detection from the current frame.

        Returns
        -------
        int
            The unique object ID assigned to the new track.
        """
        track = TrackedObject(
            object_id=self.next_object_id,
            current_position=detection.centroid,
            previous_position=None,
            trajectory=[detection.centroid],
            bbox=detection.bbox,
            disappeared_count=0,
        )
        self.objects[self.next_object_id] = track
        assigned_id = self.next_object_id
        self.next_object_id += 1
        return assigned_id

    def deregister(self, object_id: int) -> None:
        """
        Permanently remove a track from the active registry.

        Called when a track's ``disappeared_count`` exceeds ``max_disappeared``.

        Parameters
        ----------
        object_id : int
            ID of the track to remove.
        """
        if object_id in self.objects:
            del self.objects[object_id]

    def update(self, detections: List[Detection]) -> List[TrackedObject]:
        """
        Associate *detections* from the current frame with existing tracks.

        Algorithm
        ---------
        Case A – No detections:
            Increment ``disappeared_count`` for every existing track.
            Deregister tracks that exceed ``max_disappeared``.

        Case B – No existing tracks:
            Register every detection as a new track.

        Case C – Both tracks and detections exist:
            1. Build pairwise Euclidean distance matrix
               ``D[i, j]`` = distance from track *i* centroid to detection *j* centroid.
            2. Sort tracks by their minimum distance to any detection (ascending).
            3. Greedy matching: for each track (in sorted order) take the closest
               unmatched detection, provided the distance ≤ ``max_distance``.
            4. Update matched tracks (centroid, bbox, trajectory, reset counter).
            5. Increment ``disappeared_count`` for unmatched tracks; deregister
               those that exceed the limit.
            6. Register unmatched detections as new tracks.

        Parameters
        ----------
        detections : List[Detection]
            Detections produced by :class:`~src.detector.ObjectDetector` for
            the current frame.

        Returns
        -------
        List[TrackedObject]
            All currently active tracked objects (matched + temporarily missing).
        """
        # ── Case A: empty detection list ──────────────────────────────────
        if len(detections) == 0:
            self._increment_disappeared_all()
            return list(self.objects.values())

        # ── Case B: no existing tracks ────────────────────────────────────
        if len(self.objects) == 0:
            for det in detections:
                self.register(det)
            return list(self.objects.values())

        # ── Case C: greedy nearest-centroid matching ──────────────────────
        object_ids   = list(self.objects.keys())
        track_pts    = np.array([self.objects[oid].current_position for oid in object_ids],
                                dtype=np.float32)
        det_pts      = np.array([d.centroid for d in detections], dtype=np.float32)

        # D shape: (num_tracks, num_detections)
        # Computed via broadcasting without an explicit loop
        D = np.linalg.norm(
            track_pts[:, np.newaxis, :] - det_pts[np.newaxis, :, :], axis=2
        )

        # Sort tracks by their closest detection (ascending minimum distance)
        sorted_track_indices = D.min(axis=1).argsort()

        used_track_indices: Set[int] = set()
        used_det_indices:   Set[int] = set()

        for ti in sorted_track_indices:
            # Among all unmatched detections, find the closest one
            remaining_dets = [j for j in range(len(detections)) if j not in used_det_indices]
            if not remaining_dets:
                break

            closest_det = min(remaining_dets, key=lambda j: D[ti, j])
            dist = D[ti, closest_det]

            if dist > self.max_distance:
                # Closest detection is still too far – no match for this track
                continue

            obj_id = object_ids[ti]
            self._update_track(self.objects[obj_id], detections[closest_det])
            used_track_indices.add(ti)
            used_det_indices.add(closest_det)

        # Unmatched tracks – increment disappeared counter
        unmatched_track_indices = set(range(len(object_ids))) - used_track_indices
        lost_ids: List[int] = []
        for ti in unmatched_track_indices:
            obj_id = object_ids[ti]
            self.objects[obj_id].disappeared_count += 1
            if self.objects[obj_id].disappeared_count > self.max_disappeared:
                lost_ids.append(obj_id)
        for obj_id in lost_ids:
            self.deregister(obj_id)

        # Unmatched detections – register as new tracks
        unmatched_det_indices = set(range(len(detections))) - used_det_indices
        for di in unmatched_det_indices:
            self.register(detections[di])

        return list(self.objects.values())

    def get_tracks(self) -> List[TrackedObject]:
        """
        Return a snapshot list of all currently active :class:`~src.data_models.TrackedObject`.

        This is the primary read interface for downstream modules (visualizer,
        motion analyser).

        Returns
        -------
        List[TrackedObject]
        """
        return list(self.objects.values())

    def get_trajectories(self) -> Dict[int, List[Tuple[int, int]]]:
        """
        Return a mapping of object_id → trajectory (list of historical centroids).

        Useful for the visualizer to draw movement paths.

        Returns
        -------
        Dict[int, List[Tuple[int, int]]]
            Keys are object IDs; values are ordered centroid history lists.
        """
        return {obj_id: list(track.trajectory)
                for obj_id, track in self.objects.items()}

    def reset(self) -> None:
        """
        Discard all active tracks and reset the ID counter to 1.

        Call this when switching video sources mid-session.
        """
        self.objects.clear()
        self.next_object_id = 1

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    def _update_track(self, track: TrackedObject, detection: Detection) -> None:
        """
        Apply a matched detection's data to an existing track in-place.

        - Saves the current position as ``previous_position``.
        - Updates ``current_position`` from the detection centroid.
        - Updates ``bbox`` from the detection bounding box.
        - Appends the new centroid to ``trajectory``; evicts the oldest entry
          if the trajectory exceeds ``max_trajectory_length``.
        - Resets ``disappeared_count`` to 0.

        Parameters
        ----------
        track : TrackedObject
            The existing track to update.
        detection : Detection
            The matched detection from the current frame.
        """
        track.previous_position = track.current_position
        track.current_position  = detection.centroid
        track.bbox              = detection.bbox

        track.trajectory.append(detection.centroid)
        if len(track.trajectory) > self.max_trajectory_length:
            track.trajectory.pop(0)   # FIFO – discard oldest point

        track.disappeared_count = 0

    def _increment_disappeared_all(self) -> None:
        """
        Increment ``disappeared_count`` for every active track and deregister
        those exceeding ``max_disappeared``.

        Called when the current frame produces zero detections.
        """
        lost_ids: List[int] = []
        for obj_id, track in self.objects.items():
            track.disappeared_count += 1
            if track.disappeared_count > self.max_disappeared:
                lost_ids.append(obj_id)
        for obj_id in lost_ids:
            self.deregister(obj_id)

    # ------------------------------------------------------------------ #
    # Convenience / read-only properties                                  #
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """
        Reset tracker state.

        Clears all active tracks and resets the ID allocation counter back to 1.
        """
        self.next_object_id = 1
        self.objects.clear()

    @property
    def active_count(self) -> int:
        """Number of currently active tracked objects."""
        return len(self.objects)

    @staticmethod
    def euclidean_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        """
        Compute the Euclidean distance between two 2-D integer points.

        d = sqrt((x2 - x1)^2 + (y2 - y1)^2)

        Exposed as a static method so it can be unit-tested independently.

        Parameters
        ----------
        p1, p2 : Tuple[int, int]
            Points as (x, y) pixel coordinates.

        Returns
        -------
        float
            Distance in pixels.
        """
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])
