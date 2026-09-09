"""
Unit tests for the object tracking module.

Verifies:
- ObjectTracker initialization.
- Dynamic object registration and unique ID allocation.
- Identity persistence across sequential frame updates.
- Trajectory recording and historical length management.
- Object deregistration upon prolonged disappearance.
"""

from src.config import TrackerConfig
from src.data_models import BoundingBox, Detection
from src.tracker import ObjectTracker


def _create_detection(x: int, y: int, w: int = 20, h: int = 20) -> Detection:
    """Helper to create dummy Detection objects."""
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    centroid = (x + w // 2, y + h // 2)
    return Detection(bbox=bbox, centroid=centroid, confidence=1.0, area=float(w * h))


def test_tracker_initialization():
    """Verify ObjectTracker initializes with empty registry."""
    tracker = ObjectTracker()
    assert tracker.next_object_id == 1
    assert len(tracker.objects) == 0


def test_tracker_registration():
    """Verify newly appearing detections are registered with incremental IDs."""
    tracker = ObjectTracker()
    det1 = _create_detection(50, 50)
    det2 = _create_detection(200, 200)

    tracks = tracker.update([det1, det2])
    assert len(tracks) == 2
    assert tracker.next_object_id == 3
    track_ids = {t.object_id for t in tracks}
    assert track_ids == {1, 2}


def test_tracker_identity_persistence():
    """Verify tracked object preserves its ID when moving within distance limits."""
    tracker = ObjectTracker(TrackerConfig(max_distance_threshold=50.0))

    # Frame 1: Object at (50, 50) -> centroid (60, 60)
    tracks_f1 = tracker.update([_create_detection(50, 50)])
    assert len(tracks_f1) == 1
    first_id = tracks_f1[0].object_id
    assert tracks_f1[0].current_position == (60, 60)
    assert tracks_f1[0].previous_position is None

    # Frame 2: Object moves slightly to (55, 55) -> centroid (65, 65)
    tracks_f2 = tracker.update([_create_detection(55, 55)])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].object_id == first_id
    assert tracks_f2[0].current_position == (65, 65)
    assert tracks_f2[0].previous_position == (60, 60)
    assert len(tracks_f2[0].trajectory) == 2


def test_tracker_deregistration():
    """Verify object is removed after exceeding max_disappeared frames."""
    tracker = ObjectTracker(TrackerConfig(max_disappeared=2))
    tracker.update([_create_detection(100, 100)])
    assert len(tracker.objects) == 1

    # Disappeared frame 1
    tracker.update([])
    assert len(tracker.objects) == 1

    # Disappeared frame 2
    tracker.update([])
    assert len(tracker.objects) == 1

    # Disappeared frame 3 (> max_disappeared = 2)
    tracks = tracker.update([])
    assert len(tracks) == 0
    assert len(tracker.objects) == 0
