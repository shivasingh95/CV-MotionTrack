"""
Data models for CV-MotionTrack.

Defines standardized data classes for inter-module communication across
the computer vision pipeline.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BoundingBox:
    """Represents an axis-aligned 2D bounding box (x, y, w, h)."""
    x: int
    y: int
    width: int
    height: int

    @property
    def top_left(self) -> Tuple[int, int]:
        """Return the (x, y) coordinates of the top-left corner."""
        return (self.x, self.y)

    @property
    def bottom_right(self) -> Tuple[int, int]:
        """Return the (x + width, y + height) coordinates of the bottom-right corner."""
        return (self.x + self.width, self.y + self.height)

    @property
    def area(self) -> int:
        """Return the pixel area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> Tuple[int, int]:
        """Return the integer centroid (cx, cy) of the bounding box."""
        return (self.x + self.width // 2, self.y + self.height // 2)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        """Return (x, y, width, height) tuple."""
        return (self.x, self.y, self.width, self.height)


@dataclass
class Detection:
    """
    Represents an individual object detection produced by the detector module.

    Attributes:
        bbox: BoundingBox object enclosing the candidate region.
        centroid: (cx, cy) center coordinate of the detection.
        confidence: Confidence score or normalized saliency metric (0.0 to 1.0).
        area: Area in pixels of the detected contour.
    """
    bbox: BoundingBox
    centroid: Tuple[int, int]
    confidence: float = 1.0
    area: float = 0.0


@dataclass
class TrackedObject:
    """
    Represents a persistent tracked entity across consecutive video frames.

    Attributes:
        object_id: Unique integer identifier assigned to this object.
        current_position: (cx, cy) current centroid coordinates in pixels.
        previous_position: (cx, cy) previous frame centroid coordinates (if any).
        trajectory: Chronological list of historical centroid positions.
        bbox: Current bounding box enclosing the object.
        disappeared_count: Number of consecutive frames this object was unobserved.
    """
    object_id: int
    current_position: Tuple[int, int]
    previous_position: Optional[Tuple[int, int]] = None
    trajectory: List[Tuple[int, int]] = field(default_factory=list)
    bbox: Optional[BoundingBox] = None
    disappeared_count: int = 0


@dataclass
class MotionData:
    """
    Kinematic motion attributes estimated for a tracked object.

    Attributes:
        object_id: Identifier of the tracked object.
        displacement: Magnitude of positional shift between consecutive frames in pixels.
        direction: Angle of displacement in degrees (0 to 360, relative to horizontal axis).
        approximate_velocity: Estimated velocity in pixels per second.
        instantaneous_speed: Speed in pixels per frame.
    """
    object_id: int
    displacement: float
    direction: float
    approximate_velocity: float
    instantaneous_speed: float = 0.0


@dataclass
class FrameMetadata:
    """
    Metadata associated with a captured video frame.

    Attributes:
        frame_index: Sequential zero-based frame counter.
        timestamp: Presentation timestamp in seconds.
        dimensions: (height, width, channels) of the frame.
    """
    frame_index: int
    timestamp: float
    dimensions: Tuple[int, int, int]
