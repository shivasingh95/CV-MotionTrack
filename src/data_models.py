"""
Data models for CV-MotionTrack.

Defines standardized data classes for inter-module communication across
the computer vision pipeline.
"""

from dataclasses import dataclass, field
import math
from typing import List, Optional, Tuple, Union


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
        dx: Horizontal positional shift in pixels (x_current - x_previous).
        dy: Vertical positional shift in pixels (y_current - y_previous).
        displacement: Euclidean distance moved in current frame (pixels).
        direction: Qualitative image-space direction ('RIGHT', 'UP-LEFT', 'STATIONARY', etc.).
        angle: Heading angle in degrees [0, 360) relative to positive horizontal axis.
        velocity: Image-space velocity in pixels/frame.
        path_length: Cumulative trajectory path length in pixels.
        net_displacement: Direct straight-line distance from trajectory start to end.
        approximate_velocity: Velocity in pixels per second (derived from FPS).
        instantaneous_speed: Instantaneous frame-to-frame displacement in pixels.
    """
    object_id: int
    dx: float = 0.0
    dy: float = 0.0
    displacement: float = 0.0
    direction: Union[str, float] = "STATIONARY"
    angle: float = 0.0
    velocity: float = 0.0
    path_length: float = 0.0
    net_displacement: float = 0.0
    approximate_velocity: float = 0.0
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


@dataclass
class OpticalFlowPoint:
    """
    Represents an individual sparse feature point tracked across consecutive video frames
    using Lucas-Kanade optical flow.

    Attributes:
        previous_x: X-coordinate in previous frame (x_previous).
        previous_y: Y-coordinate in previous frame (y_previous).
        current_x: X-coordinate in current frame (x_current).
        current_y: Y-coordinate in current frame (y_current).
        dx: Apparent motion displacement vector along X axis (x_current - x_previous).
        dy: Apparent motion displacement vector along Y axis (y_current - y_previous).
    """
    previous_x: float
    previous_y: float
    current_x: float
    current_y: float
    dx: float
    dy: float

    @property
    def previous_point(self) -> Tuple[float, float]:
        """Return previous position as (x, y) tuple."""
        return (self.previous_x, self.previous_y)

    @property
    def current_point(self) -> Tuple[float, float]:
        """Return current position as (x, y) tuple."""
        return (self.current_x, self.current_y)

    @property
    def vector(self) -> Tuple[float, float]:
        """Return apparent motion displacement vector as (dx, dy) tuple."""
        return (self.dx, self.dy)

    @property
    def magnitude(self) -> float:
        """
        Calculate Euclidean motion vector magnitude:
        magnitude = sqrt(dx^2 + dy^2).
        """
        return math.hypot(self.dx, self.dy)
