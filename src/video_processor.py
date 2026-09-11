"""
Video processing and acquisition module for CV-MotionTrack.

Handles stream initialization, frame acquisition, validation of input sources
(webcam or video file), and graceful resource cleanup.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import VideoConfig


class VideoProcessor:
    """
    Manages video frame acquisition from a camera device or recorded video file.

    Attributes:
        config: VideoConfig instance holding acquisition parameters.
        source: Device index or file system path.
        capture: Underlying OpenCV VideoCapture instance.
        frame_counter: Sequential count of read frames.
    """

    def __init__(
        self,
        config: Optional[VideoConfig] = None,
        source: Optional[Union[int, str]] = None,
    ) -> None:
        """
        Initialize the VideoProcessor.

        Args:
            config: Video configuration instance. If None, default settings are used.
            source: Optional override for the video source specified in config.
        """
        self.config = config or VideoConfig()
        self.source = source if source is not None else self.config.source
        self.capture: Optional[cv2.VideoCapture] = None
        self.frame_counter: int = 0

    def open(self) -> bool:
        """
        Open the video source (webcam or file) and validate readiness.

        Returns:
            bool: True if source opened successfully, False otherwise.

        Raises:
            FileNotFoundError: If the specified video file path does not exist.
        """
        if isinstance(self.source, str):
            if self.source.isdigit():
                self.source = int(self.source)
                self.capture = cv2.VideoCapture(self.source)
            else:
                if not os.path.exists(self.source):
                    raise FileNotFoundError(f"Video file not found at path: {self.source}")
                self.capture = cv2.VideoCapture(self.source)
        else:
            self.capture = cv2.VideoCapture(self.source)

        if not self.capture.isOpened():
            return False

        # Attempt to set desired resolution for cameras if configured
        if isinstance(self.source, int):
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.target_width)
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.target_height)

        self.frame_counter = 0
        return True

    def is_opened(self) -> bool:
        """Check whether the underlying video stream is active and open."""
        return self.capture is not None and self.capture.isOpened()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read the next video frame from the active stream.

        Returns:
            Tuple[bool, Optional[np.ndarray]]: (True, frame) if successfully acquired,
            or (False, None) if end-of-stream or read error occurs.
        """
        if not self.is_opened():
            return False, None

        ret, frame = self.capture.read()
        if not ret or frame is None:
            return False, None

        if self.config.enforce_target_size:
            frame = cv2.resize(
                frame,
                (self.config.target_width, self.config.target_height),
                interpolation=cv2.INTER_AREA,
            )

        self.frame_counter += 1
        return True, frame

    def get_properties(self) -> Dict[str, Any]:
        """
        Retrieve metadata and properties of the active video stream.

        Returns:
            Dict[str, Any]: Dictionary containing fps, dimensions, and frame counts.
        """
        if not self.is_opened():
            return {
                "source": self.source,
                "is_opened": False,
                "frame_count": self.frame_counter,
                "fps": 0.0,
                "width": 0,
                "height": 0,
            }

        fps = self.capture.get(cv2.CAP_PROP_FPS)
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

        return {
            "source": self.source,
            "is_opened": True,
            "frame_count": self.frame_counter,
            "fps": fps if fps > 0 else self.config.fps_limit,
            "width": width,
            "height": height,
            "total_frames": total_frames,
        }

    @property
    def width(self) -> int:
        """Frame width in pixels (or 0 if not opened)."""
        if not self.is_opened():
            return 0
        return int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        """Frame height in pixels (or 0 if not opened)."""
        if not self.is_opened():
            return 0
        return int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        """
        Stream acquisition rate in frames per second (FPS).

        Academic Limitation Note:
            If capturing from an uncalibrated webcam where cv2.CAP_PROP_FPS is 0.0 or
            unavailable, a safe fallback (config.fps_limit) is returned for timing and
            display purposes. This fallback is purely for image-space timing and must
            NOT be confused with calibrated physical-world speed.
        """
        if not self.is_opened():
            return 0.0
        reported_fps = self.capture.get(cv2.CAP_PROP_FPS)
        return reported_fps if reported_fps > 0 else self.config.fps_limit

    def release(self) -> None:
        """Release video capture resources."""
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def __enter__(self) -> "VideoProcessor":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
