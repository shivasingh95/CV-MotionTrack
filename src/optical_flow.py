"""
Optical flow analysis module for CV-MotionTrack.

Provides clean interfaces for estimating apparent pixel motion between consecutive
video frames using classical differential techniques (Lucas-Kanade / KLT features).
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import OpticalFlowConfig


class OpticalFlowAnalyzer:
    """
    Computes sparse optical flow vectors between consecutive video frames
    using the Lucas-Kanade pyramidal algorithm.

    Attributes:
        config: OpticalFlowConfig with feature detection and window hyperparameters.
        prev_gray: Previous frame cached as single-channel grayscale.
        prev_points: Cached sparse feature points tracked from the previous frame.
    """

    def __init__(self, config: Optional[OpticalFlowConfig] = None) -> None:
        """
        Initialize the OpticalFlowAnalyzer.

        Args:
            config: Configuration settings for feature detection and optical flow.
        """
        self.config = config or OpticalFlowConfig()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None

        # Feature detection parameters (Shi-Tomasi corner detector)
        self.feature_params = dict(
            maxCorners=self.config.feature_max_corners,
            qualityLevel=self.config.feature_quality_level,
            minDistance=self.config.feature_min_distance,
            blockSize=self.config.feature_block_size,
        )

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=self.config.lk_win_size,
            maxLevel=self.config.lk_max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )

    def detect_features(
        self, gray_frame: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Identify prominent corner features suitable for differential tracking.

        Args:
            gray_frame: Grayscale image array.
            mask: Optional region-of-interest binary mask.

        Returns:
            Optional[np.ndarray]: Array of detected feature coordinates, or None.
        """
        if gray_frame is None or gray_frame.size == 0:
            return None

        points = cv2.goodFeaturesToTrack(
            gray_frame, mask=mask, **self.feature_params
        )
        return points

    def compute_flow(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute optical flow between two distinct grayscale frames.

        Args:
            prev_frame: Previous grayscale image frame.
            curr_frame: Current grayscale image frame.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - good_old: Array of valid source points in prev_frame.
                - good_new: Array of corresponding destination points in curr_frame.
                - status: Vector indicating successfully tracked points (1) or lost points (0).
        """
        empty_pts = np.empty((0, 2), dtype=np.float32)
        empty_status = np.empty((0, 1), dtype=np.uint8)

        if prev_frame is None or curr_frame is None:
            return empty_pts, empty_pts, empty_status

        # Find features in previous frame
        p0 = self.detect_features(prev_frame)
        if p0 is None or len(p0) == 0:
            return empty_pts, empty_pts, empty_status

        # Calculate optical flow
        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            prev_frame, curr_frame, p0, None, **self.lk_params
        )

        if p1 is None or status is None:
            return empty_pts, empty_pts, empty_status

        # Select only valid tracked points
        good_new = p1[status == 1]
        good_old = p0[status == 1]

        return good_old, good_new, status

    def update(
        self, curr_gray_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Stateful sequential update: computes flow relative to previously passed frame.

        Args:
            curr_gray_frame: Current grayscale frame.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (good_old_points, good_new_points)
        """
        empty_pts = np.empty((0, 2), dtype=np.float32)

        if self.prev_gray is None:
            self.prev_gray = curr_gray_frame.copy()
            self.prev_points = self.detect_features(self.prev_gray)
            return empty_pts, empty_pts

        if self.prev_points is None or len(self.prev_points) == 0:
            self.prev_points = self.detect_features(self.prev_gray)
            if self.prev_points is None or len(self.prev_points) == 0:
                self.prev_gray = curr_gray_frame.copy()
                return empty_pts, empty_pts

        p1, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, curr_gray_frame, self.prev_points, None, **self.lk_params
        )

        if p1 is None or status is None:
            self.prev_gray = curr_gray_frame.copy()
            self.prev_points = None
            return empty_pts, empty_pts

        good_new = p1[status == 1]
        good_old = self.prev_points[status == 1]

        # Cache state for next invocation
        self.prev_gray = curr_gray_frame.copy()
        self.prev_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None

        return good_old, good_new

    def reset(self) -> None:
        """Clear cached frames and tracked feature points."""
        self.prev_gray = None
        self.prev_points = None
