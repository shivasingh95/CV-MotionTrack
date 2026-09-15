"""
Optical Flow and Lucas-Kanade / KLT Motion-Estimation Module for CV-MotionTrack.

Academic Coursework: CSE3010 Computer Vision

Mathematical and Algorithmic Foundations:
=========================================
1. Brightness Constancy Assumption:
   Under the assumption that scene illumination is locally invariant across small temporal
   intervals dt, a pixel at coordinate (x, y) at time t retains its intensity at (x+u, y+v)
   at time t+1:
       I(x, y, t) ≈ I(x + u, y + v, t + 1)

2. Linearization via First-Order 2D Taylor Series Expansion:
       I(x + u, y + v, t + 1) ≈ I(x, y, t) + (∂I/∂x)*u + (∂I/∂y)*v + (∂I/∂t)
   Subtracting I(x, y, t) yields the fundamental Optical Flow Constraint Equation:
       Ix * u + Iy * v + It = 0
   where:
       Ix = ∂I/∂x (horizontal spatial image gradient)
       Iy = ∂I/∂y (vertical spatial image gradient)
       It = ∂I/∂t (temporal image gradient between consecutive frames)
       u = dx/dt, v = dy/dt (apparent horizontal and vertical motion velocities)

3. The Aperture Problem & Local Spatial Constraint:
   A single equation with two unknowns (u, v) is ill-posed (underdetermined). Only motion
   normal to edges can be recovered (the aperture problem). Lucas and Kanade (1981) resolved
   this by assuming that apparent motion is locally coherent across a small spatial window
   Ω of size W x W (e.g. 21x21) centered at pixel p:
       Ix(pi) * u + Iy(pi) * v = -It(pi)   for each pi ∈ Ω

   In matrix form (A * [u, v]^T = b):
       A = [ [Ix(p1), Iy(p1)],
             [Ix(p2), Iy(p2)],
             ...
             [Ix(pn), Iy(pn)] ],   b = -[ It(p1), It(p2), ..., It(pn) ]^T

   Multiplying by A^T gives the normal equations:
       (A^T A) * [u, v]^T = A^T b

   The system has a unique, stable solution if the 2x2 structure tensor (A^T A) is invertible
   and well-conditioned:
       A^T A = [ [ ∑ Ix^2,   ∑ Ix*Iy ],
                 [ ∑ Ix*Iy, ∑ Iy^2   ] ]

4. Feature Selection via Shi-Tomasi Corner Detector (Good Features to Track):
   Flat regions yield A^T A ≈ 0 (rank 0). Edges yield one large eigenvalue and one near zero
   (rank 1, aperture problem). Corners yield two large eigenvalues (λ1, λ2 ≥ λ_min > 0).
   Shi and Tomasi (1994) demonstrated that tracking quality is maximized by selecting feature
   points where R = min(λ1, λ2) > threshold, as implemented in cv2.goodFeaturesToTrack().

5. Pyramidal Lucas-Kanade for Large Displacements (max_level):
   Standard Lucas-Kanade relies on Taylor linearization, which is only valid for small pixel
   displacements (|u|, |v| < 1-2 pixels). For larger object motions, OpenCV builds Gaussian image
   pyramids (subsampling by factor 2 at each level 0..max_level). Optical flow is solved first
   at the coarsest pyramid level where motion is small in downsampled coordinates, then propagated
   and refined coarse-to-fine to the highest resolution (level 0).
"""

from dataclasses import dataclass
import math
from typing import Any, List, Optional, Tuple, Union
import cv2
import numpy as np

from src.config import (
    OPTICAL_FLOW_MAX_CORNERS,
    OPTICAL_FLOW_MAX_LEVEL,
    OPTICAL_FLOW_MIN_DISTANCE,
    OPTICAL_FLOW_MIN_FEATURES,
    OPTICAL_FLOW_QUALITY_LEVEL,
    OPTICAL_FLOW_WIN_SIZE,
    OpticalFlowConfig,
)
from src.data_models import OpticalFlowPoint


class OpticalFlowAnalyzer:
    """
    Sparse Optical Flow analyzer using OpenCV's Pyramidal Lucas-Kanade algorithm
    and Shi-Tomasi corner detection.

    Attributes:
        win_size: Spatial window size (width, height) for Lucas-Kanade tracking.
        max_level: Number of pyramid levels for coarse-to-fine motion estimation.
        max_corners: Maximum number of Shi-Tomasi corners to detect.
        quality_level: Minimum accepted quality fraction for corner eigenvalues.
        min_distance: Minimum Euclidean distance in pixels between detected features.
        min_features: Threshold below which automatic re-detection is triggered.
        prev_gray: Cached previous frame (single-channel 2D uint8 ndarray).
        prev_points: Cached feature points tracked from the previous frame.
        flow_points: List of OpticalFlowPoint instances computed in the last update.
    """

    def __init__(
        self,
        win_size: Union[Tuple[int, int], OpticalFlowConfig] = OPTICAL_FLOW_WIN_SIZE,
        max_level: int = OPTICAL_FLOW_MAX_LEVEL,
        max_corners: int = OPTICAL_FLOW_MAX_CORNERS,
        quality_level: float = OPTICAL_FLOW_QUALITY_LEVEL,
        min_distance: float = OPTICAL_FLOW_MIN_DISTANCE,
        min_features: int = OPTICAL_FLOW_MIN_FEATURES,
        config: Optional[OpticalFlowConfig] = None,
    ) -> None:
        """
        Initialize the OpticalFlowAnalyzer.

        Supports configuration via individual keyword parameters or by passing an
        OpticalFlowConfig dataclass instance (either as `win_size` or `config`).
        """
        # Handle configuration object passed as first argument
        if isinstance(win_size, OpticalFlowConfig):
            config = win_size
            win_size = OPTICAL_FLOW_WIN_SIZE

        if config is not None:
            win_size = config.win_size
            max_level = config.max_level
            max_corners = config.max_corners
            quality_level = config.quality_level
            min_distance = config.min_distance
            min_features = config.min_features

        self.win_size: Tuple[int, int] = win_size
        self.max_level: int = max_level
        self.max_corners: int = max_corners
        self.quality_level: float = quality_level
        self.min_distance: float = min_distance
        self.min_features: int = min_features
        self.block_size: int = 7

        # Lucas-Kanade termination criteria: 10 iterations or eps <= 0.03
        self.criteria: Tuple[int, int, float] = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            10,
            0.03,
        )

        # Internal tracking state
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.flow_points: List[OpticalFlowPoint] = []
        self.valid_prev_points: np.ndarray = np.empty((0, 2), dtype=np.float32)
        self.valid_curr_points: np.ndarray = np.empty((0, 2), dtype=np.float32)

    # ------------------------------------------------------------------ #
    # Validation Helpers                                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_frame(frame: Any, frame_name: str = "Frame") -> None:
        """
        Validate that the provided image frame is a non-empty 2D grayscale NumPy ndarray.

        Raises:
            ValueError: If frame is None, not an ndarray, empty, or not 2D grayscale.
        """
        if frame is None:
            raise ValueError(f"{frame_name} cannot be None.")
        if not isinstance(frame, np.ndarray):
            raise ValueError(f"{frame_name} must be a numpy ndarray, got {type(frame).__name__}.")
        if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
            raise ValueError(f"{frame_name} cannot be empty.")
        if frame.ndim != 2:
            raise ValueError(
                f"{frame_name} must be a single-channel 2D grayscale image (got shape {frame.shape})."
            )

    def _validate_frame_pair(self, previous_gray: Any, current_gray: Any) -> None:
        """
        Validate that two frames are valid grayscale arrays of matching dimensions.

        Raises:
            ValueError: If either frame is invalid or dimensions do not match.
        """
        self._validate_frame(previous_gray, "Previous frame")
        self._validate_frame(current_gray, "Current frame")
        if previous_gray.shape != current_gray.shape:
            raise ValueError(
                f"Incompatible frame dimensions: previous {previous_gray.shape} vs current {current_gray.shape}."
            )

    # ------------------------------------------------------------------ #
    # Feature Detection                                                  #
    # ------------------------------------------------------------------ #

    def detect_features(
        self, gray_frame: np.ndarray, mask: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Identify prominent corner feature points suitable for differential tracking.

        Academic Explanation:
            Uses the Shi-Tomasi corner detection algorithm (cv2.goodFeaturesToTrack).
            Corners represent locations where local image gradients vary significantly
            in both horizontal and vertical directions (both eigenvalues of the structure
            tensor A^T A are large). This overcomes the aperture problem and ensures that
            the Lucas-Kanade normal equations have a stable, invertible solution.
            Flat or uniform image regions are safely handled and return None.

        Args:
            gray_frame: Single-channel 2D grayscale image.
            mask: Optional binary mask specifying where to look for corners.

        Returns:
            Optional[np.ndarray]: Array of corner coordinates of shape (N, 1, 2)
            with dtype float32, or None if no valid features are found.
        """
        self._validate_frame(gray_frame, "gray_frame")

        points = cv2.goodFeaturesToTrack(
            gray_frame,
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
            mask=mask,
        )
        return points

    # ------------------------------------------------------------------ #
    # Optical Flow Computation                                           #
    # ------------------------------------------------------------------ #

    def calculate_flow(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
        previous_points: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate sparse Pyramidal Lucas-Kanade optical flow between two frames.

        Args:
            previous_gray: Grayscale image at time t.
            current_gray: Grayscale image at time t + 1.
            previous_points: Array of source coordinates in previous_gray,
                with shape (N, 1, 2) or (N, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - current_points: Tracked coordinates in current_gray (shape (N, 1, 2), float32).
                - status: Array (N, 1) of uint8 where 1 indicates tracking success, 0 failure.
                - error: Array (N, 1) of float32 tracking error metrics for each point.
        """
        self._validate_frame_pair(previous_gray, current_gray)

        empty_pts = np.empty((0, 1, 2), dtype=np.float32)
        empty_status = np.empty((0, 1), dtype=np.uint8)
        empty_error = np.empty((0, 1), dtype=np.float32)

        if previous_points is None or len(previous_points) == 0:
            return empty_pts, empty_status, empty_error

        p0 = np.ascontiguousarray(previous_points, dtype=np.float32)
        if p0.ndim == 2:
            p0 = p0.reshape(-1, 1, 2)

        current_points, status, error = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            p0,
            None,
            winSize=self.win_size,
            maxLevel=self.max_level,
            criteria=self.criteria,
        )

        if current_points is None or status is None:
            return empty_pts, empty_status, empty_error

        if error is None:
            error = np.zeros_like(status, dtype=np.float32)

        return current_points, status, error

    # ------------------------------------------------------------------ #
    # Point Filtering & Vector Extraction                                #
    # ------------------------------------------------------------------ #

    def filter_valid_points(
        self,
        previous_points: Optional[np.ndarray],
        current_points: Optional[np.ndarray],
        status: Optional[np.ndarray],
        error: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter tracked points based on the Lucas-Kanade status vector.

        Academic Explanation:
            The optical flow solver may fail for points that move outside the frame
            boundaries, undergo severe non-rigid deformation, or become occluded.
            cv2.calcOpticalFlowPyrLK indicates successful convergence with status == 1.
            Points with status == 0 are discarded to maintain clean motion vectors.

        Args:
            previous_points: Source feature coordinates in previous frame.
            current_points: Destination feature coordinates in current frame.
            status: Status array (1 if flow found, 0 otherwise).
            error: Optional error array returned by the optical flow solver.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - valid_previous: Array of shape (M, 2) of successfully tracked source points.
                - valid_current: Array of shape (M, 2) of corresponding destination points.
        """
        empty_pts = np.empty((0, 2), dtype=np.float32)

        if (
            previous_points is None
            or current_points is None
            or status is None
            or len(status) == 0
        ):
            return empty_pts, empty_pts

        p0 = np.asarray(previous_points, dtype=np.float32).reshape(-1, 2)
        p1 = np.asarray(current_points, dtype=np.float32).reshape(-1, 2)
        st = np.asarray(status).ravel()

        if len(p0) != len(st) or len(p1) != len(st):
            return empty_pts, empty_pts

        valid_mask = st == 1
        valid_p0 = p0[valid_mask]
        valid_p1 = p1[valid_mask]

        return valid_p0, valid_p1

    def create_flow_points(
        self,
        previous_points: np.ndarray,
        current_points: np.ndarray,
    ) -> List[OpticalFlowPoint]:
        """
        Construct structured OpticalFlowPoint instances containing coordinates,
        displacement vectors (dx, dy), and magnitude.

        Displacement vector formulation:
            dx = x_current - x_previous
            dy = y_current - y_previous
            magnitude = sqrt(dx^2 + dy^2)

        Args:
            previous_points: Array of shape (M, 2) source points.
            current_points: Array of shape (M, 2) destination points.

        Returns:
            List[OpticalFlowPoint]: Structured list of valid motion points.
        """
        flow_points: List[OpticalFlowPoint] = []
        if previous_points is None or current_points is None:
            return flow_points

        p0 = np.asarray(previous_points, dtype=np.float32).reshape(-1, 2)
        p1 = np.asarray(current_points, dtype=np.float32).reshape(-1, 2)

        for (x0, y0), (x1, y1) in zip(p0, p1):
            dx = float(x1 - x0)
            dy = float(y1 - y0)
            flow_points.append(
                OpticalFlowPoint(
                    previous_x=float(x0),
                    previous_y=float(y0),
                    current_x=float(x1),
                    current_y=float(y1),
                    dx=dx,
                    dy=dy,
                )
            )

        return flow_points

    # ------------------------------------------------------------------ #
    # Lifecycle & Stateful Tracking Pipeline                             #
    # ------------------------------------------------------------------ #

    def initialize(self, gray_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Initialize tracking state with an initial grayscale frame.

        Detects initial prominent feature points and caches the frame.

        Args:
            gray_frame: Initial single-channel 2D grayscale image.

        Returns:
            Optional[np.ndarray]: Initial feature points (shape (N, 1, 2)), or None.
        """
        self._validate_frame(gray_frame, "gray_frame")
        self.prev_gray = gray_frame.copy()
        self.prev_points = self.detect_features(gray_frame)
        self.flow_points = []
        self.valid_prev_points = np.empty((0, 2), dtype=np.float32)
        self.valid_curr_points = np.empty((0, 2), dtype=np.float32)
        return self.prev_points

    def update(
        self,
        previous_gray_or_current: np.ndarray,
        current_gray: Optional[np.ndarray] = None,
    ) -> List[OpticalFlowPoint]:
        """
        Compute optical flow and update feature point tracking state.

        Supports two execution modes:
        1. Two-Frame Mode: `update(previous_gray, current_gray)`
           Computes flow between explicit consecutive frames.
        2. Sequential Stream Mode: `update(current_frame)`
           Uses cached `self.prev_gray` as the previous frame and advances state.

        Tracking Lifecycle & Automatic Reinitialization:
            If the number of successfully tracked points drops below `min_features`
            (e.g., features exit frame or get occluded), new Shi-Tomasi corners are
            automatically detected to maintain persistent tracking coverage.

        Args:
            previous_gray_or_current: Either previous frame (in 2-arg mode) or
                current frame (in 1-arg sequential mode).
            current_gray: Current frame (in 2-arg mode), or None for sequential mode.

        Returns:
            List[OpticalFlowPoint]: Clean list of successfully tracked feature points
            with displacement vectors and magnitudes.
        """
        # Determine execution mode
        if current_gray is None:
            # Sequential mode: previous_gray_or_current is current frame
            curr_frame = previous_gray_or_current
            self._validate_frame(curr_frame, "Current frame")

            if self.prev_gray is None or self.prev_gray.shape != curr_frame.shape:
                self.initialize(curr_frame)
                return []
            prev_frame = self.prev_gray
        else:
            # Two-frame mode
            prev_frame = previous_gray_or_current
            curr_frame = current_gray
            self._validate_frame_pair(prev_frame, curr_frame)

        # Ensure features exist on previous frame
        if self.prev_points is None or len(self.prev_points) < self.min_features:
            self.prev_points = self.detect_features(prev_frame)

        if self.prev_points is None or len(self.prev_points) == 0:
            # Flat frame or no corners found
            self.prev_gray = curr_frame.copy()
            self.prev_points = None
            self.valid_prev_points = np.empty((0, 2), dtype=np.float32)
            self.valid_curr_points = np.empty((0, 2), dtype=np.float32)
            self.flow_points = []
            return []

        # Calculate Pyramidal Lucas-Kanade optical flow
        curr_pts, status, error = self.calculate_flow(
            prev_frame, curr_frame, self.prev_points
        )

        # Filter valid tracked points
        valid_p0, valid_p1 = self.filter_valid_points(
            self.prev_points, curr_pts, status, error
        )
        self.valid_prev_points = valid_p0
        self.valid_curr_points = valid_p1

        # Extract structured OpticalFlowPoint records
        self.flow_points = self.create_flow_points(valid_p0, valid_p1)

        # Re-detect features if remaining valid points fall below threshold
        if len(valid_p1) < self.min_features:
            new_features = self.detect_features(curr_frame)
            self.prev_points = new_features
        else:
            self.prev_points = valid_p1.reshape(-1, 1, 2)

        # Advance cached frame
        self.prev_gray = curr_frame.copy()

        return self.flow_points

    def compute_flow(
        self, prev_frame: np.ndarray, curr_frame: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Legacy compatibility interface for computing optical flow between two frames.

        Args:
            prev_frame: Previous grayscale image.
            curr_frame: Current grayscale image.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - good_old: Valid source points (shape (M, 2)).
                - good_new: Valid destination points (shape (M, 2)).
                - status: Raw status vector.
        """
        p0 = self.detect_features(prev_frame)
        if p0 is None or len(p0) == 0:
            empty = np.empty((0, 2), dtype=np.float32)
            return empty, empty, np.empty((0, 1), dtype=np.uint8)

        p1, status, err = self.calculate_flow(prev_frame, curr_frame, p0)
        v0, v1 = self.filter_valid_points(p0, p1, status, err)
        return v0, v1, status

    def reset(self) -> None:
        """Clear cached frames, feature points, and tracking state."""
        self.prev_gray = None
        self.prev_points = None
        self.flow_points = []
        self.valid_prev_points = np.empty((0, 2), dtype=np.float32)
        self.valid_curr_points = np.empty((0, 2), dtype=np.float32)
