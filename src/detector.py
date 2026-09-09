"""
Object detection module for CV-MotionTrack.

Implements classical moving-object detection via:
  1. Statistical background modeling (Gaussian Mixture Model / MOG2 or KNN).
  2. Foreground mask extraction by background subtraction.
  3. Shadow removal via thresholding.
  4. Morphological cleaning (Opening + Closing) to suppress noise.
  5. Contour analysis to extract candidate bounding boxes and centroids.

This module is intentionally decoupled from the tracking module.
The output List[Detection] is the contract that the tracker consumes.

CSE3010 concepts demonstrated
──────────────────────────────
• Background modeling    – adaptive per-pixel Gaussian mixture
• Background subtraction – per-pixel temporal differencing against model
• Foreground extraction  – binary mask of moving-object pixels
• Thresholding           – shadow removal, binary binarisation
• Morphological opening  – erode then dilate → remove isolated noise
• Morphological closing  – dilate then erode → fill intra-object gaps
• Contour analysis       – find blob boundaries, compute bounding rects
                           and image-moment centroids
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import (
    BACKGROUND_HISTORY,
    BACKGROUND_VAR_THRESHOLD,
    DETECT_SHADOWS,
    MAX_OBJECT_AREA,
    MIN_OBJECT_AREA,
    MORPH_CLOSE_ITERATIONS,
    MORPH_KERNEL_SIZE,
    MORPH_OPEN_ITERATIONS,
    SHADOW_PIXEL_VALUE,
    DetectorConfig,
)
from src.data_models import BoundingBox, Detection


class ObjectDetector:
    """
    Detects moving foreground objects in video frames using classical
    background subtraction and morphological filtering.

    The MOG2 background subtractor maintains a Gaussian mixture model (GMM)
    for every pixel.  Over *history* frames it learns the statistical
    distribution of pixel intensities that constitute the background.  When a
    new frame arrives, pixels whose intensity falls outside the learned
    distribution by more than *var_threshold* standard deviations are flagged
    as foreground (moving object).  Pixels that partially differ are labelled
    as shadows (pixel value 127) when *detect_shadows* is True.

    Attributes
    ----------
    history        : int   – number of frames used to build the background model.
    var_threshold  : float – Mahalanobis distance threshold for foreground/bg decision.
    detect_shadows : bool  – whether MOG2 should detect and label shadow regions.
    min_area       : float – minimum contour area (pixels²) to count as an object.
    max_area       : float – maximum contour area; huge blobs are rejected as noise.
    morph_kernel   : np.ndarray – structuring element used for morphological ops.
    _subtractor    : cv2.BackgroundSubtractor – the underlying OpenCV GMM model.
    """

    def __init__(
        self,
        history: int = BACKGROUND_HISTORY,
        var_threshold: float = BACKGROUND_VAR_THRESHOLD,
        detect_shadows: bool = DETECT_SHADOWS,
        min_area: float = MIN_OBJECT_AREA,
        max_area: float = MAX_OBJECT_AREA,
        morph_kernel_size: Tuple[int, int] = MORPH_KERNEL_SIZE,
        morph_open_iterations: int = MORPH_OPEN_ITERATIONS,
        morph_close_iterations: int = MORPH_CLOSE_ITERATIONS,
        shadow_threshold: int = SHADOW_PIXEL_VALUE,
        config: Optional[DetectorConfig] = None,
    ) -> None:
        """
        Initialise the ObjectDetector.

        A *config* object, if provided, overrides all keyword arguments.
        This allows the detector to be wired into the central AppConfig
        while still being easily instantiated standalone in tests.

        Parameters
        ----------
        history              : frames used to train the background model.
        var_threshold        : foreground decision threshold (pixel variance).
        detect_shadows       : enable MOG2 shadow detection (pixel value 127).
        min_area             : minimum contour area to accept as a detection.
        max_area             : maximum contour area to accept as a detection.
        morph_kernel_size    : (w, h) of the rectangular structuring element.
        morph_open_iterations : erosion+dilation passes for noise removal.
        morph_close_iterations: dilation+erosion passes to fill blob holes.
        shadow_threshold     : pixel value used by MOG2 to mark shadows.
        config               : optional DetectorConfig overriding all above.
        """
        if config is not None:
            history               = config.history
            var_threshold         = config.var_threshold
            detect_shadows        = config.detect_shadows
            min_area              = config.min_contour_area
            max_area              = config.max_contour_area
            morph_kernel_size     = config.morph_kernel_size
            morph_open_iterations = config.morph_open_iterations
            morph_close_iterations = config.morph_close_iterations
            shadow_threshold      = config.shadow_threshold

        self.history: int = history
        self.var_threshold: float = var_threshold
        self.detect_shadows: bool = detect_shadows
        self.min_area: float = min_area
        self.max_area: float = max_area
        self.morph_open_iterations: int = morph_open_iterations
        self.morph_close_iterations: int = morph_close_iterations
        self.shadow_threshold: int = shadow_threshold

        # Build rectangular structuring element (kernel) once; reused each frame.
        self.morph_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_RECT, morph_kernel_size
        )

        # Create the MOG2 background subtractor.
        # MOG2 (Zivkovic 2004) models each pixel with a mixture of K Gaussians
        # and adapts the mixture weights online.  *history* controls how many
        # past frames influence the model; smaller values make it adapt faster
        # but become noisy in slow-moving scenes.
        self._subtractor: cv2.BackgroundSubtractor = (
            cv2.createBackgroundSubtractorMOG2(
                history=self.history,
                varThreshold=self.var_threshold,
                detectShadows=self.detect_shadows,
            )
        )

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def apply_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Feed *frame* to the background model and return the raw foreground mask.

        The mask contains three possible pixel values when shadow detection is on:
            0   – background (no change)
            127 – shadow pixel (partial illumination change, no new object)
            255 – foreground (genuine moving object)

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed grayscale (H, W) or colour (H, W, C) image.

        Returns
        -------
        np.ndarray
            Raw single-channel foreground mask (dtype uint8).

        Raises
        ------
        ValueError
            If *frame* is None or an empty array.
        """
        self._validate_frame(frame)
        return self._subtractor.apply(frame)

    def clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Convert the raw MOG2 mask into a clean binary foreground mask.

        Steps
        -----
        1. **Shadow removal** – threshold to discard shadow pixels (127) and
           keep only genuine foreground (255).  Binary output: {0, 255}.
        2. **Morphological Opening** (erosion → dilation) – removes isolated
           noise speckles whose area is smaller than the structuring element.
        3. **Morphological Closing** (dilation → erosion) – bridges small gaps
           *inside* foreground blobs, producing solid filled regions.

        The kernel size and iteration counts come from the central config so
        they are never hardcoded in this method.

        Parameters
        ----------
        mask : np.ndarray
            Raw single-channel mask from apply_background_subtraction().

        Returns
        -------
        np.ndarray
            Clean binary mask (0 or 255), same spatial dimensions as input.

        Raises
        ------
        ValueError
            If *mask* is None or empty.
        """
        if mask is None:
            raise ValueError("Mask passed to clean_mask() is None.")
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            raise ValueError("Mask passed to clean_mask() is empty or not a NumPy array.")

        # Step 1 – Shadow removal.
        # MOG2 marks shadow pixels as 127.  Any value above shadow_threshold
        # (default 127) is treated as a genuine foreground pixel (255).
        _, binary = cv2.threshold(
            mask,
            self.shadow_threshold,   # pixels <= threshold become 0
            255,                     # pixels >  threshold become 255
            cv2.THRESH_BINARY,
        )

        # Step 2 – Morphological Opening: erode then dilate.
        # Erosion shrinks foreground regions; small isolated noise blobs
        # (narrower than the kernel) disappear completely.  The subsequent
        # dilation restores the size of the surviving regions.
        if self.morph_open_iterations > 0:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_OPEN,
                self.morph_kernel,
                iterations=self.morph_open_iterations,
            )

        # Step 3 – Morphological Closing: dilate then erode.
        # Dilation bridges small dark holes or gaps inside a foreground blob.
        # The subsequent erosion returns the outer boundary to its original
        # position, leaving the interior holes filled.
        if self.morph_close_iterations > 0:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=self.morph_close_iterations,
            )

        return binary

    def get_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Convenience method: apply background subtraction then clean the mask.

        This is the single call a caller needs to go from a preprocessed frame
        to a clean binary foreground mask ready for contour extraction.

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed (grayscale or colour) image frame.

        Returns
        -------
        np.ndarray
            Clean binary foreground mask (values 0 or 255).
        """
        raw_mask = self.apply_background_subtraction(frame)
        return self.clean_mask(raw_mask)

    def detect_objects(self, mask: np.ndarray) -> List[Detection]:
        """
        Extract candidate object regions from a clean binary foreground mask.

        Algorithm
        ---------
        1. Find external contours (cv2.RETR_EXTERNAL) using border-following.
        2. Filter by contour area: discard < min_area and > max_area.
        3. Compute bounding rectangle (cv2.boundingRect) for each kept contour.
        4. Compute sub-pixel centroid from zeroth and first image moments:
               c_x = M10 / M00,   c_y = M01 / M00
           Falls back to bounding-rect centre when M00 == 0.
        5. Return as a list of Detection dataclass instances.

        No tracking IDs are assigned here.  That is the tracker's
        responsibility in the next pipeline stage.

        Parameters
        ----------
        mask : np.ndarray
            Clean binary foreground mask (output of get_foreground_mask or
            clean_mask).  Values should be 0 (background) or 255 (foreground).

        Returns
        -------
        List[Detection]
            Potentially empty list.  Each entry contains a BoundingBox, centroid
            (cx, cy), contour area, and confidence=1.0.

        Raises
        ------
        ValueError
            If *mask* is None or empty.
        """
        if mask is None:
            raise ValueError("Mask passed to detect_objects() is None.")
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            raise ValueError("Mask passed to detect_objects() is empty or not a NumPy array.")

        # Ensure the mask is single-channel uint8 for findContours.
        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)

            # Filter out noise (too small) and trivially large regions.
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(contour)

            # Moment-based centroid – more accurate than bbox centre for
            # non-rectangular blobs.
            moments = cv2.moments(contour)
            if moments["m00"] != 0.0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
            else:
                # Degenerate contour: fall back to bounding-rect centre.
                cx = x + w // 2
                cy = y + h // 2

            detection = Detection(
                bbox=BoundingBox(x=x, y=y, width=w, height=h),
                centroid=(cx, cy),
                confidence=1.0,
                area=float(area),
            )
            detections.append(detection)

        return detections

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Full pipeline: frame → foreground mask → clean mask → detections.

        This is the primary method used by main.py to drive the detection stage.

        Parameters
        ----------
        frame : np.ndarray
            Preprocessed image frame (grayscale preferred).

        Returns
        -------
        Tuple[List[Detection], np.ndarray]
            (list of Detection objects, clean binary foreground mask)
        """
        clean_mask = self.get_foreground_mask(frame)
        detections = self.detect_objects(clean_mask)
        return detections, clean_mask

    def reset(self) -> None:
        """
        Discard the learned background model and start fresh.

        Useful when switching video sources mid-session.
        """
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self.history,
            varThreshold=self.var_threshold,
            detectShadows=self.detect_shadows,
        )

    # ------------------------------------------------------------------ #
    # Private helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _validate_frame(frame: np.ndarray) -> None:
        """Raise ValueError for None or empty frames."""
        if frame is None:
            raise ValueError("Frame passed to ObjectDetector is None.")
        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError(
                "Frame passed to ObjectDetector is empty or not a NumPy array."
            )
