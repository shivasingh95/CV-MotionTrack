"""
Object detection module for CV-MotionTrack.

Implements classical motion detection using background modeling (MOG2 / KNN),
thresholding, morphological filtering (opening and closing), and contour analysis.
"""

from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.config import DetectorConfig
from src.data_models import BoundingBox, Detection


class ObjectDetector:
    """
    Detects moving foreground objects using background subtraction and contour extraction.

    Attributes:
        config: DetectorConfig with threshold and filtering parameters.
        subtractor: Underlying OpenCV background subtractor (MOG2 or KNN).
        morph_kernel: Structuring element for morphological filtering.
    """

    def __init__(self, config: Optional[DetectorConfig] = None) -> None:
        """
        Initialize the ObjectDetector.

        Args:
            config: Detector configuration. If None, default settings are used.
        """
        self.config = config or DetectorConfig()
        self.subtractor = self._create_subtractor()
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.config.morph_kernel_size
        )

    def _create_subtractor(self) -> cv2.BackgroundSubtractor:
        """Instantiate the background subtractor according to configuration."""
        sub_type = self.config.subtractor_type.upper()
        if sub_type == "MOG2":
            return cv2.createBackgroundSubtractorMOG2(
                history=self.config.history,
                varThreshold=self.config.var_threshold,
                detectShadows=self.config.detect_shadows,
            )
        elif sub_type == "KNN":
            return cv2.createBackgroundSubtractorKNN(
                history=self.config.history,
                dist2Threshold=self.config.var_threshold,
                detectShadows=self.config.detect_shadows,
            )
        else:
            # Fallback to standard MOG2
            return cv2.createBackgroundSubtractorMOG2(
                history=self.config.history,
                varThreshold=self.config.var_threshold,
                detectShadows=self.config.detect_shadows,
            )

    def apply_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute the raw foreground segmentation mask via background modeling.

        Args:
            frame: Current preprocessed video frame.

        Returns:
            np.ndarray: Raw foreground mask (values typically 0, 127 for shadows, 255 for foreground).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Frame provided to background subtractor is empty.")
        return self.subtractor.apply(frame)

    def remove_noise(self, mask: np.ndarray) -> np.ndarray:
        """
        Refine the foreground mask using thresholding and morphological operations.

        Applies:
            1. Binary thresholding to discard shadow pixels (if shadows enabled).
            2. Morphological Opening (erosion followed by dilation) to eliminate isolated false-positive noise.
            3. Morphological Closing (dilation followed by erosion) to bridge intra-object gaps.

        Args:
            mask: Raw foreground mask from background subtraction.

        Returns:
            np.ndarray: Cleaned binary mask (0 or 255).
        """
        if mask is None or mask.size == 0:
            raise ValueError("Mask provided for noise removal is empty.")

        # If shadows are detected, OpenCV marks them as ~127. Threshold to keep only true foreground (255)
        if self.config.detect_shadows:
            _, binary_mask = cv2.threshold(
                mask, self.config.shadow_threshold, 255, cv2.THRESH_BINARY
            )
        else:
            binary_mask = mask.copy()

        # Morphological Opening to remove small spatial noise
        if self.config.morph_open_iterations > 0:
            binary_mask = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_OPEN,
                self.morph_kernel,
                iterations=self.config.morph_open_iterations,
            )

        # Morphological Closing to seal holes inside detected blobs
        if self.config.morph_close_iterations > 0:
            binary_mask = cv2.morphologyEx(
                binary_mask,
                cv2.MORPH_CLOSE,
                self.morph_kernel,
                iterations=self.config.morph_close_iterations,
            )

        return binary_mask

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Execute full detection sequence on a video frame:
        background subtraction -> morphological cleanup -> contour extraction -> bounding boxes.

        Args:
            frame: Current preprocessed image frame.

        Returns:
            Tuple[List[Detection], np.ndarray]:
                - List of Detection instances meeting area criteria.
                - Refined binary foreground mask.
        """
        raw_mask = self.apply_background_subtraction(frame)
        clean_mask = self.remove_noise(raw_mask)

        # Find external contours representing moving blobs
        contours, _ = cv2.findContours(
            clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections: List[Detection] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if self.config.min_contour_area <= area <= self.config.max_contour_area:
                x, y, w, h = cv2.boundingRect(contour)
                moments = cv2.moments(contour)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                else:
                    cx = x + w // 2
                    cy = y + h // 2

                bbox = BoundingBox(x=x, y=y, width=w, height=h)
                detection = Detection(
                    bbox=bbox,
                    centroid=(cx, cy),
                    confidence=1.0,
                    area=float(area),
                )
                detections.append(detection)

        return detections, clean_mask

    def reset(self) -> None:
        """Reset the background model state."""
        self.subtractor = self._create_subtractor()
