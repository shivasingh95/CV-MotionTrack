"""
Image preprocessing module for CV-MotionTrack.

Implements the fundamental classical Computer Vision preprocessing stages:
input validation, color-space conversion (BGR to Grayscale), and 2D Gaussian
spatial filtering for noise suppression before background modeling and detection.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.config import GAUSSIAN_KERNEL_SIZE, GAUSSIAN_SIGMA, PreprocessingConfig


class Preprocessor:
    """
    Applies image preprocessing operations to prepare raw video frames
    for downstream computer vision tasks (detection, optical flow, tracking).

    Attributes:
        gaussian_kernel_size: (width, height) of Gaussian kernel (positive odd integers).
        gaussian_sigma: Standard deviation for Gaussian kernel. If 0, computed from kernel size.
        config: Associated PreprocessingConfig object.
    """

    def __init__(
        self,
        gaussian_kernel_size: Tuple[int, int] = GAUSSIAN_KERNEL_SIZE,
        gaussian_sigma: float = GAUSSIAN_SIGMA,
        config: Optional[PreprocessingConfig] = None,
    ) -> None:
        """
        Initialize the Preprocessor.

        Args:
            gaussian_kernel_size: Tuple (kw, kh) of positive odd integers.
                                  Defaults to (5, 5).
            gaussian_sigma: Gaussian kernel standard deviation. Defaults to 0.
            config: Optional PreprocessingConfig instance. If provided, values
                    from config override defaults unless explicitly specified.

        Raises:
            ValueError: If gaussian_kernel_size has invalid (non-positive or even) dimensions.
        """
        if config is not None:
            self.config = config
            ksize = gaussian_kernel_size if gaussian_kernel_size != GAUSSIAN_KERNEL_SIZE else config.gaussian_kernel_size
            sigma = gaussian_sigma if gaussian_sigma != GAUSSIAN_SIGMA else config.gaussian_sigma_x
        else:
            ksize = gaussian_kernel_size
            sigma = gaussian_sigma
            self.config = PreprocessingConfig(
                gaussian_kernel_size=ksize,
                gaussian_sigma_x=sigma,
                gaussian_sigma_y=sigma,
            )

        self._validate_kernel_size(ksize)
        self.gaussian_kernel_size: Tuple[int, int] = ksize
        self.gaussian_sigma: float = sigma

    @staticmethod
    def _validate_kernel_size(kernel_size: Tuple[int, int]) -> None:
        """
        Validate that Gaussian kernel dimensions are positive odd integers.

        Args:
            kernel_size: (width, height) tuple to validate.

        Raises:
            ValueError: If kernel_size is not a 2-tuple of positive odd integers.
        """
        if not isinstance(kernel_size, (tuple, list)) or len(kernel_size) != 2:
            raise ValueError(
                f"Gaussian kernel size must be a tuple of 2 elements, got: {kernel_size}"
            )

        kw, kh = kernel_size
        if not isinstance(kw, int) or not isinstance(kh, int):
            raise ValueError(
                f"Gaussian kernel dimensions must be integers, got: ({type(kw)}, {type(kh)})"
            )

        if kw <= 0 or kh <= 0:
            raise ValueError(
                f"Gaussian kernel dimensions must be strictly positive, got: {kernel_size}"
            )

        if kw % 2 == 0 or kh % 2 == 0:
            raise ValueError(
                f"Gaussian kernel dimensions must be odd numbers, got: {kernel_size}"
            )

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert an input BGR image/frame to single-channel 8-bit grayscale.

        Args:
            frame: OpenCV image array (H, W, 3) in BGR format, or (H, W) grayscale.

        Returns:
            np.ndarray: Grayscale image with shape (H, W) and dtype uint8.

        Raises:
            ValueError: If frame is None, empty, not a numpy array, or has an invalid shape.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray):
            raise ValueError(f"Input frame must be a NumPy array, got: {type(frame)}")

        if frame.size == 0:
            raise ValueError("Input frame is empty (size is 0).")

        if frame.ndim == 2:
            # Already single-channel 2D grayscale
            return frame.copy()

        if frame.ndim == 3:
            channels = frame.shape[2]
            if channels == 3:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif channels == 4:
                return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
            elif channels == 1:
                return frame.squeeze(axis=2).copy()

        raise ValueError(
            f"Unsupported frame shape for grayscale conversion: {frame.shape}. "
            f"Expected (H, W, 3) BGR or (H, W) 2D array."
        )

    def gaussian_blur(
        self,
        frame: np.ndarray,
        kernel_size: Optional[Tuple[int, int]] = None,
        sigma: Optional[float] = None,
    ) -> np.ndarray:
        """
        Apply 2D Gaussian spatial smoothing to suppress high-frequency noise.

        Uses the discrete Gaussian filter formula:
            G(x, y) = 1 / (2 * pi * sigma^2) * exp(-(x^2 + y^2) / (2 * sigma^2))

        Args:
            frame: Grayscale or color image array.
            kernel_size: Optional override for (kw, kh). If None, uses self.gaussian_kernel_size.
            sigma: Optional override for Gaussian sigma. If None, uses self.gaussian_sigma.

        Returns:
            np.ndarray: Gaussian-smoothed image preserving frame dimensions.

        Raises:
            ValueError: If frame is invalid or kernel dimensions are not positive odd integers.
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        if not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Input frame must be a non-empty NumPy array.")

        ksize = kernel_size if kernel_size is not None else self.gaussian_kernel_size
        self._validate_kernel_size(ksize)

        sig = sigma if sigma is not None else self.gaussian_sigma

        return cv2.GaussianBlur(frame, ksize, sigmaX=sig, sigmaY=sig)

    def apply_gaussian_blur(
        self,
        frame: np.ndarray,
        kernel_size: Optional[Tuple[int, int]] = None,
        sigma_x: Optional[float] = None,
        sigma_y: Optional[float] = None,
    ) -> np.ndarray:
        """
        Backward-compatible alias for gaussian_blur.
        """
        sig = sigma_x if sigma_x is not None else sigma_y
        return self.gaussian_blur(frame, kernel_size=kernel_size, sigma=sig)

    def process(self, frame: np.ndarray) -> np.ndarray:
        """
        Execute the primary preprocessing pipeline on an incoming video frame.

        Sequential steps:
            1. Input validation
            2. Color-space conversion (BGR to Grayscale)
            3. 2D Gaussian smoothing

        Args:
            frame: Raw input video frame (BGR or Grayscale).

        Returns:
            np.ndarray: Preprocessed single-channel smoothed image frame (H, W).

        Raises:
            ValueError: If input frame is None, empty, or structurally invalid.
        """
        # Step 1 & 2: Validate and convert to grayscale
        gray = self.to_grayscale(frame)

        # Step 3: Apply Gaussian spatial smoothing
        smoothed = self.gaussian_blur(gray)

        return smoothed

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Alias for process() to ensure compatibility across the pipeline.
        """
        return self.process(frame)
