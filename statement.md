# Project Statement

## Problem Statement
Detecting, identifying, and characterizing moving entities within video streams is a cornerstone problem in computer vision with widespread applications in traffic flow monitoring, intelligent surveillance, and biomechanical analysis. Modern solutions frequently deploy heavy deep learning detectors that act as opaque black boxes and require dedicated GPU hardware. 

For academic learning and resource-constrained environments, there is a strong need to understand, construct, and evaluate classical computer vision pipelines from first principles. Such pipelines build upon spatial filtering, probabilistic background modeling, morphological transformations, feature tracking, and kinematic estimation to provide transparent, interpretable, and computationally lean motion perception.

## Scope
The scope of the **CV-MotionTrack** system encompasses:
- Real-time video ingestion from integrated/USB webcams and recorded local video files.
- Classical spatial image preprocessing, including color conversions and Gaussian spatial smoothing to reduce sensor noise.
- Moving foreground segmentation using statistical background subtraction algorithms (Gaussian Mixture Models / MOG2 and K-Nearest Neighbors) combined with morphological cleaning filters.
- Multi-object tracking across sequential video frames using centroid association to assign and maintain unique identities and historical trajectory paths.
- Apparent motion vector estimation via differential optical flow (Lucas-Kanade / KLT features).
- Kinematic spatio-temporal parameter extraction, estimating displacement magnitude, directional heading angle, and velocities.
- Diagnostic overlay rendering on video frames showing bounding boxes, IDs, trajectories, vectors, and performance HUD.

### Exclusions & Boundaries
- This milestone focuses exclusively on modular architecture, contracts, and classical methods; heavy deep learning architectures (e.g., YOLO, Mask R-CNN) are explicitly outside the scope of this project.
- Real-time performance is targeted for standard resolution feeds (e.g., 640x480) on conventional CPU hardware.

## Target Users
1. **Academic Instructors and Evaluators (CSE3010)**: Evaluating student comprehension of core computer vision principles, mathematical formulations, and software engineering practices.
2. **Computer Vision Students and Researchers**: Seeking an interpretable, modular reference implementation of classical video analysis algorithms.
3. **Embedded and Low-Resource Developers**: Requiring deterministic, lightweight motion tracking pipelines that function without GPU acceleration.

## High-Level Features & Functional Requirements

The system satisfies ten comprehensive functional requirements (FR1–FR10):

- **FR1: Video Stream Acquisition & Validation** (`src/video_processor.py`)
  - Ingests video from local camera indices (`0`, `1`) and recorded video files (`.mp4`, `.avi`).
  - Validates stream health, verifies resolution, reports frame counts, and handles stream EOF or camera disconnections gracefully.
- **FR2: Spatial Image Preprocessing** (`src/preprocessing.py`)
  - Converts 3-channel BGR input to scalar luminance (grayscale) via standard ITU-R BT.601 psychophysical weighting.
  - Applies 2D isotropic Gaussian blur ($k \times k$, $\sigma$) to attenuate high-frequency sensor noise.
- **FR3: Background Modeling & Foreground Extraction** (`src/detector.py`)
  - Employs adaptive Gaussian Mixture Models (MOG2) to learn multi-modal background intensity distributions online.
  - Evaluates deviation per pixel and segments foreground masks with shadow discrimination.
- **FR4: Morphological Filtering & Contour Region Extraction** (`src/detector.py`)
  - Applies binary thresholding to suppress shadow pixels.
  - Applies morphological Opening (erosion then dilation) to remove salt-and-pepper noise and Closing (dilation then erosion) to seal internal object holes.
  - Traces external contours (`cv2.findContours`), filters blobs by pixel area thresholds (`MIN_OBJECT_AREA`, `MAX_OBJECT_AREA`), and computes bounding boxes and moment-based centroids.
- **FR5: Multi-Object Tracking & Identity Management** (`src/tracker.py`)
  - Maintains unique object identities using greedy Euclidean centroid distance matching.
  - Supports configurable maximum matching distance, tracks consecutive disappeared frames, and manages birth, active, and deregistration states.
  - Records continuous historical spatio-temporal centroid trajectory paths.
- **FR6: Sparse Optical Flow & Point Tracking** (`src/optical_flow.py`)
  - Detects salient corner features using Shi-Tomasi minimum eigenvalue criterion ($R = \min(\lambda_1, \lambda_2)$).
  - Estimates inter-frame motion vectors using pyramidal Lucas-Kanade differential optical flow under brightness constancy.
  - Automatically replenishes feature points when valid track counts drop below configurable limits.
- **FR7: Spatio-Temporal Motion Parameter Extraction** (`src/motion_analysis.py`)
  - Computes frame-to-frame displacement, total cumulative path length, and net displacement.
  - Classifies heading direction into 8 continuous $45^\circ$ sectors (Right, Down-Right, Down, Down-Left, Left, Up-Left, Up, Up-Right) or Stationary.
  - Calculates instantaneous and rolling moving-average image-space velocities in pixels/frame and pixels/second.
- **FR8: Real-Time Diagnostic Visual Overlay** (`src/visualizer.py`)
  - Renders annotated bounding boxes, object IDs, centroid marker points, trajectory trails, optical flow vectors, and telemetry HUD cards.
- **FR9: Quantitative Evaluation & Profiling** (`src/evaluation.py`)
  - Measures execution latency, frame rates (FPS), detection rates, track lifetimes, and kinematic distributions.
  - Exports standardized empirical metrics into CSV and JSON files for comparative analysis.
- **FR10: Desktop Graphical User Interface & Interactive Controls** (`src/ui.py`)
  - Provides a Tkinter desktop GUI running the video pipeline on a background worker thread.
  - Offers live aspect-ratio video display, playback controls (Start, Pause, Resume, Reset, Stop), live hyperparameter adjustment sliders, real-time telemetry HUD, and timestamped screenshot capture.

---

## Non-Functional Requirements (NFRs)

- **NFR1: Real-Time Performance**: Processes standard resolution video feeds (640x480) at real-time speeds (>= 25–30 FPS) on modern multi-core CPU architectures without hardware acceleration.
- **NFR2: Modularity & Separation of Concerns**: Each subsystem is encapsulated in its own module with strict type-annotated interfaces and communicates through typed Python dataclasses (`Detection`, `TrackedObject`, `OpticalFlowPoint`, `MotionData`).
- **NFR3: Explainability & Classical Toolchain**: Strictly avoids opaque deep-learning models or external black boxes; all algorithms are derived from classical computer vision theory taught in CSE3010.
- **NFR4: Robustness & Testability**: Backed by a comprehensive automated test suite (`pytest`) covering unit, integration, edge cases, and UI interactions with 100% passing status.
- **NFR5: Platform Portability**: Written entirely in standard Python 3 with OpenCV (`opencv-python`), NumPy, Pillow, and Tkinter, running seamlessly across Windows, Linux, and macOS.

---

## Technologies Used & Academic Justification

| Technology | Purpose | Academic Justification |
|---|---|---|
| **Python 3.10+** | Core Programming Language | High readability, rich scientific computing ecosystem, rapid prototyping of algorithmic pipelines. |
| **OpenCV (`cv2`)** | Computer Vision Operations | Industry-standard optimized library for image transformations, spatial filtering, background modeling, optical flow, and contour analysis. |
| **NumPy** | Numerical Array Manipulation | High-performance vectorized matrix math for coordinate calculations, distance metrics, and moments. |
| **Pillow (`PIL`)** | GUI Image Interfacing | Clean, robust bridging between OpenCV BGR numpy arrays and Tkinter canvas image objects. |
| **Tkinter / ttk** | Graphical User Interface | Standard cross-platform GUI library included with Python; ensures zero heavyweight external GUI dependencies. |
| **Pytest** | Automated Testing Framework | Test discovery, parameterized testing, fixtures, and assertions ensuring algorithmic reliability and preventing regression. |

---

## Expected Outcome

Upon full implementation of all milestones, CV-MotionTrack delivers:
1. A stable, educational software system capable of processing video feeds at real-time frame rates (30–90+ FPS on CPU).
2. Clean visual tracking of multiple moving objects with continuous trajectories under moderate scene clutter.
3. Accurate kinematic motion telemetry that correctly reflects physical directions and relative velocity trends.
4. Thorough documentation of theoretical foundations, architectural decisions, and comparative algorithm analyses.

