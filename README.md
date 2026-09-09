# CV-MotionTrack: Real-Time Object Detection, Tracking and Motion Analysis System

> **Academic Course Project**: CSE3010 Computer Vision  
> **Status**: Step 4 — Background Modeling, Subtraction & Contour Detection Implemented & Verified

---

## Overview

**CV-MotionTrack** is a modular, classical Computer Vision software system designed to analyze live webcam streams or recorded video sequences. It preprocesses incoming video frames, performs foreground modeling and background subtraction to detect moving objects, tracks identities and trajectories across successive frames, computes apparent motion fields using optical flow, and derives kinematic motion metrics (displacement, direction, and velocity).

Rather than relying on opaque deep-learning models or external black-box APIs, CV-MotionTrack emphasizes rigorous algorithmic concepts established in classical Computer Vision as taught in the **CSE3010** curriculum.

---

## Problem Statement

Automated visual surveillance, traffic flow monitoring, and kinematic analysis require robust, real-time motion perception without the substantial hardware overhead and lack of interpretability associated with end-to-end deep neural networks. In constrained computational environments or academic scenarios where transparency into intermediate spatial and temporal representations is paramount, classical Computer Vision techniques—such as statistical background subtraction, morphological noise suppression, centroid tracking, and differential optical flow—provide a deterministic, explainable, and computationally efficient solution.

The primary challenge is designing an integrated, loosely coupled pipeline that gracefully handles high-frequency spatial sensor noise, dynamic background variations, identity association across frames, and accurate spatio-temporal velocity estimation in real time.

---

## Objectives

1. **Modular System Architecture**: Design an extensible, decoupled computer vision architecture separating video acquisition, spatial preprocessing, foreground segmentation, tracking, optical flow, and visualization.
2. **Classical Preprocessing & Noise Suppression**: Apply intensity normalization and 2D Gaussian spatial filtering to suppress camera sensor noise while preserving edge boundaries.
3. **Foreground Segmentation**: Implement statistical background modeling (Gaussian Mixture Models / MOG2 and KNN) coupled with morphological filtering to segment moving regions.
4. **Multi-Object Association & Tracking**: Associate detected foreground regions across frames using centroid Euclidean distance matching, maintaining unique object identities and trajectory histories.
5. **Differential Motion Estimation**: Integrate Lucas-Kanade sparse optical flow to estimate local apparent velocity vectors.
6. **Kinematic Parameter Estimation**: Derive physical motion metrics including Euclidean displacement, heading angle in degrees, and instantaneous/average velocity in pixels per second.
7. **Diagnostic Visual Telemetry**: Render real-time visual HUD overlays showing bounding boxes, centroid paths, motion vectors, and performance statistics.

---

## Features

- **Decoupled Modular Pipeline**: Pure separation of concerns where modules communicate via typed Python dataclasses (`Detection`, `TrackedObject`, `MotionData`).
- **Flexible Video Acquisition**: Support for both live USB/integrated webcam inputs and recorded video files (`.mp4`, `.avi`).
- **Centralized Parameter Management**: Unified configuration through `AppConfig` to eliminate hardcoded hyperparameters.
- **Adaptive Background Modeling**: Online Gaussian Mixture Model (MOG2) that learns the scene background over time and flags deviating pixels as foreground.
- **Moving-Object Region Identification**: Thresholding, morphological cleaning (Opening + Closing), and contour extraction isolate candidate moving-object bounding boxes from every frame.
- **Morphological Artifact Removal**: Erosion and dilation filters (Opening/Closing) eradicate false-positive camera noise and fill interior blob voids.
- **Persistent Trajectory Trails**: Historical tracking of object paths with configurable memory limits (interface scaffolded; implementation planned).
- **Motion Kinematics & HUD**: Calculation of directional heading, displacement, and velocity displayed on a live telemetry dashboard (planned).
- **Headless Execution Mode**: Command-line flag allowing automated testing and server-side execution without requiring an X11/GUI display.

---

## Computer Vision Concepts Used

This project directly implements foundational concepts from the CSE3010 Computer Vision syllabus:

| Syllabus Concept | Module | Implementation Status | Technical Description |
|---|---|---|---|
| **Grayscale Conversion** | `src/preprocessing.py` | **✅ Implemented & Tested** | Luminance weighting $Y = 0.299R + 0.587G + 0.114B$ via `cv2.cvtColor`. |
| **Gaussian Spatial Filtering** | `src/preprocessing.py` | **✅ Implemented & Tested** | 2D isotropic Gaussian convolution $G(x, y; \sigma)$ for high-frequency sensor noise attenuation. |
| **Background Modeling** | `src/detector.py` | **✅ Implemented & Tested** | Adaptive Gaussian Mixture Model (MOG2) — online per-pixel statistical background learning. |
| **Background Subtraction & Foreground Extraction** | `src/detector.py` | **✅ Implemented & Tested** | Per-pixel comparison against GMM; ternary mask (bg=0, shadow=127, fg=255) + binary threshold. |
| **Morphological Processing** | `src/detector.py` | **✅ Implemented & Tested** | Opening $(A \ominus B) \oplus B$ removes noise; Closing $(A \oplus B) \ominus B$ fills holes. |
| **Contour-Based Object Detection** | `src/detector.py` | **✅ Implemented & Tested** | `cv2.findContours` + area filter + bounding rect + moment centroid → `List[Detection]`. |
| **Object Tracking & Association** | `src/tracker.py` | Planned (Interface scaffolded) | Pairwise Euclidean centroid matching and state-machine trajectory maintenance. |
| **Optical Flow (Lucas-Kanade)** | `src/optical_flow.py` | Planned (Interface scaffolded) | Differential intensity spatial-temporal gradients for motion field estimation. |
| **KLT Feature Tracking** | `src/optical_flow.py` | Planned (Interface scaffolded) | Shi-Tomasi corner eigenvalue extraction (`goodFeaturesToTrack`). |
| **Spatio-Temporal Kinematics** | `src/motion_analysis.py` | Planned (Interface scaffolded) | Quantitative displacement $\Delta d$, angular direction $\theta$, and velocity $v = \Delta d / \Delta t$. |

---

## System Architecture

The pipeline processes video sequentially frame by frame:

```
+-------------------------------------------------------------------------------+
|                                VIDEO SOURCE                                   |
|                    (Webcam Device 0 or Video File Stream)                     |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                       VIDEO PROCESSOR (src/video_processor.py)                |
|                    - Stream validation & resolution control                   |
+-------------------------------------------------------------------------------+
                                       | Raw BGR Frame
                                       v
+-------------------------------------------------------------------------------+
|                        PREPROCESSOR (src/preprocessing.py)                    |
|                    - Color conversion: BGR -> Grayscale                       |
|                    - 2D Gaussian smoothing: G(x, y; sigma)                    |
+-------------------------------------------------------------------------------+
                                       | Preprocessed Grayscale Frame
                                       v
+-------------------------------------------------------------------------------+
|                       OBJECT DETECTOR (src/detector.py)                       |
|                    - Background modeling (MOG2 / KNN)                         |
|                    - Morphological Opening & Closing                          |
|                    - Contour extraction -> BoundingBox & Detection dataclass  |
+-------------------------------------------------------------------------------+
                                       | List[Detection]
                                       v
+-------------------------------------------------------------------------------+
|                        OBJECT TRACKER (src/tracker.py)                        |
|                    - Centroid Euclidean distance matching                     |
|                    - ID assignment, state maintenance, trajectories           |
+-------------------------------------------------------------------------------+
                                       | List[TrackedObject]
                                       +-----------------------+
                                       |                       |
                                       v                       v
+---------------------------------------------+ +-------------------------------+
|     OPTICAL FLOW (src/optical_flow.py)      | | MOTION ANALYSIS               |
|  - Shi-Tomasi corner detection              | | (src/motion_analysis.py)      |
|  - Lucas-Kanade differential flow vectors   | | - Displacement magnitude      |
+---------------------------------------------+ | - Heading direction (degrees) |
                                       |        | - Velocity (px/s)             |
                                       |        +-------------------------------+
                                       |                       | Dict[int, MotionData]
                                       +-----------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                         VISUALIZER (src/visualizer.py)                        |
|                    - Bounding boxes & unique ID tags                          |
|                    - Historical trajectory trails                             |
|                    - Directional motion vector arrows                         |
|                    - Semi-transparent telemetry dashboard HUD                |
+-------------------------------------------------------------------------------+
                                       |
                                       v
+-------------------------------------------------------------------------------+
|                                OUTPUT DISPLAY                                 |
|               (OpenCV HighGUI Window / Saved Output Video File)               |
+-------------------------------------------------------------------------------+
```

---

## Project Structure

```
CV-MotionTrack/
│
├── main.py                     # Main CLI pipeline entrypoint and orchestrator
├── requirements.txt            # Project dependencies
├── .gitignore                  # Git ignore rules for Python, cache, and media
├── README.md                   # Project overview and academic documentation
├── statement.md                # Formal project statement and scope
│
├── src/                        # Core application source modules
│   ├── __init__.py             # Package exports and semantic versioning
│   ├── config.py               # Centralized configuration dataclasses
│   ├── data_models.py          # Inter-module communication dataclasses
│   ├── video_processor.py      # Video acquisition and stream management
│   ├── preprocessing.py        # Grayscale conversion and Gaussian filtering
│   ├── detector.py             # Background subtraction and contour detection
│   ├── tracker.py              # Centroid multi-object tracker and trajectories
│   ├── optical_flow.py         # Lucas-Kanade optical flow analyzer
│   ├── motion_analysis.py      # Kinematic parameter estimation (v, d, theta)
│   └── visualizer.py           # Rendering overlays, bounding boxes, and HUD
│
├── tests/                      # Automated unit test suite (pytest)
│   ├── __init__.py
│   ├── test_preprocessing.py   # Validation and filtering tests
│   ├── test_detector.py        # Background subtraction and detection tests
│   ├── test_tracker.py         # ID persistence and tracking logic tests
│   └── test_motion_analysis.py # Kinematic formulas and statistics tests
│
├── data/                       # Media directories
│   ├── input/                  # Test video clips (.gitkeep)
│   └── output/                 # Exported annotated videos (.gitkeep)
│
├── results/                    # Experimental outputs
│   ├── screenshots/            # Annotated frame captures (.gitkeep)
│   └── reports/                # Benchmark summaries (.gitkeep)
│
└── docs/                       # Architectural and algorithmic documentation
    ├── architecture.md         # Detailed subsystem architecture
    ├── workflow.md             # Execution workflow and state machines
    └── algorithms.md           # Theoretical review of Computer Vision algorithms
```

---

## Technologies

- **Language**: Python 3.10+
- **Computer Vision**: OpenCV (`opencv-python >= 4.8.0`)
- **Numerical Computing**: NumPy (`numpy >= 1.24.0`)
- **Plotting & Analysis**: Matplotlib (`matplotlib >= 3.7.0`)
- **Automated Testing**: pytest (`pytest >= 7.4.0`)
- **Version Control**: Git

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd CV-MotionTrack
   ```

2. **Create and activate a virtual environment** (recommended):
   ```bash
   # Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## How to Run

### Live Webcam Stream (Default)
```bash
python main.py --source 0
```

### Process a Video File
```bash
python main.py --source data/input/sample.mp4
```

### Save Annotated Output Video
```bash
python main.py --source data/input/sample.mp4 --output data/output/annotated.avi
```

### Headless Mode (No GUI Window, ideal for testing/servers)
```bash
python main.py --source 0 --headless --max-frames 100
```

---

## Testing

Run the automated test suite using `pytest`:

```bash
# Run all unit tests with verbose output
python -m pytest tests/ -v

# Run tests for a specific subsystem
python -m pytest tests/test_preprocessing.py -v
```

All test cases validate module initialization, contract compliance, error handling, and mathematical accuracy using synthetic NumPy arrays.

---

## Results

*Note: As this project is currently in the initial scaffolding and interface design phase (Milestone 1), quantitative experimental benchmarks (e.g., MOTA, precision-recall curves, FPS across varying resolutions) will be populated upon completion of algorithmic evaluation.*

Planned evaluation metrics:
- **Throughput**: Mean frame processing latency and frames-per-second (FPS).
- **Segmentation Quality**: Qualitative evaluation under varying illumination.
- **Tracking Stability**: Object ID switch count and trajectory continuity under partial occlusion.

---

## Future Enhancements

- Implementation of Kalman Filtering for state prediction and handling brief occlusions.
- Hungarian (Munkres) algorithm for optimal bipartite matching in dense object fields.
- Multi-scale dense optical flow (Farneback method) comparison against sparse Lucas-Kanade.
- Automated export of spatio-temporal trajectories to CSV/JSON for kinematic data mining.

---

## Authors

- **Student Name**: Shiva Raghuwanshi  
- **Course**: CSE3010 Computer Vision  
- **Institution**: Academic Project Submission  
