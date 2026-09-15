# CV-MotionTrack

> **Academic Course Project**: CSE3010 Computer Vision  
> **Evaluation System Status**: Submission Ready & Verified

---

## 1. Project Overview

**CV-MotionTrack** is a modular, classical Computer Vision software system designed to analyze live video feeds from local webcams or recorded video streams. It receives raw RGB/BGR video frames, applies spatial intensity transformations and noise filtering, performs foreground modeling via statistical background subtraction, tracks moving-object identities and trajectories across successive frames, estimates apparent motion fields using sparse optical flow, and derives quantitative image-space kinematic metrics (displacement, heading direction, and velocity).

Rather than relying on opaque deep-learning models or external black-box frameworks (such as YOLO or DeepSORT), CV-MotionTrack is constructed strictly from foundational computer vision principles as taught in the **CSE3010** curriculum. Every intermediate spatial mask, feature point, and kinematic observation remains mathematically explainable, transparent, and computationally efficient.

---

## 2. Problem Statement

Automated visual surveillance, traffic flow monitoring, and biomechanical analysis require robust, real-time motion perception without the substantial hardware overhead and lack of interpretability associated with end-to-end deep neural networks. In constrained computational environments or academic evaluation scenarios where transparency into intermediate spatial and temporal representations is paramount, classical Computer Vision techniques—such as statistical background subtraction, morphological noise suppression, centroid tracking, and differential optical flow—provide a deterministic, explainable, and lightweight solution.

The primary challenge is designing an integrated, loosely coupled pipeline that gracefully handles spatial sensor noise, dynamic background variations, identity association across frames, and accurate spatio-temporal velocity estimation in real time.

---

## 3. Objectives

1. **Modular Architecture**: Design an extensible, decoupled computer vision architecture separating video acquisition, spatial preprocessing, foreground segmentation, tracking, optical flow, kinematic analysis, evaluation, and presentation.
2. **Classical Preprocessing**: Apply luminance normalization and 2D Gaussian spatial filtering to attenuate camera sensor noise while preserving edge boundaries.
3. **Foreground Segmentation**: Implement statistical background modeling (Gaussian Mixture Models / MOG2) coupled with morphological filtering to segment moving foreground regions.
4. **Multi-Object Tracking**: Associate detected foreground regions across frames using greedy Euclidean distance matching, maintaining unique object identities and trajectory histories.
5. **Differential Motion Estimation**: Integrate Lucas-Kanade sparse optical flow to estimate local apparent velocity vectors.
6. **Kinematic Parameter Extraction**: Derive physical motion metrics including Euclidean displacement, heading angle in degrees, 8-sector directional classification, and rolling moving-average velocity in pixels per second.
7. **Dual-Mode Execution**: Support both a zero-dependency Command-Line Interface (CLI) for automated grading and server execution, and an optional desktop Graphical User Interface (GUI).

---

## 4. Features

- **Command-Line Interface (CLI)**: Full support for `--source webcam`, `--source video --input <path>`, `--min-area`, `--max-distance`, `--output`, and `--save-results`.
- **Flexible Video Acquisition**: Support for both physical USB/integrated webcams (device index `0`) and local video files (`.mp4`, `.avi`, `.mov`).
- **Adaptive Background Subtraction**: Online Gaussian Mixture Model (MOG2) that learns multi-modal background distributions over time.
- **Morphological Artifact Suppression**: Erosion and dilation filters (Opening/Closing) to eradicate false-positive camera noise and fill internal blob voids.
- **Multi-Object Centroid Tracking**: Greedy nearest-neighbor centroid association with unique ID persistence, disappearance grace tolerance, and trajectory recording.
- **Sparse Optical Flow Analysis**: Pyramidal Lucas-Kanade differential optical flow tracking Shi-Tomasi corner features across consecutive video frames with dynamic point replenishment.
- **Image-Space Kinematics**: Frame-to-frame displacement, 8-sector qualitative directional classification, heading angle, path length, and rolling moving-average velocity.
- **Optional Desktop GUI**: Modern dark-themed Tkinter app (`python main.py --gui`) with background worker thread, live aspect-ratio video display, live hyperparameter tuning sliders, live telemetry table, and screenshot capture.
- **Automated Test Suite**: 89 automated pytest tests verifying individual subsystem components, edge cases, integration workflows, and CLI flags.

---

## 5. Computer Vision Techniques

CV-MotionTrack directly implements foundational concepts from the CSE3010 Computer Vision syllabus:

- **Image Preprocessing**: Converts 3-channel BGR input to scalar luminance (grayscale) via standard ITU-R BT.601 psychophysical weighting ($Y = 0.299R + 0.587G + 0.114B$).
- **Gaussian Filtering**: Convolves grayscale frames with an isotropic 2D Gaussian kernel ($G(x, y; \sigma) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$) to suppress high-frequency thermal sensor noise.
- **Background Subtraction**: Models temporal pixel intensity history as a mixture of $K$ adaptive Gaussians (MOG2) to segment foreground pixels.
- **Morphological Processing**: Applies binary thresholding to eliminate shadow pixels, followed by morphological Opening $((A \ominus B) \oplus B)$ and Closing $((A \oplus B) \ominus B)$ using rectangular structuring elements.
- **Contour-Based Detection**: Traces external contours (`cv2.findContours`), filters blobs by pixel area thresholds (`MIN_OBJECT_AREA`), and computes spatial image moment centroids ($c_x = \frac{M_{10}}{M_{00}}, c_y = \frac{M_{01}}{M_{00}}$).
- **Centroid-Based Tracking**: Associates foreground detections to existing tracks across frames via Euclidean distance minimization.
- **Sparse Optical Flow (KLT)**: Detects salient corner features using the Shi-Tomasi minimum eigenvalue criterion ($R = \min(\lambda_1, \lambda_2) > \tau$) and tracks them across frames using pyramidal Lucas-Kanade solving the brightness constancy equation ($I_x u + I_y v + I_t = 0$).
- **Spatio-Temporal Motion Analysis**: Derives frame-to-frame displacement ($\Delta d$), heading angle ($\theta = \text{atan2}(dy, dx)$), 8-sector qualitative direction, and rolling moving-average velocity ($v_{\text{sec}} = v \cdot \text{FPS}$).

---

## 6. CSE3010 Concept Mapping

| CSE3010 Syllabus Concept | Project Implementation | Source Module | Primary Function / Class |
|---|---|---|---|
| **Image Preprocessing** | Grayscale conversion & Gaussian smoothing | `src/preprocessing.py` | `Preprocessor.process()` |
| **Background Subtraction** | Adaptive Gaussian Mixture Model (MOG2) | `src/detector.py` | `ObjectDetector.apply_background_subtraction()` |
| **Object Detection** | Binary thresholding, morphology & contours | `src/detector.py` | `ObjectDetector.detect_objects()` |
| **Object Tracking** | Centroid-based Euclidean distance association | `src/tracker.py` | `ObjectTracker.update()` |
| **Optical Flow** | Pyramidal Lucas-Kanade motion vectors | `src/optical_flow.py` | `OpticalFlowAnalyzer.update()` |
| **KLT Point Tracking** | Shi-Tomasi feature tracking & replenishment | `src/optical_flow.py` | `OpticalFlowAnalyzer._detect_features()` |
| **Spatio-Temporal Analysis** | Rolling trajectory history queues | `src/motion_analysis.py` | `MotionAnalyzer.update()` |
| **Motion Parameter Estimation** | Displacement, 8-sector direction, smoothed velocity | `src/motion_analysis.py` | `MotionAnalyzer._compute_velocity()` |

---

## 7. System Architecture

```
                 ┌── CLI Mode (src/cli.py) ──┐
                 │                           │
Input Stream ──► VideoProcessor ──► Core CV Pipeline ──► Results / Video Output / Metrics
                 │                           │
                 └── GUI Mode (src/ui.py) ───┘
```

The core Computer Vision algorithms (`src/preprocessing.py`, `src/detector.py`, `src/tracker.py`, `src/optical_flow.py`, `src/motion_analysis.py`) remain completely shared and decoupled from presentation controllers.

---

## 8. Workflow

```
Video Frame (BGR)
       │
       ▼
Preprocessing (Grayscale + 2D Gaussian Blur)
       │
       ▼
Background Subtraction (MOG2 + Shadow Thresholding)
       │
       ▼
Morphological Filtering (Opening + Closing)
       │
       ▼
Contour Detection & Spatial Moment Centroids
       │
       ▼
Multi-Object Centroid Association & ID Tracking
       │
       ▼
Sparse Optical Flow (Shi-Tomasi Corners + Pyramidal Lucas-Kanade)
       │
       ▼
Spatio-Temporal Motion Kinematics (Displacement, Direction, Velocity)
       │
       ▼
Diagnostic HUD Rendering & Evaluation Logging
```

---

## 9. Project Structure

```
Computer_vision_project/
│
├── main.py                         # Application entry point (CLI and GUI dispatcher)
├── requirements.txt                # Python package dependencies
├── README.md                       # Comprehensive submission-ready documentation
├── statement.md                    # Problem statement, scope, FR1-FR10, and NFRs
├── conftest.py                     # Root pytest configuration
│
├── src/                            # Core Computer Vision source package
│   ├── __init__.py                 # Package declaration and public exports
│   ├── cli.py                      # Command-Line Interface module
│   ├── config.py                   # Centralized AppConfig dataclasses
│   ├── data_models.py              # Typed data models (Detection, TrackedObject, etc.)
│   ├── preprocessing.py            # Grayscale conversion and Gaussian blur
│   ├── detector.py                 # MOG2 background subtraction and morphology
│   ├── tracker.py                  # Multi-object centroid association tracker
│   ├── optical_flow.py             # Shi-Tomasi corners and Lucas-Kanade flow
│   ├── motion_analysis.py          # Kinematic parameter extraction and smoothing
│   ├── visualizer.py               # Telemetry HUD and visual overlay renderer
│   ├── video_processor.py          # Video capture stream validation and handling
│   ├── evaluation.py               # Automated performance profiling engine
│   └── ui.py                       # Tkinter desktop graphical user interface
│
├── docs/                           # Architecture, workflow, algorithms, evaluation & UML docs
│   ├── architecture.md
│   ├── workflow.md
│   ├── algorithms.md
│   ├── evaluation.md
│   └── uml.md
│
├── experiments/                    # Reproducible experiment testbench
│   ├── generate_scenarios.py
│   └── run_experiments.py
│
├── data/                           # Video asset directories
│   ├── input/                      # Input video clips (sample clips and synthetic scenarios)
│   └── output/                     # Exported processed videos
│
├── results/                        # Generated experimental artifacts
│   ├── metrics/                    # CSV/JSON benchmark results
│   ├── reports/                    # Summary evaluation reports
│   └── screenshots/                # Exported annotated screenshots
│
└── tests/                          # Automated pytest suite (89 tests)
    ├── test_cli.py
    ├── test_preprocessing.py
    ├── test_detector.py
    ├── test_tracker.py
    ├── test_optical_flow.py
    ├── test_motion_analysis.py
    ├── test_integration.py
    ├── test_evaluation.py
    └── test_ui.py
```

---

## 10. Requirements

- **Python**: Version 3.10 or higher
- **OpenCV** (`opencv-python` $\ge 4.8.0$)
- **NumPy** ($\ge 1.24.0$)
- **Pillow** ($\ge 10.0.0$)
- **Pytest** ($\ge 7.4.0$)
- **Matplotlib** ($\ge 3.7.0$)

---

## 11. Installation

### Step 1: Clone Repository
```bash
git clone https://github.com/shivasingh95/CV-MotionTrack.git
cd CV-MotionTrack
```

### Step 2: Create Virtual Environment
**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 12. Command-Line Execution

The application is fully executable from the command line without requiring a GUI display:

### View Help & Options
```bash
python main.py --help
```

### Execute on Webcam Input
```bash
python main.py --source webcam
```

### Execute on Video File Input
```bash
python main.py --source video --input data/input/synthetic_demo.avi
```

### Execute with Custom Hyperparameters & Output Recording
```bash
python main.py --source video --input data/input/synthetic_demo.avi --min-area 100 --max-distance 60 --output data/output/annotated_result.mp4 --save-results
```

### Execute in Headless Mode (Server / Automated Evaluation)
```bash
python main.py --source video --input data/input/synthetic_demo.avi --headless --save-results
```

---

## 13. GUI Execution

To launch the optional desktop Graphical User Interface (Tkinter GUI):
```bash
python main.py --gui
```
*(or run `python src/ui.py` directly)*

---

## 14. Testing

Run the automated test suite to verify system integrity across 89 unit and integration tests:
```bash
pytest -v
```

*Tested Components*:
- `test_cli.py`: Argument parsing, input path validation, and CLI runner.
- `test_preprocessing.py`: Grayscale conversion, Gaussian blur kernel sizing, error handling.
- `test_detector.py`: MOG2 mask generation, shadow thresholding, morphology, contour area filtering.
- `test_tracker.py`: Centroid registration, association matching, disappearance handling.
- `test_optical_flow.py`: Shi-Tomasi corner detection, Lucas-Kanade flow, feature replenishment.
- `test_motion_analysis.py`: Inter-frame displacement, 8-sector direction classification, velocity smoothing.
- `test_integration.py`: End-to-end pipeline execution on synthetic streams.
- `test_evaluation.py`: Frame profiling, latency calculation, CSV/JSON metrics export.
- `test_ui.py`: Tkinter app initialization, settings propagation, treeview updating, frame saving.

---

## 15. Evaluation

The system includes an automated evaluation profiling engine (`src/evaluation.py`) that collects per-frame performance metrics:
- Sub-millisecond latency per frame using `time.perf_counter()`.
- Pipeline throughput in frames per second (FPS).
- Detections per frame and zero-detection count.
- Active track longevity and track deregistrations.
- Lucas-Kanade optical flow point counts and feature re-initialization events.
- Kinematic metrics: mean displacement (pixels) and velocity ($\text{px/s}$).

---

## 16. Results

All results reported below are **actual empirical measurements** collected across standardized benchmark test scenarios and exported to `results/metrics/experiment_results.csv`:

| Scenario | Input Video File | Frames | Avg FPS | Latency (ms/frame) | Registered Tracks | Avg Lifetime (frames) | Avg Displacement (px/frame) | Avg Velocity (px/s) |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Scenario 1** | `scenario_1_single_object.avi` | 45 | **219.3** | 4.56 ms | 1 | 37.0 | 8.09 px | 161.85 px/s |
| **Scenario 2** | `scenario_2_multi_object.avi` | 50 | **120.2** | 8.32 ms | 2 | 41.0 | 6.40 px | 128.11 px/s |
| **Scenario 3** | `scenario_3_slow_motion.avi` | 50 | **105.7** | 9.46 ms | 1 | 34.0 | 15.30 px | 306.05 px/s |
| **Scenario 4** | `scenario_4_fast_motion.avi` | 35 | **119.6** | 8.36 ms | 1 | 15.0 | 19.03 px | 380.59 px/s |
| **Scenario 5** | `scenario_5_reentry.avi` | 65 | **129.2** | 7.74 ms | 2 | 22.5 | 12.27 px | 245.35 px/s |
| **Scenario 6** | `scenario_6_background_variation.avi` | 50 | **128.9** | 7.76 ms | 1 | 42.0 | 6.89 px | 137.74 px/s |

---

## 17. Screenshots

Annotated screenshots captured during real-time video processing are stored in the `results/screenshots/` directory:
- `demo_annotated_frame.jpg`: Composite video overlay featuring bounding boxes, ID badges, trajectory trails, and telemetry HUD card.
- `comparison_config_A.jpg`: Performance comparison under low-compute configuration settings.
- `comparison_config_B.jpg`: Performance comparison under high-compute configuration settings.

---

## 18. Limitations

1. **Uncalibrated Monocular Depth Ambiguity**: Without 3D camera intrinsic/extrinsic calibration or ground-plane homography, physical-world velocity ($\text{m/s}$, $\text{km/h}$) cannot be determined. Motion is strictly reported in image-space pixels and pixels/second.
2. **Static Camera Assumption**: Statistical background subtraction (MOG2) assumes a static camera setup. Sudden camera ego-motion induces global foreground noise.
3. **Occlusion Overlap**: When two moving targets cross directly along the line of sight, single-contour extraction momentarily merges them until spatial separation resumes.

---

## 19. Future Enhancements

- **Kalman Filter Integration**: Incorporating linear Kalman filters for state vector prediction $(x, y, \dot{x}, \dot{y})$ during target occlusions.
- **Hungarian Bipartite Assignment**: Upgrading greedy Euclidean centroid matching to the Kuhn-Munkres (Hungarian) algorithm.
- **Planar Homography Calibration**: Allowing user-specified 4-point ground-plane homography matrix calibration to translate pixel displacement into metric distance.

---

## 20. Technologies & References

### Technologies Used
- **Python 3.10+**: Core programming language.
- **OpenCV (`cv2`)**: Computer Vision operations (grayscale, Gaussian blur, MOG2, contours, Lucas-Kanade optical flow).
- **NumPy**: Matrix computations, Euclidean distance calculation, image moments.
- **Pillow (`PIL`)**: Image format conversion for Tkinter display.
- **Tkinter**: Desktop GUI widget layout.
- **Pytest**: Automated testing framework.

### References & Theoretical Foundations
1. Zivkovic, Z. (2004). "Improved adaptive Gaussian mixture model for background subtraction." *Proceedings of the 17th International Conference on Pattern Recognition (ICPR)*.
2. Lucas, B. D., & Kanade, T. (1981). "An iterative image registration technique with an application to stereo vision." *Proceedings of Imaging Understanding Workshop*.
3. Shi, J., & Tomasi, C. (1994). "Good features to track." *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*.
