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

## High-Level Features
- **Strictly Modular Architecture**: Clear separation of concerns with type-annotated interfaces and typed dataclasses (`Detection`, `TrackedObject`, `MotionData`).
- **Configurable Pipeline**: Centralized parameter management (`AppConfig`) covering all filtering, detection, tracking, and visualization hyperparameters.
- **Robust Input Handling**: Stream validation supporting both device camera indices and local video file paths.
- **Classical Vision Toolchain**: Fully implemented with OpenCV, NumPy, and standard Python libraries without external opaque dependencies.
- **Comprehensive Test Suite**: Automated unit tests using `pytest` verifying subsystem interfaces, mathematical calculations, and edge cases.
- **Dual Operating Modes**: Interactive HighGUI visualization mode and non-interactive `--headless` mode for automated evaluation.

## Expected Outcome
Upon full implementation of all milestones, CV-MotionTrack will deliver:
1. A stable, educational software system capable of processing video feeds at real-time frame rates (20–30+ FPS on modern CPUs).
2. Clean visual tracking of multiple moving objects with continuous trajectories under moderate scene clutter.
3. Accurate kinematic motion telemetry that correctly reflects physical directions and relative velocity trends.
4. Thorough documentation of all theoretical foundations, architectural decisions, and comparative algorithm analyses.
