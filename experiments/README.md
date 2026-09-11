# CV-MotionTrack Experimentation & Evaluation Suite

This directory contains the reproducible experimental scenarios, benchmark execution scripts, and configuration profiles for **CV-MotionTrack** (CSE3010 Computer Vision Academic Project).

---

## Evaluation Objective

The evaluation framework quantitatively and empirically measures:
1. **Pipeline Latency & Throughput**: Average frame processing latency (ms/frame) and real-time processing rate (FPS) using high-resolution timers (`time.perf_counter()`).
2. **Detection Behavior**: Object counts per frame, minimum/maximum detections, and zero-detection frame frequency under various motion dynamics.
3. **Tracking Continuity**: Active tracks, ID persistence, track lifetimes, and deregistration handling across object entry/exit.
4. **Optical Flow Stability**: Number of tracked Lucas-Kanade feature points and reinitialization frequency under visual and motion variation.
5. **Image-Space Kinematics**: Measured displacements and image-space velocities across slow, moderate, and fast motion scenarios.

> **Academic Limitation**: All reported metrics represent empirical system-level measurements on test video sequences. No ground-truth precision/recall accuracy is claimed without manually annotated ground-truth datasets. Velocity is strictly reported in **pixels/frame** and time-scaled **pixels/second**.

---

## Test Scenarios

| Scenario ID | Name | Objective & Visual Dynamics |
|---|---|---|
| **Scenario 1** | Single Moving Object | Evaluates baseline detection, single-track ID assignment, and trajectory accumulation for a single rectilinear moving object. |
| **Scenario 2** | Multiple Moving Objects | Evaluates simultaneous tracking of multiple objects with distinct trajectories and persistent distinct IDs. |
| **Scenario 3** | Slow-Moving Object | Evaluates the threshold sensitivity of background subtraction and motion analysis under sub-pixel and near-stationary displacements. |
| **Scenario 4** | Fast-Moving Object | Evaluates tracking stability and Lucas-Kanade optical flow convergence under large inter-frame spatial displacements. |
| **Scenario 5** | Object Re-entry | Evaluates tracker disappearance counter, object deregistration when exiting the frame, and new track registration upon re-entering. |
| **Scenario 6** | Background Variation & Shadows | Observes classical MOG2 background subtractor behavior under global illumination fluctuations and dynamic shadow casting. |

---

## Comparison Experiment (Config A vs Config B)

To measure the trade-off between computational throughput (FPS) and feature density:
- **Configuration A (Low Compute / Fast)**: Resolution $320 \times 240$, max 50 Shi-Tomasi corners, LK pyramid level 1, Gaussian kernel $(3, 3)$.
- **Configuration B (High Compute / Dense)**: Resolution $640 \times 480$, max 200 Shi-Tomasi corners, LK pyramid level 4, Gaussian kernel $(7, 7)$.

---

## How to Run the Experiments

### 1. Generate Synthetic Video Scenarios
```powershell
python experiments/generate_scenarios.py
```
This generates 6 deterministic test video files in `data/input/`:
- `data/input/scenario_1_single_object.avi`
- `data/input/scenario_2_multi_object.avi`
- `data/input/scenario_3_slow_motion.avi`
- `data/input/scenario_4_fast_motion.avi`
- `data/input/scenario_5_reentry.avi`
- `data/input/scenario_6_background_variation.avi`

### 2. Run All Experiments & Benchmark Suite
```powershell
python experiments/run_experiments.py
```
This runs all scenarios and comparison configurations, prints console summary tables, and exports:
- CSV Summary: `results/metrics/experiment_results.csv`
- Structured JSON Reports: `results/metrics/experiment_results.json`
- Annotated Screenshots: `results/screenshots/scenario_*.jpg`
