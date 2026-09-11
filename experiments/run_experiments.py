"""
Automated experiment execution and evaluation suite for CV-MotionTrack (Step 9).

Executes the complete pipeline across all 6 test scenarios + 2 comparison configurations:
1. Scenario 1: Single moving object
2. Scenario 2: Multiple moving objects
3. Scenario 3: Slow-moving object
4. Scenario 4: Fast-moving object
5. Scenario 5: Object re-entry
6. Scenario 6: Background variation & shadows
7. Comparison Configuration A: Low compute / fast
8. Comparison Configuration B: High compute / dense

Collects empirical metrics, appends to CSV, exports JSON, and captures screenshots.
"""

import json
import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import numpy as np

from src.config import (
    AppConfig,
    DetectorConfig,
    MotionAnalysisConfig,
    OpticalFlowConfig,
    PreprocessingConfig,
    TrackerConfig,
    VideoConfig,
    VisualizationConfig,
)
from src.detector import ObjectDetector
from src.evaluation import Evaluator
from src.motion_analysis import MotionAnalyzer
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer
from experiments.generate_scenarios import generate_all_scenarios


def evaluate_video(
    video_path: str,
    experiment_name: str,
    config: AppConfig,
    screenshot_save_path: str = None,
    screenshot_frame_idx: int = 25,
) -> dict:
    """
    Execute the CV-MotionTrack pipeline on a given video and record evaluation metrics.

    Args:
        video_path: Path to video file.
        experiment_name: Name of the experiment.
        config: Application configuration.
        screenshot_save_path: Path to save representative screenshot.
        screenshot_frame_idx: Frame index at which to grab screenshot.

    Returns:
        dict: Summary metrics dictionary.
    """
    video_config = config.video
    video_config.source = video_path

    video_processor = VideoProcessor(config=video_config)
    preprocessor = Preprocessor(config=config.preprocessing)
    detector = ObjectDetector(config=config.detector)
    tracker = ObjectTracker(config=config.tracker)
    optical_flow = OpticalFlowAnalyzer(config=config.optical_flow)
    motion_analyzer = MotionAnalyzer(config=config.motion)
    visualizer = Visualizer(config=config.visualization)
    evaluator = Evaluator(experiment_name=experiment_name)

    if not video_processor.open():
        raise RuntimeError(f"Could not open video at: {video_path}")

    stream_fps = video_processor.fps
    motion_analyzer.config.fps = stream_fps

    frame_idx = 0
    saved_screenshot = False

    while True:
        evaluator.start_frame()

        # Step 1: Ingest
        ret, frame = video_processor.read_frame()
        if not ret or frame is None:
            break

        frame_idx += 1

        # Step 2: Preprocess
        gray_frame = preprocessor.process(frame)

        # Step 3, 4, 5: Background subtraction & Detection
        detections, fg_mask = detector.detect(gray_frame)

        # Step 6: Tracking
        tracked_objects = tracker.update(detections)

        # Step 7: Optical Flow
        flow_points = optical_flow.update(gray_frame)

        # Step 8: Motion Analysis
        motion_data = motion_analyzer.update(tracked_objects, flow_points)

        # Step 9: Record metrics (high-resolution timer stopped inside record_frame)
        _ = evaluator.record_frame(
            detections=detections,
            tracked_objects=tracked_objects,
            optical_flow_points=flow_points,
            motion_data=motion_data,
        )

        # Step 10: Render for screenshot capture if requested
        if screenshot_save_path and not saved_screenshot and frame_idx >= screenshot_frame_idx:
            stats = motion_analyzer.get_summary_statistics()
            annotated = visualizer.render(
                frame=frame,
                detections=detections,
                tracked_objects=tracked_objects,
                motion_data=motion_data,
                optical_flow_points=flow_points,
                fps=evaluator.frame_history[-1].fps if evaluator.frame_history else 30.0,
                stats=stats,
            )
            os.makedirs(os.path.dirname(os.path.abspath(screenshot_save_path)), exist_ok=True)
            cv2.imwrite(screenshot_save_path, annotated)
            saved_screenshot = True

    video_processor.release()
    return evaluator


def run_all_benchmarks():
    """Execute complete benchmark suite across all scenarios and save unified results."""
    # Ensure all test videos exist
    scenarios = generate_all_scenarios()

    metrics_dir = os.path.join("results", "metrics")
    screenshots_dir = os.path.join("results", "screenshots")
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)

    csv_path = os.path.join(metrics_dir, "experiment_results.csv")
    json_path = os.path.join(metrics_dir, "experiment_results.json")

    # Clear previous CSV/JSON if present for fresh experiment run
    if os.path.exists(csv_path):
        os.remove(csv_path)

    all_evaluators = []
    unified_results = []

    # 1. Base Configuration Experiments (Scenarios 1 - 6)
    base_config = AppConfig()
    base_config.detector.history = 30
    base_config.detector.var_threshold = 16.0
    base_config.detector.min_contour_area = 100.0

    experiment_plan = [
        ("Experiment 1: Single Object", scenarios["scenario_1_single_object"], "scenario_1_single_object.jpg", 25, base_config),
        ("Experiment 2: Multi Object", scenarios["scenario_2_multi_object"], "scenario_2_multi_object.jpg", 25, base_config),
        ("Experiment 3: Slow Motion", scenarios["scenario_3_slow_motion"], "scenario_3_slow_motion.jpg", 25, base_config),
        ("Experiment 4: Fast Motion", scenarios["scenario_4_fast_motion"], "scenario_4_fast_motion.jpg", 18, base_config),
        ("Experiment 5: Object Reentry", scenarios["scenario_5_reentry"], "scenario_5_reentry.jpg", 45, base_config),
        ("Experiment 6: Background Variation", scenarios["scenario_6_background_variation"], "scenario_6_background_variation.jpg", 25, base_config),
    ]

    print("\n" + "=" * 70)
    print("RUNNING SCENARIO BENCHMARKS (1 - 6)")
    print("=" * 70)

    for exp_name, vid_path, screenshot_name, shot_frame, cfg in experiment_plan:
        print(f"\n[RUNNING] {exp_name} on {vid_path}...")
        shot_path = os.path.join(screenshots_dir, screenshot_name)
        evaluator = evaluate_video(
            video_path=vid_path,
            experiment_name=exp_name,
            config=cfg,
            screenshot_save_path=shot_path,
            screenshot_frame_idx=shot_frame,
        )
        evaluator.print_terminal_summary()
        evaluator.save_csv(csv_path, video_name=os.path.basename(vid_path))
        all_evaluators.append(evaluator)
        unified_results.append({
            "experiment": exp_name,
            "video": os.path.basename(vid_path),
            "summary": evaluator.get_summary().__dict__,
        })

    # 2. Computational Comparison: Config A (Low Compute) vs Config B (High Compute) on Scenario 2
    print("\n" + "=" * 70)
    print("RUNNING COMPUTATIONAL COMPARISON: CONFIG A (Low Compute) vs CONFIG B (High Compute)")
    print("=" * 70)

    # Config A: Low Compute (small kernel, fewer corners, 1 pyramid level)
    config_a = AppConfig()
    config_a.preprocessing.gaussian_kernel_size = (3, 3)
    config_a.optical_flow.max_corners = 50
    config_a.optical_flow.max_level = 1
    config_a.detector.history = 20

    print("\n[RUNNING] Comparison Config A (Low Compute / Fast)...")
    eval_a = evaluate_video(
        video_path=scenarios["scenario_2_multi_object"],
        experiment_name="Comparison: Config A (Low Compute)",
        config=config_a,
        screenshot_save_path=os.path.join(screenshots_dir, "comparison_config_A.jpg"),
        screenshot_frame_idx=25,
    )
    eval_a.print_terminal_summary()
    eval_a.save_csv(csv_path, video_name="scenario_2_multi_object.avi (Config A)")
    all_evaluators.append(eval_a)
    unified_results.append({
        "experiment": "Comparison: Config A (Low Compute)",
        "video": "scenario_2_multi_object.avi",
        "summary": eval_a.get_summary().__dict__,
    })

    # Config B: High Compute (large kernel, dense corners, 4 pyramid levels)
    config_b = AppConfig()
    config_b.preprocessing.gaussian_kernel_size = (7, 7)
    config_b.optical_flow.max_corners = 200
    config_b.optical_flow.max_level = 4
    config_b.detector.history = 80

    print("\n[RUNNING] Comparison Config B (High Compute / Dense)...")
    eval_b = evaluate_video(
        video_path=scenarios["scenario_2_multi_object"],
        experiment_name="Comparison: Config B (High Compute)",
        config=config_b,
        screenshot_save_path=os.path.join(screenshots_dir, "comparison_config_B.jpg"),
        screenshot_frame_idx=25,
    )
    eval_b.print_terminal_summary()
    eval_b.save_csv(csv_path, video_name="scenario_2_multi_object.avi (Config B)")
    all_evaluators.append(eval_b)
    unified_results.append({
        "experiment": "Comparison: Config B (High Compute)",
        "video": "scenario_2_multi_object.avi",
        "summary": eval_b.get_summary().__dict__,
    })

    # Save Unified JSON Report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(unified_results, f, indent=2)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY")
    print(f"Metrics CSV saved to  : {csv_path}")
    print(f"Unified JSON saved to : {json_path}")
    print(f"Screenshots saved to  : {screenshots_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_benchmarks()
