"""
Main pipeline coordinator for CV-MotionTrack.

Academic Computer Vision project for CSE3010.
Executes the sequential modular pipeline:
  Video Input -> Frame Extraction -> Preprocessing -> Background Subtraction ->
  Object Detection -> Object Tracking -> Optical Flow -> Motion Analysis ->
  Visualization -> Interactive Display / Recording / Evaluation Profiling

Coordinates modular components without embedding algorithm logic directly.
"""

import argparse
from datetime import datetime
import os
import sys
import time
from typing import Optional
import cv2

from src.config import AppConfig
from src.detector import ObjectDetector
from src.evaluation import Evaluator
from src.motion_analysis import MotionAnalyzer
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="CV-MotionTrack: Real-Time Object Detection, Tracking and Motion Analysis (CSE3010)"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam device index (e.g., '0') or path to video file (e.g., 'data/input/sample.mp4')",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch desktop graphical user interface (Tkinter GUI)",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run via command line / OpenCV interactive display instead of Tkinter GUI",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without GUI display window (recommended for headless servers or automated testing)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before exiting (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save annotated video output (.avi or .mp4)",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video to default path if --output is not explicitly given",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Enable evaluation profiling and print evaluation summary",
    )
    parser.add_argument(
        "--eval-csv",
        type=str,
        default=None,
        help="Optional destination path for evaluation CSV results",
    )
    parser.add_argument(
        "--eval-json",
        type=str,
        default=None,
        help="Optional destination path for evaluation JSON results",
    )
    return parser.parse_args()


def run_pipeline(
    source: str = "0",
    headless: bool = False,
    max_frames: Optional[int] = None,
    output_path: Optional[str] = None,
    save_video: bool = False,
    enable_eval: bool = True,
    eval_csv: Optional[str] = None,
    eval_json: Optional[str] = None,
) -> Optional[Evaluator]:
    """
    Coordinate and execute the CV-MotionTrack processing pipeline.

    Args:
        source: Camera index string or path to video file.
        headless: If True, suppress cv2.imshow GUI display.
        max_frames: Optional frame limit.
        output_path: Optional target video path for saving results.
        save_video: Flag to trigger video recording.
        enable_eval: Whether to collect and print evaluation metrics.
        eval_csv: Optional CSV path to export metrics.
        eval_json: Optional JSON path to export metrics.

    Returns:
        Optional[Evaluator]: Populated Evaluator instance if evaluation was enabled.
    """
    # Parse source as integer device index if numeric
    video_source = int(source) if source.isdigit() else source

    # Initialize configuration
    config = AppConfig()
    config.video.source = video_source

    if output_path is not None:
        config.output.save_video = True
        config.output.output_path = output_path
    elif save_video:
        config.output.save_video = True
        if config.output.output_path is None:
            os.makedirs("results", exist_ok=True)
            config.output.output_path = os.path.join("results", "annotated_output.avi")

    print("=" * 65)
    print("CV-MotionTrack: Computer Vision Pipeline Initializing")
    print(f"Source                  : {video_source}")
    print(f"Background Subtractor   : {config.detector.subtractor_type}")
    print(f"Headless Mode           : {headless}")
    print(f"Recording Video         : {config.output.save_video} ({config.output.output_path})")
    print(f"Evaluation Mode         : {enable_eval}")
    print("Keyboard Controls       : 'q'=Quit | 'p'=Pause/Resume | 'r'=Reset | 's'=Save Frame | 'Esc'=Exit")
    print("=" * 65)

    # Initialize pipeline subsystems
    video_processor = VideoProcessor(config=config.video)
    preprocessor = Preprocessor(config=config.preprocessing)
    detector = ObjectDetector(config=config.detector)
    tracker = ObjectTracker(config=config.tracker)
    optical_flow = OpticalFlowAnalyzer(config=config.optical_flow)
    motion_analyzer = MotionAnalyzer(config=config.motion)
    visualizer = Visualizer(config=config.visualization)
    evaluator = Evaluator(experiment_name=f"pipeline_run_{video_source}") if enable_eval else None

    # Open video stream
    try:
        if not video_processor.open():
            print(f"[ERROR] Failed to open video source: {video_source}", file=sys.stderr)
            return None
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return None

    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    fps_estimate = 0.0
    paused = False

    # Configure MotionAnalyzer with reported stream FPS if available
    stream_props = video_processor.get_properties()
    stream_fps = stream_props.get("fps", 30.0)
    motion_analyzer.config.fps = stream_fps

    # Ensure screenshot directory exists
    screenshot_dir = config.output.screenshot_dir
    os.makedirs(screenshot_dir, exist_ok=True)

    try:
        while True:
            # Handle paused state
            if paused:
                if not headless:
                    key = cv2.waitKey(30) & 0xFF
                    if key in (ord("p"), ord("P")):
                        paused = False
                        print("[INFO] Pipeline resumed.")
                    elif key in (ord("q"), ord("Q"), 27):
                        print("[INFO] User requested termination while paused.")
                        break
                    elif key in (ord("r"), ord("R")):
                        tracker.reset()
                        motion_analyzer.reset()
                        optical_flow.reset()
                        if evaluator:
                            evaluator.reset()
                        print("[INFO] State reset by user.")
                continue

            if evaluator:
                evaluator.start_frame()

            # STEP 1: Video Input & Frame Acquisition
            ret, frame = video_processor.read_frame()
            if not ret or frame is None:
                print("\n[INFO] End of video stream reached or no frames received.")
                break

            frame_idx += 1

            # STEP 2: Preprocessing (Grayscale + Gaussian Spatial Smoothing)
            preprocessed_frame = preprocessor.process(frame)

            # STEP 3 & 4 & 5: Background Subtraction, Mask Cleaning & Object Detection
            detections, fg_mask = detector.detect(preprocessed_frame)

            # STEP 6: Object Tracking (Centroid Association & Persistent IDs)
            tracked_objects = tracker.update(detections)

            # STEP 7: Optical Flow (Lucas-Kanade Sparse Feature Motion Vectors)
            optical_flow_points = optical_flow.update(preprocessed_frame)

            # STEP 8: Motion Analysis (Displacement, Direction, Velocity, Trajectory Metrics)
            motion_data = motion_analyzer.update(
                tracked_objects=tracked_objects,
                optical_flow_points=optical_flow_points,
            )

            # Record frame telemetry
            if evaluator:
                frame_m = evaluator.record_frame(
                    detections=detections,
                    tracked_objects=tracked_objects,
                    optical_flow_points=optical_flow_points,
                    motion_data=motion_data,
                )
                fps_estimate = frame_m.fps

            # STEP 9: Visualization (Rendering Composite Annotations)
            stats = motion_analyzer.get_summary_statistics()
            annotated_frame = visualizer.render(
                frame=frame,
                detections=detections,
                tracked_objects=tracked_objects,
                motion_data=motion_data,
                optical_flow_points=optical_flow_points,
                active_count=len(tracked_objects),
                fps=fps_estimate,
                stats=stats,
            )

            # STEP 10: Video Recording (Optional)
            if config.output.save_video and config.output.output_path is not None:
                if writer is None:
                    h, w = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*config.output.codec)
                    writer_fps = stream_fps if stream_fps > 0 else config.output.fps
                    writer = cv2.VideoWriter(
                        config.output.output_path,
                        fourcc,
                        writer_fps,
                        (w, h),
                    )
                    if not writer.isOpened():
                        print(
                            f"[WARNING] Failed to initialize VideoWriter at: {config.output.output_path}",
                            file=sys.stderr,
                        )
                        writer = None

                if writer is not None:
                    writer.write(annotated_frame)

            # STEP 11: Interactive Display & Keyboard Controls
            if not headless:
                display_frame = annotated_frame.copy()
                cv2.imshow("CV-MotionTrack - Main View", display_frame)
                cv2.imshow("CV-MotionTrack - Foreground Mask", fg_mask)

                key = cv2.waitKey(1) & 0xFF

                # 'q' or ESC: Quit application
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] User requested termination.")
                    break

                # 'p': Pause / Resume
                elif key in (ord("p"), ord("P")):
                    paused = True
                    print("\n[INFO] Pipeline paused. Press 'p' to resume.")
                    # Overlay paused banner
                    h, w = display_frame.shape[:2]
                    cv2.putText(
                        display_frame,
                        "[PAUSED - Press 'p' to Resume]",
                        (w // 2 - 170, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("CV-MotionTrack - Main View", display_frame)

                # 'r': Reset tracking and motion analysis state
                elif key in (ord("r"), ord("R")):
                    tracker.reset()
                    motion_analyzer.reset()
                    optical_flow.reset()
                    if evaluator:
                        evaluator.reset()
                    print("\n[INFO] State reset by user: Tracker, MotionAnalyzer, OpticalFlow, and Evaluator cleared.")

                # 's': Save current annotated screenshot
                elif key in (ord("s"), ord("S")):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"screenshot_frame_{frame_idx:05d}_{timestamp}.jpg"
                    save_path = os.path.join(screenshot_dir, filename)
                    cv2.imwrite(save_path, annotated_frame)
                    print(f"\n[INFO] Screenshot saved to: {save_path}")

            # Check optional max frame termination
            if max_frames is not None and frame_idx >= max_frames:
                print(f"\n[INFO] Reached requested max frame limit: {max_frames}")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user (KeyboardInterrupt).")

    finally:
        # Safe resource teardown
        video_processor.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

        if evaluator and len(evaluator.frame_history) > 0:
            evaluator.print_terminal_summary()
            if eval_csv:
                evaluator.save_csv(eval_csv, video_name=str(video_source))
                print(f"[INFO] Evaluation metrics CSV saved to: {eval_csv}")
            if eval_json:
                evaluator.save_json(eval_json, video_name=str(video_source))
                print(f"[INFO] Evaluation metrics JSON saved to: {eval_json}")

    return evaluator


def launch_gui() -> None:
    """Launch the professional Tkinter desktop UI application."""
    import tkinter as tk
    from src.ui import CVMotionTrackApp

    root = tk.Tk()
    app = CVMotionTrackApp(root)
    root.mainloop()


def main() -> None:
    """
    Main application entry point.

    Launches the professional desktop GUI by default if no arguments are passed,
    or if --ui is explicitly specified.
    Otherwise, executes the command-line / OpenCV processing pipeline.
    """
    if len(sys.argv) == 1:
        launch_gui()
        return

    args = parse_args()

    if args.ui:
        launch_gui()
        return

    run_pipeline(
        source=args.source,
        headless=args.headless,
        max_frames=args.max_frames,
        output_path=args.output,
        save_video=args.save_video,
        enable_eval=args.eval or True,
        eval_csv=args.eval_csv,
        eval_json=args.eval_json,
    )


if __name__ == "__main__":
    main()
