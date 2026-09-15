"""
Command-Line Interface (CLI) module for CV-MotionTrack.

Academic Computer Vision project for CSE3010.
Coordinates command-line argument parsing, terminal progress reporting,
input stream validation, and pipeline execution without requiring a GUI.
"""

import argparse
from datetime import datetime
import os
import sys
import time
from typing import List, Optional, Tuple

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


def create_arg_parser() -> argparse.ArgumentParser:
    """
    Construct command-line argument parser for CV-MotionTrack.

    Returns:
        argparse.ArgumentParser: Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="CV-MotionTrack: Real-Time Object Detection, Tracking & Motion Analysis (CSE3010)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --help
  python main.py --source webcam
  python main.py --source video --input data/input/synthetic_demo.avi
  python main.py --source video --input data/input/synthetic_demo.avi --output data/output/result.mp4 --save-results
  python main.py --gui
        """,
    )

    parser.add_argument(
        "--source",
        type=str,
        choices=["webcam", "video"],
        default=None,
        help="Input source type: 'webcam' or 'video' (default: 'video' if --input specified, else 'webcam')",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input video file (required when --source video is specified)",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=None,
        help="Minimum contour area in pixels² to be classified as a moving object",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="Maximum Euclidean centroid matching distance in pixels for tracking association",
    )
    parser.add_argument(
        "--max-disappeared",
        type=int,
        default=None,
        help="Maximum consecutive frames an object can be missing before deregistration",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional destination path to save annotated video output (.mp4 or .avi)",
    )
    parser.add_argument(
        "--save-results",
        action="store_true",
        help="Save CSV and JSON evaluation metrics to results/metrics/ upon completion",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the desktop graphical user interface (Tkinter GUI)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode (suppresses OpenCV highgui video window)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to process before exiting (optional)",
    )

    return parser


def parse_cli_args(args_list: Optional[List[str]] = None) -> argparse.Namespace:
    """
    Parse and validate command-line arguments.

    Args:
        args_list: Optional list of argument strings (uses sys.argv[1:] if None).

    Returns:
        argparse.Namespace: Parsed argument namespace.
    """
    parser = create_arg_parser()
    args = parser.parse_args(args_list)

    # Determine default source if --source not explicitly passed
    if args.source is None:
        if args.input is not None:
            args.source = "video"
        elif not args.gui:
            args.source = "webcam"

    return args


def run_cli_pipeline(args: argparse.Namespace) -> int:
    """
    Execute the command-line CV-MotionTrack pipeline with validated options.

    Args:
        args: Parsed command-line arguments.

    Returns:
        int: Exit status code (0 for success, 1 for error).
    """
    # 1. Determine and Validate Input Source
    source_val: str
    if args.source == "webcam":
        source_val = "0"
        display_source_name = "Webcam (Device 0)"
    elif args.source == "video":
        if not args.input:
            print("ERROR: --input path is required when --source video is specified.", file=sys.stderr)
            print("Usage example: python main.py --source video --input data/input/synthetic_demo.avi", file=sys.stderr)
            return 1

        norm_input = os.path.normpath(os.path.abspath(args.input))
        if not os.path.exists(norm_input):
            print(f"ERROR: Input video does not exist:\n{args.input}", file=sys.stderr)
            return 1
        source_val = norm_input
        display_source_name = os.path.basename(norm_input)
    else:
        # Fallback to webcam if no source provided
        source_val = "0"
        display_source_name = "Webcam (Device 0)"

    # 2. Configure Subsystem Parameters
    config = AppConfig()

    if args.min_area is not None:
        config.detector.min_contour_area = args.min_area
    if args.max_distance is not None:
        config.tracker.max_distance = args.max_distance
    if args.max_disappeared is not None:
        config.tracker.max_disappeared = args.max_disappeared
    if args.output is not None:
        config.output.save_video = True
        config.output.output_path = args.output

    # 3. Initialize Pipeline Modules
    try:
        video_processor = VideoProcessor(source=source_val, config=config.video)
        if not video_processor.open():
            print(f"ERROR: Unable to open video source: {source_val}", file=sys.stderr)
            return 1
    except FileNotFoundError as e:
        print(f"ERROR: Input video does not exist:\n{args.input}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Failed to open video source:\n{e}", file=sys.stderr)
        return 1

    preprocessor = Preprocessor(config=config.preprocessing)
    detector = ObjectDetector(config=config.detector)
    tracker = ObjectTracker(config=config.tracker)
    optical_flow = OpticalFlowAnalyzer(config=config.optical_flow)
    motion_analyzer = MotionAnalyzer(config=config.motion)
    visualizer = Visualizer(config=config.visualization)
    evaluator = Evaluator(experiment_name=f"cli_{display_source_name}")

    # Stream properties
    props = video_processor.get_properties()
    res_w = props.get("width", 640)
    res_h = props.get("height", 480)
    fps_prop = props.get("fps", 30.0)
    total_frames = props.get("total_frames", 0)

    total_str = str(total_frames) if total_frames > 0 else "Unknown"

    print("\nCV-MotionTrack")
    print("-" * 35)
    print(f"Source: {display_source_name}")
    print(f"Resolution: {res_w}x{res_h}")
    print(f"FPS: {fps_prop:.1f}")
    print("-" * 35)
    print("Processing:")

    # Video Writer Initialization
    writer: Optional[cv2.VideoWriter] = None
    if config.output.save_video and config.output.output_path:
        out_dir = os.path.dirname(config.output.output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*config.output.codec)
        writer = cv2.VideoWriter(
            config.output.output_path,
            fourcc,
            fps_prop if fps_prop > 0 else 30.0,
            (res_w, res_h),
        )

    # 4. Processing Loop
    frame_idx = 0
    t_pipeline_start = time.perf_counter()
    fps_history: List[float] = []

    try:
        while True:
            t_frame_start = time.perf_counter()
            ret, frame = video_processor.read_frame()
            if not ret or frame is None:
                break

            frame_idx += 1
            evaluator.start_frame()

            # Algorithm Pipeline Execution
            prep_frame = preprocessor.process(frame)
            detections, fg_mask = detector.detect(prep_frame)
            tracked_objects = tracker.update(detections)
            optical_flow_points = optical_flow.update(prep_frame)
            motion_data = motion_analyzer.update(
                tracked_objects=tracked_objects,
                optical_flow_points=optical_flow_points,
            )

            # Record frame metrics
            frame_m = evaluator.record_frame(
                detections=detections,
                tracked_objects=tracked_objects,
                optical_flow_points=optical_flow_points,
                motion_data=motion_data,
            )

            # Compute smoothed FPS
            t_frame_elapsed = time.perf_counter() - t_frame_start
            inst_fps = 1.0 / max(0.0001, t_frame_elapsed)
            fps_history.append(inst_fps)
            if len(fps_history) > 10:
                fps_history.pop(0)
            avg_proc_fps = float(sum(fps_history) / len(fps_history))

            # Composite Visual Annotations
            stats = motion_analyzer.get_summary_statistics()
            annotated_frame = visualizer.render(
                frame=frame,
                detections=detections,
                tracked_objects=tracked_objects,
                motion_data=motion_data,
                optical_flow_points=optical_flow_points,
                active_count=len(tracked_objects),
                fps=avg_proc_fps,
                stats=stats,
            )

            if writer is not None:
                writer.write(annotated_frame)

            # Terminal Progress Logging
            if frame_idx == 1 or frame_idx % 10 == 0:
                print(
                    f"Frame: {frame_idx}/{total_str} | "
                    f"Detected Objects: {len(detections)} | "
                    f"Active Tracks: {len(tracked_objects)} | "
                    f"Flow Points: {len(optical_flow_points)} | "
                    f"Processing FPS: {avg_proc_fps:.1f}"
                )

            # Interactive Window (if not headless)
            if not args.headless:
                cv2.imshow("CV-MotionTrack - Output", annotated_frame)
                cv2.imshow("CV-MotionTrack - Foreground Mask", fg_mask)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] User requested termination.")
                    break

            if args.max_frames and frame_idx >= args.max_frames:
                print(f"\n[INFO] Reached requested max frame limit: {args.max_frames}")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        video_processor.release()
        if writer is not None:
            writer.release()
        if not args.headless:
            cv2.destroyAllWindows()

    # 5. Final Summary Output
    t_total = time.perf_counter() - t_pipeline_start
    final_avg_fps = frame_idx / max(0.0001, t_total)

    print("-" * 35)
    print("Processing completed.")
    print(f"Frames processed: {frame_idx}")
    print(f"Average FPS: {final_avg_fps:.1f}")

    if config.output.save_video and config.output.output_path:
        print(f"Annotated video saved to: {config.output.output_path}")

    if args.save_results or config.output.save_video:
        os.makedirs(os.path.join("results", "metrics"), exist_ok=True)
        csv_path = os.path.join("results", "metrics", "experiment_results.csv")
        json_path = os.path.join("results", "metrics", "experiment_results.json")
        evaluator.save_csv(csv_path, video_name=display_source_name)
        evaluator.save_json(json_path, video_name=display_source_name)
        print(f"Results saved to: {csv_path}")

    return 0
