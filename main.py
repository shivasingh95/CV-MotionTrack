"""
Main pipeline coordinator for CV-MotionTrack.

Academic Computer Vision project for CSE3010.
Executes the sequential modular pipeline:
  Video Input -> Frame Extraction -> Preprocessing -> Background Subtraction ->
  Object Detection -> Object Tracking -> Optical Flow -> Motion Analysis ->
  Visualization -> Output

Note: All algorithmic logic is encapsulated within respective src/ modules.
"""

import argparse
import sys
import time
from typing import Optional
import cv2

from src.config import AppConfig
from src.detector import ObjectDetector
from src.motion_analysis import MotionAnalyzer
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for pipeline execution."""
    parser = argparse.ArgumentParser(
        description="CV-MotionTrack: Real-Time Object Detection, Tracking and Motion Analysis"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source: webcam index (e.g., '0') or path to video file (default: '0')",
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
    return parser.parse_args()


def run_pipeline(
    source: str = "0",
    headless: bool = False,
    max_frames: Optional[int] = None,
    output_path: Optional[str] = None,
) -> None:
    """
    Coordinate and execute the CV-MotionTrack processing pipeline.

    Args:
        source: Camera index string or path to video file.
        headless: If True, suppress cv2.imshow GUI display.
        max_frames: Optional frame limit.
        output_path: Optional target video path for saving results.
    """
    # Parse source as integer device index if numeric
    video_source = int(source) if source.isdigit() else source

    # Initialize configuration
    config = AppConfig()
    config.video.source = video_source

    print("=" * 60)
    print("CV-MotionTrack: Computer Vision Pipeline Initializing")
    print(f"Source: {video_source}")
    print(f"Subtractor: {config.detector.subtractor_type}")
    print(f"Headless mode: {headless}")
    print("=" * 60)

    # Initialize pipeline subsystems
    video_processor = VideoProcessor(config=config.video)
    preprocessor = Preprocessor(config=config.preprocessing)
    detector = ObjectDetector(config=config.detector)
    tracker = ObjectTracker(config=config.tracker)
    optical_flow = OpticalFlowAnalyzer(config=config.optical_flow)
    motion_analyzer = MotionAnalyzer(config=config.motion)
    visualizer = Visualizer(config=config.visualization)

    # Open video stream
    try:
        if not video_processor.open():
            print(f"[ERROR] Failed to open video source: {video_source}", file=sys.stderr)
            return
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return

    writer: Optional[cv2.VideoWriter] = None
    frame_idx = 0
    start_time = time.time()
    fps_estimate = 0.0

    try:
        while True:
            t_frame_start = time.time()

            # 1. Video Input & Frame Extraction
            ret, frame = video_processor.read_frame()
            if not ret or frame is None:
                print("\n[INFO] End of video stream reached or no frames received.")
                break

            frame_idx += 1

            # 2. Preprocessing (Grayscale + Gaussian Spatial Smoothing)
            preprocessed_frame = preprocessor.process(frame)

            # 3. Background Subtraction & 4. Object Detection (Contours -> BBoxes)
            detections, fg_mask = detector.detect(preprocessed_frame)

            # 5. Object Tracking (Centroid Association & Trajectories)
            tracked_objects = tracker.update(detections)

            # 6. Optical Flow (Lucas-Kanade apparent motion estimate)
            _ = optical_flow.update(preprocessed_frame)

            # 7. Motion Analysis (Displacement, Direction, Velocity Kinematics)
            motion_data = motion_analyzer.update(tracked_objects)

            # Calculate instantaneous throughput
            elapsed_frame = time.time() - t_frame_start
            if elapsed_frame > 0:
                fps_estimate = 0.9 * fps_estimate + 0.1 * (1.0 / elapsed_frame)

            # 8. Visualization (Composite HUD, bboxes, IDs, vectors)
            stats = motion_analyzer.get_summary_statistics()
            annotated_frame = visualizer.render(
                frame=frame,
                detections=detections,
                tracked_objects=tracked_objects,
                motion_data=motion_data,
                active_count=len(tracked_objects),
                fps=fps_estimate,
                stats=stats,
            )

            # Optional video writer setup
            if output_path is not None and writer is None:
                h, w = annotated_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                writer = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h))

            if writer is not None:
                writer.write(annotated_frame)

            # 9. Interactive GUI Output
            if not headless:
                cv2.imshow("CV-MotionTrack - Main View", annotated_frame)
                cv2.imshow("CV-MotionTrack - Foreground Mask", fg_mask)

                # Press 'q' or 'ESC' to exit
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[INFO] User requested termination.")
                    break

            if max_frames is not None and frame_idx >= max_frames:
                print(f"\n[INFO] Reached requested max frame limit: {max_frames}")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Pipeline interrupted by user (KeyboardInterrupt).")

    finally:
        total_time = time.time() - start_time
        avg_fps = frame_idx / total_time if total_time > 0 else 0.0

        # Resource teardown
        video_processor.release()
        if writer is not None:
            writer.release()
        if not headless:
            cv2.destroyAllWindows()

        print("=" * 60)
        print("CV-MotionTrack: Execution Summary")
        print(f"Total Frames Processed : {frame_idx}")
        print(f"Total Elapsed Time     : {total_time:.2f} s")
        print(f"Average Pipeline FPS   : {avg_fps:.1f}")

        final_stats = motion_analyzer.get_summary_statistics()
        print(f"Total Objects Tracked  : {final_stats.get('total_objects_tracked', 0)}")
        print(f"Average Object Velocity: {final_stats.get('average_velocity_px_s', 0.0):.1f} px/s")
        print("=" * 60)


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_pipeline(
        source=args.source,
        headless=args.headless,
        max_frames=args.max_frames,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
