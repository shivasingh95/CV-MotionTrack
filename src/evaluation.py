"""
Evaluation and performance benchmarking module for CV-MotionTrack.

Academic Computer Vision project for CSE3010 - Step 9.
Measures system-level metrics across the pipeline:
- Latency & FPS throughput (via high-resolution time.perf_counter)
- Detection statistics (object counts per frame, min/max, zero-detection frames)
- Tracking statistics (active tracks, unique registered IDs, track lifetime)
- Optical flow statistics (detected features, valid tracked points, reinitializations)
- Motion statistics (displacement, image-space velocity, moving vs stationary counts)
- Machine-readable result export (CSV, JSON) and terminal reporting

Academic Disclaimer:
Evaluation measures system-level computational and kinematic behavior.
No formal ground-truth accuracy, precision, or recall is claimed without annotated datasets.
"""

import csv
from dataclasses import asdict, dataclass, field
import json
import os
import time
from typing import Any, Dict, List, Optional

from src.data_models import Detection, MotionData, OpticalFlowPoint, TrackedObject


@dataclass
class FrameMetrics:
    """Telemetry captured for a single processed frame."""
    frame_index: int
    processing_time_sec: float
    fps: float
    detection_count: int
    active_track_count: int
    valid_flow_points: int
    mean_displacement_px: float
    mean_velocity_px_s: float
    moving_objects_count: int
    stationary_objects_count: int


@dataclass
class EvaluationSummary:
    """Aggregated evaluation metrics across an entire video stream or experiment."""
    total_frames: int = 0
    total_processing_time_sec: float = 0.0
    average_processing_time_ms: float = 0.0
    average_fps: float = 0.0
    min_fps: float = 0.0
    max_fps: float = 0.0

    # Detection statistics
    total_detections: int = 0
    average_detections_per_frame: float = 0.0
    max_detections_per_frame: int = 0
    min_detections_per_frame: int = 0
    frames_with_zero_detections: int = 0

    # Tracking statistics
    total_registered_tracks: int = 0
    deregistered_tracks_count: int = 0
    average_active_tracks_per_frame: float = 0.0
    max_active_tracks_per_frame: int = 0
    average_track_lifetime_frames: float = 0.0
    max_track_lifetime_frames: int = 0

    # Optical flow statistics
    average_valid_flow_points: float = 0.0
    min_valid_flow_points: int = 0
    max_valid_flow_points: int = 0
    flow_reinitialization_events: int = 0

    # Motion analysis statistics
    average_displacement_px_frame: float = 0.0
    max_displacement_px_frame: float = 0.0
    average_velocity_px_s: float = 0.0
    max_velocity_px_s: float = 0.0
    total_moving_observations: int = 0
    total_stationary_observations: int = 0


class Evaluator:
    """
    Monitors, profiles, and evaluates the real-time CV-MotionTrack pipeline.

    Collects per-frame metrics with high-resolution performance timers,
    aggregates empirical summary statistics, and exports results to CSV and JSON.
    """

    def __init__(self, experiment_name: str = "default_experiment") -> None:
        """
        Initialize the Evaluator.

        Args:
            experiment_name: Descriptive tag for the benchmark trial.
        """
        self.experiment_name: str = experiment_name
        self.frame_history: List[FrameMetrics] = []
        self._frame_start_time: Optional[float] = None
        self._current_frame_idx: int = 0

        # Persistent track lifetime tracking: object_id -> lifetime (frames)
        self._track_lifetimes: Dict[int, int] = {}
        self._all_registered_ids: set = set()
        self._last_flow_point_count: int = 0
        self._flow_reinit_count: int = 0

    def start_frame(self) -> None:
        """Mark the beginning of processing for a single frame using high-resolution timer."""
        self._frame_start_time = time.perf_counter()

    def record_frame(
        self,
        detections: Optional[List[Detection]] = None,
        tracked_objects: Optional[List[TrackedObject]] = None,
        optical_flow_points: Optional[List[OpticalFlowPoint]] = None,
        motion_data: Optional[Dict[int, MotionData]] = None,
        reinitialized_flow: bool = False,
    ) -> FrameMetrics:
        """
        Record pipeline telemetry for the current frame and compute per-frame metrics.

        Args:
            detections: List of candidate detections.
            tracked_objects: List of actively tracked objects.
            optical_flow_points: List of valid tracked optical flow features.
            motion_data: Mapping of object_id -> MotionData.
            reinitialized_flow: Flag indicating if optical flow re-detected features.

        Returns:
            FrameMetrics: Recorded metrics for this frame.
        """
        # Stop high-resolution timer
        end_time = time.perf_counter()
        if self._frame_start_time is not None:
            proc_time = max(1e-7, end_time - self._frame_start_time)
        else:
            proc_time = 1e-4

        self._current_frame_idx += 1
        frame_fps = 1.0 / proc_time if proc_time > 0 else 0.0

        # 1. Detection metrics
        det_list = detections or []
        det_count = len(det_list)

        # 2. Tracking metrics
        tracks_list = tracked_objects or []
        track_count = len(tracks_list)
        for track in tracks_list:
            oid = track.object_id
            self._all_registered_ids.add(oid)
            self._track_lifetimes[oid] = len(track.trajectory) if track.trajectory else 1

        # 3. Optical flow metrics
        flow_list = optical_flow_points or []
        flow_count = len(flow_list)
        if reinitialized_flow:
            self._flow_reinit_count += 1
        self._last_flow_point_count = flow_count

        # 4. Motion metrics
        m_dict = motion_data or {}
        moving_count = 0
        stationary_count = 0
        displacements: List[float] = []
        velocities: List[float] = []

        for m in m_dict.values():
            displacements.append(m.displacement)
            vel = m.approximate_velocity if m.approximate_velocity > 0 else m.velocity
            velocities.append(vel)
            if m.direction == "STATIONARY":
                stationary_count += 1
            else:
                moving_count += 1

        mean_disp = float(sum(displacements) / len(displacements)) if displacements else 0.0
        mean_vel = float(sum(velocities) / len(velocities)) if velocities else 0.0

        metrics = FrameMetrics(
            frame_index=self._current_frame_idx,
            processing_time_sec=proc_time,
            fps=frame_fps,
            detection_count=det_count,
            active_track_count=track_count,
            valid_flow_points=flow_count,
            mean_displacement_px=mean_disp,
            mean_velocity_px_s=mean_vel,
            moving_objects_count=moving_count,
            stationary_objects_count=stationary_count,
        )
        self.frame_history.append(metrics)
        self._frame_start_time = None
        return metrics

    def get_summary(self) -> EvaluationSummary:
        """
        Aggregate per-frame metrics into an EvaluationSummary.

        Returns:
            EvaluationSummary: Comprehensive aggregated performance and CV metrics.
        """
        total_frames = len(self.frame_history)
        if total_frames == 0:
            return EvaluationSummary()

        proc_times = [m.processing_time_sec for m in self.frame_history]
        fps_values = [m.fps for m in self.frame_history]
        total_time = sum(proc_times)
        avg_proc_ms = (total_time / total_frames) * 1000.0
        avg_fps = total_frames / total_time if total_time > 0 else 0.0

        # Detection aggregates
        det_counts = [m.detection_count for m in self.frame_history]
        total_dets = sum(det_counts)
        avg_dets = float(total_dets / total_frames)
        max_dets = max(det_counts)
        min_dets = min(det_counts)
        zero_det_frames = sum(1 for c in det_counts if c == 0)

        # Tracking aggregates
        track_counts = [m.active_track_count for m in self.frame_history]
        avg_active_tracks = float(sum(track_counts) / total_frames)
        max_active_tracks = max(track_counts)
        total_registered = len(self._all_registered_ids)
        current_active = track_counts[-1] if track_counts else 0
        deregistered_count = max(0, total_registered - current_active)

        lifetimes = list(self._track_lifetimes.values())
        avg_lifetime = float(sum(lifetimes) / len(lifetimes)) if lifetimes else 0.0
        max_lifetime = max(lifetimes) if lifetimes else 0

        # Optical flow aggregates
        flow_counts = [m.valid_flow_points for m in self.frame_history]
        avg_flow = float(sum(flow_counts) / total_frames)
        min_flow = min(flow_counts)
        max_flow = max(flow_counts)

        # Motion aggregates
        disp_values = [m.mean_displacement_px for m in self.frame_history if m.mean_displacement_px > 0]
        vel_values = [m.mean_velocity_px_s for m in self.frame_history if m.mean_velocity_px_s > 0]
        avg_disp = float(sum(disp_values) / len(disp_values)) if disp_values else 0.0
        max_disp = max([m.mean_displacement_px for m in self.frame_history]) if self.frame_history else 0.0
        avg_vel = float(sum(vel_values) / len(vel_values)) if vel_values else 0.0
        max_vel = max([m.mean_velocity_px_s for m in self.frame_history]) if self.frame_history else 0.0

        moving_total = sum(m.moving_objects_count for m in self.frame_history)
        stat_total = sum(m.stationary_objects_count for m in self.frame_history)

        return EvaluationSummary(
            total_frames=total_frames,
            total_processing_time_sec=float(total_time),
            average_processing_time_ms=float(avg_proc_ms),
            average_fps=float(avg_fps),
            min_fps=float(min(fps_values)),
            max_fps=float(max(fps_values)),
            total_detections=total_dets,
            average_detections_per_frame=float(avg_dets),
            max_detections_per_frame=int(max_dets),
            min_detections_per_frame=int(min_dets),
            frames_with_zero_detections=int(zero_det_frames),
            total_registered_tracks=int(total_registered),
            deregistered_tracks_count=int(deregistered_count),
            average_active_tracks_per_frame=float(avg_active_tracks),
            max_active_tracks_per_frame=int(max_active_tracks),
            average_track_lifetime_frames=float(avg_lifetime),
            max_track_lifetime_frames=int(max_lifetime),
            average_valid_flow_points=float(avg_flow),
            min_valid_flow_points=int(min_flow),
            max_valid_flow_points=int(max_flow),
            flow_reinitialization_events=int(self._flow_reinit_count),
            average_displacement_px_frame=float(avg_disp),
            max_displacement_px_frame=float(max_disp),
            average_velocity_px_s=float(avg_vel),
            max_velocity_px_s=float(max_vel),
            total_moving_observations=int(moving_total),
            total_stationary_observations=int(stat_total),
        )

    def print_terminal_summary(self) -> None:
        """Print an undergraduate-friendly formatted evaluation card."""
        s = self.get_summary()
        print("=" * 65)
        print(f"CV-MotionTrack Evaluation: {self.experiment_name}")
        print("=" * 65)
        print(f"Frames processed          : {s.total_frames}")
        print(f"Total processing time     : {s.total_processing_time_sec:.3f} s")
        print(f"Average pipeline FPS      : {s.average_fps:.1f} (min: {s.min_fps:.1f}, max: {s.max_fps:.1f})")
        print(f"Avg processing latency    : {s.average_processing_time_ms:.2f} ms/frame")
        print("-" * 65)
        print(f"Avg detections / frame    : {s.average_detections_per_frame:.2f} (max: {s.max_detections_per_frame}, min: {s.min_detections_per_frame})")
        print(f"Frames with zero detections: {s.frames_with_zero_detections}")
        print("-" * 65)
        print(f"Total registered tracks   : {s.total_registered_tracks}")
        print(f"Deregistered tracks       : {s.deregistered_tracks_count}")
        print(f"Avg track lifetime        : {s.average_track_lifetime_frames:.1f} frames (max: {s.max_track_lifetime_frames})")
        print("-" * 65)
        print(f"Avg optical-flow points   : {s.average_valid_flow_points:.1f} (min: {s.min_valid_flow_points}, max: {s.max_valid_flow_points})")
        print(f"Flow reinitializations    : {s.flow_reinitialization_events}")
        print("-" * 65)
        print(f"Avg image-space disp      : {s.average_displacement_px_frame:.2f} px/frame (max: {s.max_displacement_px_frame:.2f})")
        print(f"Avg image-space velocity  : {s.average_velocity_px_s:.2f} px/s (max: {s.max_velocity_px_s:.2f})")
        print("=" * 65)

    def save_csv(
        self,
        file_path: str,
        video_name: str = "stream",
    ) -> None:
        """
        Append summary metrics as a row in a machine-readable CSV table.

        Args:
            file_path: Destination path for the CSV summary file.
            video_name: Name or path of the evaluated input video.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        file_exists = os.path.exists(file_path)

        summary = self.get_summary()
        row = {
            "experiment": self.experiment_name,
            "video": video_name,
            "frames": summary.total_frames,
            "total_time_s": round(summary.total_processing_time_sec, 3),
            "avg_fps": round(summary.average_fps, 2),
            "avg_processing_ms": round(summary.average_processing_time_ms, 2),
            "avg_detections": round(summary.average_detections_per_frame, 2),
            "max_detections": summary.max_detections_per_frame,
            "zero_det_frames": summary.frames_with_zero_detections,
            "total_tracks": summary.total_registered_tracks,
            "avg_track_lifetime": round(summary.average_track_lifetime_frames, 2),
            "avg_flow_points": round(summary.average_valid_flow_points, 2),
            "flow_reinits": summary.flow_reinitialization_events,
            "avg_displacement_px": round(summary.average_displacement_px_frame, 2),
            "avg_velocity_px_s": round(summary.average_velocity_px_s, 2),
        }

        fieldnames = list(row.keys())
        with open(file_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

    def save_json(
        self,
        file_path: str,
        video_name: str = "stream",
        include_frame_history: bool = True,
    ) -> None:
        """
        Export full structured benchmark evaluation to a JSON file.

        Args:
            file_path: Destination path for JSON report.
            video_name: Evaluated video name.
            include_frame_history: Whether to include per-frame telemetry series.
        """
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        summary = self.get_summary()

        data = {
            "experiment_name": self.experiment_name,
            "video_name": video_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "summary": asdict(summary),
        }

        if include_frame_history:
            data["frame_history"] = [asdict(f) for f in self.frame_history]

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def reset(self) -> None:
        """Clear all metric logs, history buffers, and timer states for a clean run."""
        self.frame_history.clear()
        self._frame_start_time = None
        self._current_frame_idx = 0
        self._track_lifetimes.clear()
        self._all_registered_ids.clear()
        self._last_flow_point_count = 0
        self._flow_reinit_count = 0


# Backward compatibility alias
EvaluationManager = Evaluator
