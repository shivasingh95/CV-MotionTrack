"""
Desktop User Interface (GUI) module for CV-MotionTrack.

Academic Computer Vision project for CSE3010 - Step 10.
Implements the professional presentation and final demonstration layer:
- Clean Tkinter desktop interface with responsive multi-threaded pipeline execution
- Live aspect-ratio preserving video canvas with idle placeholder
- Complete interactive controls: Start Webcam, Open Video, Process, Pause, Resume, Reset, Save Frame, Stop
- Real-time adjustable vision hyperparameters (Min Area, Tracker Distance, Disappeared Frames, Flow Corners, Var Threshold)
- Real-time statistics telemetry (FPS, Detections, Active Tracks, Flow Points, Moving vs Stationary)
- Live tracked object information table (ID, Direction, Displacement, Velocity, Trajectory)
- Keyboard shortcuts: Q/ESC=Exit, P=Pause/Resume, R=Reset, S=Save Frame
- Timestamped screenshot capture: results/screenshots/motiontrack_YYYY_MM_DD_HHMMSS.png

Academic Note:
The UI functions solely as the presentation controller and does not contain
computer vision algorithms directly. It coordinates the underlying modular components.
"""

from datetime import datetime
import os
import queue
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from src.config import (
    AppConfig,
    DetectorConfig,
    MotionAnalysisConfig,
    OpticalFlowConfig,
    OutputConfig,
    PreprocessingConfig,
    TrackerConfig,
    VideoConfig,
    VisualizationConfig,
)
from src.data_models import Detection, MotionData, OpticalFlowPoint, TrackedObject
from src.detector import ObjectDetector
from src.evaluation import Evaluator
from src.motion_analysis import MotionAnalyzer
from src.optical_flow import OpticalFlowAnalyzer
from src.preprocessing import Preprocessor
from src.tracker import ObjectTracker
from src.video_processor import VideoProcessor
from src.visualizer import Visualizer


class CVMotionTrackApp:
    """
    Main Tkinter desktop application controller for the CV-MotionTrack system.

    Coordinates video acquisition, algorithmic processing across an isolated
    worker thread, real-time GUI telemetry updates, and user interactions.
    """

    def __init__(self, root: tk.Tk, config: Optional[AppConfig] = None) -> None:
        """
        Initialize the CV-MotionTrack GUI application.

        Args:
            root: Root Tkinter window instance.
            config: Optional application configuration. If None, default settings are used.
        """
        self.root = root
        self.root.title("CV-MotionTrack - Real-Time Motion Analysis System")
        self.root.geometry("1240x820")
        self.root.minsize(1080, 720)

        # Apply dark theme styling
        self._setup_styles()

        # Central configuration
        self.config = config or AppConfig()

        # Pipeline subsystems
        self.video_processor: Optional[VideoProcessor] = None
        self.preprocessor = Preprocessor(config=self.config.preprocessing)
        self.detector = ObjectDetector(config=self.config.detector)
        self.tracker = ObjectTracker(config=self.config.tracker)
        self.optical_flow = OpticalFlowAnalyzer(config=self.config.optical_flow)
        self.motion_analyzer = MotionAnalyzer(config=self.config.motion)
        self.visualizer = Visualizer(config=self.config.visualization)
        self.evaluator = Evaluator(experiment_name="ui_demonstration")

        # Application state
        self.current_source: Union[int, str] = 0
        self.source_type: str = "webcam"  # 'webcam' or 'file'
        self.is_processing: bool = False
        self.is_paused: bool = False
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None

        # Thread-safe queue for frame delivery (maxsize 2 drops stale frames to preserve real-time responsiveness)
        self.frame_queue: queue.Queue = queue.Queue(maxsize=2)
        self.latest_annotated_frame: Optional[np.ndarray] = None
        self.photo_image: Optional[ImageTk.PhotoImage] = None

        # Build UI layout
        self._create_widgets()
        self._bind_shortcuts()

        # Handle window close cleanly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Start UI event polling loop
        self.root.after(20, self._poll_queue)

    # ---------------------------------------------------------------------- #
    # UI Styling and Layout                                                  #
    # ---------------------------------------------------------------------- #

    def _setup_styles(self) -> None:
        """Configure modern dark ttk visual styles."""
        self.root.configure(bg="#1e1e1e")
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Color palette
        bg_dark = "#1e1e1e"
        bg_card = "#252526"
        fg_white = "#ffffff"
        fg_light = "#cccccc"
        accent_blue = "#007acc"

        # Frame styles
        style.configure("TFrame", background=bg_dark)
        style.configure("Card.TFrame", background=bg_card, relief="flat")

        # Label styles
        style.configure("TLabel", background=bg_dark, foreground=fg_white, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=bg_card, foreground=fg_white, font=("Segoe UI", 10))
        style.configure("HeaderTitle.TLabel", background=bg_dark, foreground="#4fc1ff", font=("Segoe UI", 18, "bold"))
        style.configure("HeaderSub.TLabel", background=bg_dark, foreground=fg_light, font=("Segoe UI", 10))
        style.configure("StatKey.TLabel", background=bg_card, foreground="#9cdcfe", font=("Segoe UI", 9, "bold"))
        style.configure("StatVal.TLabel", background=bg_card, foreground="#4ec9b0", font=("Consolas", 10, "bold"))

        # Button styles
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=5)
        style.map("TButton", background=[("active", "#37373d"), ("!disabled", "#2d2d30")], foreground=[("!disabled", fg_white)])
        style.configure("Action.TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.map("Action.TButton", background=[("active", "#1f8ad2"), ("!disabled", accent_blue)], foreground=[("!disabled", fg_white)])

        # LabelFrame styles
        style.configure("TLabelframe", background=bg_card, foreground="#4fc1ff")
        style.configure("TLabelframe.Label", background=bg_card, foreground="#4fc1ff", font=("Segoe UI", 10, "bold"))

        # Treeview styles
        style.configure("Treeview", background="#1e1e1e", foreground=fg_white, fieldbackground="#1e1e1e", font=("Consolas", 9), rowheight=22)
        style.configure("Treeview.Heading", background="#2d2d30", foreground="#9cdcfe", font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#094771")], foreground=[("selected", fg_white)])

    def _create_widgets(self) -> None:
        """Construct the main responsive UI grid layout."""
        # Top Header Frame
        header_frame = ttk.Frame(self.root, padding="15 10 15 5")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_label = ttk.Label(header_frame, text="CV-MotionTrack", style="HeaderTitle.TLabel")
        title_label.pack(anchor=tk.W)

        subtitle_label = ttk.Label(
            header_frame,
            text="Real-Time Object Detection, Tracking and Motion Analysis System (CSE3010)",
            style="HeaderSub.TLabel",
        )
        subtitle_label.pack(anchor=tk.W)

        # Horizontal separator
        sep = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=15, pady=5)

        # Main Body Splitter: Left = Video Canvas, Right = Controls & Statistics
        body_paned = ttk.Frame(self.root, padding="15 5 15 15")
        body_paned.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Column: Video View & Status Bar
        left_frame = ttk.Frame(body_paned)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        # Video Canvas Container
        self.video_container = tk.Frame(left_frame, bg="#111111", bd=2, relief=tk.SUNKEN)
        self.video_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(
            self.video_container,
            text="No Video Loaded\n\nClick 'Start Webcam' or 'Open Video' to begin processing.",
            bg="#111111",
            fg="#888888",
            font=("Segoe UI", 12),
            justify=tk.CENTER,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Status Bar beneath video
        status_bar = ttk.Frame(left_frame, padding="5 5 5 0")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.StringVar(value="Status: Ready | Select input source")
        self.status_label = ttk.Label(status_bar, textvariable=self.status_text, font=("Segoe UI", 9, "italic"), foreground="#b5cea8")
        self.status_label.pack(side=tk.LEFT)

        self.source_text = tk.StringVar(value="Source: None")
        self.source_label = ttk.Label(status_bar, textvariable=self.source_text, font=("Segoe UI", 9), foreground="#dcdcaa")
        self.source_label.pack(side=tk.RIGHT)

        # Right Column: Controls, Telemetry HUD, Objects Table, Tunable Settings
        right_frame = ttk.Frame(body_paned, width=440)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_frame.pack_propagate(False)

        # 1. Control Buttons Card
        controls_group = ttk.LabelFrame(right_frame, text=" Pipeline Controls ", padding=10)
        controls_group.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Source Selection
        self.btn_webcam = ttk.Button(controls_group, text="🎥 Start Webcam", command=self.start_webcam)
        self.btn_webcam.grid(row=0, column=0, padx=4, pady=4, sticky="ew")

        self.btn_open_file = ttk.Button(controls_group, text="📁 Open Video", command=self.open_video_dialog)
        self.btn_open_file.grid(row=0, column=1, padx=4, pady=4, sticky="ew")

        # Row 2: Pipeline Execution
        self.btn_start = ttk.Button(controls_group, text="▶ Start Processing", style="Action.TButton", command=self.start_processing)
        self.btn_start.grid(row=1, column=0, padx=4, pady=4, sticky="ew")

        self.btn_stop = ttk.Button(controls_group, text="⏹ Stop", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.grid(row=1, column=1, padx=4, pady=4, sticky="ew")

        # Row 3: Live Actions
        self.btn_pause = ttk.Button(controls_group, text="⏸ Pause", command=self.pause_processing, state=tk.DISABLED)
        self.btn_pause.grid(row=2, column=0, padx=4, pady=4, sticky="ew")

        self.btn_resume = ttk.Button(controls_group, text="⏯ Resume", command=self.resume_processing, state=tk.DISABLED)
        self.btn_resume.grid(row=2, column=1, padx=4, pady=4, sticky="ew")

        # Row 4: Utility Actions
        self.btn_reset = ttk.Button(controls_group, text="🔄 Reset State", command=self.reset_pipeline)
        self.btn_reset.grid(row=3, column=0, padx=4, pady=4, sticky="ew")

        self.btn_save_frame = ttk.Button(controls_group, text="📸 Save Frame", command=self.save_current_frame)
        self.btn_save_frame.grid(row=3, column=1, padx=4, pady=4, sticky="ew")

        controls_group.columnconfigure(0, weight=1)
        controls_group.columnconfigure(1, weight=1)

        # 2. Live Telemetry Statistics Card
        stats_group = ttk.LabelFrame(right_frame, text=" Live Pipeline Telemetry ", padding=10)
        stats_group.pack(fill=tk.X, pady=(0, 10))

        self.stat_vars: Dict[str, tk.StringVar] = {
            "fps": tk.StringVar(value="0.0 FPS"),
            "frame": tk.StringVar(value="0"),
            "detections": tk.StringVar(value="0"),
            "tracks": tk.StringVar(value="0"),
            "flow_points": tk.StringVar(value="0"),
            "moving": tk.StringVar(value="0"),
            "stationary": tk.StringVar(value="0"),
        }

        # Telemetry labels in 2-column grid
        stat_rows = [
            ("Throughput (FPS):", "fps", "Frame Number:", "frame"),
            ("Detected Blobs:", "detections", "Active Tracks:", "tracks"),
            ("Optical Flow Pts:", "flow_points", "Moving / Stationary:", "moving"),
        ]

        for r, (k1, v1, k2, v2) in enumerate(stat_rows):
            ttk.Label(stats_group, text=k1, style="StatKey.TLabel").grid(row=r, column=0, sticky="w", pady=2)
            ttk.Label(stats_group, textvariable=self.stat_vars[v1], style="StatVal.TLabel").grid(row=r, column=1, sticky="w", padx=(0, 10), pady=2)

            ttk.Label(stats_group, text=k2, style="StatKey.TLabel").grid(row=r, column=2, sticky="w", pady=2)
            if v2 == "moving":
                # Special combined label for moving / stationary
                self.stat_moving_stat_var = tk.StringVar(value="0 / 0")
                ttk.Label(stats_group, textvariable=self.stat_moving_stat_var, style="StatVal.TLabel").grid(row=r, column=3, sticky="w", pady=2)
            else:
                ttk.Label(stats_group, textvariable=self.stat_vars[v2], style="StatVal.TLabel").grid(row=r, column=3, sticky="w", pady=2)

        stats_group.columnconfigure(0, weight=1)
        stats_group.columnconfigure(1, weight=1)
        stats_group.columnconfigure(2, weight=1)
        stats_group.columnconfigure(3, weight=1)

        # 3. Tracked Objects Telemetry Table
        table_group = ttk.LabelFrame(right_frame, text=" Tracked Object Kinematics ", padding=8)
        table_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        columns = ("id", "direction", "disp", "velocity", "length")
        self.objects_tree = ttk.Treeview(table_group, columns=columns, show="headings", height=5)

        self.objects_tree.heading("id", text="ID")
        self.objects_tree.heading("direction", text="Direction")
        self.objects_tree.heading("disp", text="Disp (px)")
        self.objects_tree.heading("velocity", text="Vel (px/s)")
        self.objects_tree.heading("length", text="Trail")

        self.objects_tree.column("id", width=42, anchor="center")
        self.objects_tree.column("direction", width=95, anchor="center")
        self.objects_tree.column("disp", width=70, anchor="e")
        self.objects_tree.column("velocity", width=85, anchor="e")
        self.objects_tree.column("length", width=55, anchor="center")

        tree_scroll = ttk.Scrollbar(table_group, orient=tk.VERTICAL, command=self.objects_tree.yview)
        self.objects_tree.configure(yscrollcommand=tree_scroll.set)

        self.objects_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # 4. Tunable Hyperparameters Settings Panel
        settings_group = ttk.LabelFrame(right_frame, text=" Real-Time Vision Settings ", padding=8)
        settings_group.pack(fill=tk.X)

        self.setting_vars = {
            "min_area": tk.DoubleVar(value=float(self.config.detector.min_contour_area)),
            "max_dist": tk.DoubleVar(value=float(self.config.tracker.max_distance)),
            "max_disp": tk.IntVar(value=int(self.config.tracker.max_disappeared)),
            "flow_corners": tk.IntVar(value=int(self.config.optical_flow.max_corners)),
            "var_threshold": tk.DoubleVar(value=float(self.config.detector.var_threshold)),
        }

        settings_specs = [
            ("Min Object Area (px²):", "min_area", 50.0, 3000.0, 50.0),
            ("Tracker Match Dist (px):", "max_dist", 10.0, 150.0, 5.0),
            ("Max Disappeared (frames):", "max_disp", 2, 40, 1),
            ("Optical Flow Max Corners:", "flow_corners", 20, 250, 10),
            ("Background Var Threshold:", "var_threshold", 4.0, 64.0, 2.0),
        ]

        for row_idx, (label_txt, var_key, v_min, v_max, v_step) in enumerate(settings_specs):
            ttk.Label(settings_group, text=label_txt, font=("Segoe UI", 8)).grid(row=row_idx, column=0, sticky="w", pady=2)
            scale = ttk.Scale(
                settings_group,
                from_=v_min,
                to=v_max,
                variable=self.setting_vars[var_key],
                orient=tk.HORIZONTAL,
                command=lambda val, k=var_key: self._on_setting_changed(k, val),
            )
            scale.grid(row=row_idx, column=1, sticky="ew", padx=6, pady=2)
            val_lbl = ttk.Label(settings_group, width=6, font=("Consolas", 8))
            val_lbl.grid(row=row_idx, column=2, sticky="w", pady=2)

            # Link live value label
            if isinstance(self.setting_vars[var_key], tk.IntVar):
                val_lbl.configure(textvariable=self.setting_vars[var_key])
            else:
                # Custom trace for float formatting
                def _update_float_lbl(*args, lbl=val_lbl, k=var_key):
                    lbl.configure(text=f"{self.setting_vars[k].get():.0f}")
                self.setting_vars[var_key].trace_add("write", _update_float_lbl)
                val_lbl.configure(text=f"{self.setting_vars[var_key].get():.0f}")

        settings_group.columnconfigure(1, weight=1)

    def _bind_shortcuts(self) -> None:
        """Bind application-wide keyboard shortcuts."""
        self.root.bind("<q>", lambda e: self.on_closing())
        self.root.bind("<Q>", lambda e: self.on_closing())
        self.root.bind("<Escape>", lambda e: self.on_closing())
        self.root.bind("<p>", lambda e: self.toggle_pause())
        self.root.bind("<P>", lambda e: self.toggle_pause())
        self.root.bind("<r>", lambda e: self.reset_pipeline())
        self.root.bind("<R>", lambda e: self.reset_pipeline())
        self.root.bind("<s>", lambda e: self.save_current_frame())
        self.root.bind("<S>", lambda e: self.save_current_frame())

    # ---------------------------------------------------------------------- #
    # Settings Management                                                    #
    # ---------------------------------------------------------------------- #

    def _on_setting_changed(self, setting_key: str, raw_val: Any) -> None:
        """Dynamically propagate hyperparameter adjustments to active modules."""
        try:
            val = float(raw_val)
            if setting_key == "min_area":
                self.config.detector.min_contour_area = val
                self.detector.min_contour_area = val
            elif setting_key == "max_dist":
                self.config.tracker.max_distance = val
                self.tracker.max_distance = val
            elif setting_key == "max_disp":
                ival = int(val)
                self.config.tracker.max_disappeared = ival
                self.tracker.max_disappeared = ival
            elif setting_key == "flow_corners":
                ival = int(val)
                self.config.optical_flow.max_corners = ival
                self.optical_flow.max_corners = ival
            elif setting_key == "var_threshold":
                self.config.detector.var_threshold = val
                if hasattr(self.detector, "subtractor") and self.detector.subtractor is not None:
                    try:
                        self.detector.subtractor.setVarThreshold(val)
                    except AttributeError:
                        pass
        except Exception as e:
            print(f"[DEBUG] Setting change error: {e}", file=sys.stderr)

    # ---------------------------------------------------------------------- #
    # Source Selection & Pipeline Control                                    #
    # ---------------------------------------------------------------------- #

    def start_webcam(self) -> None:
        """Configure input source as default webcam."""
        if self.is_processing:
            self.stop_processing()
        self.current_source = 0
        self.source_type = "webcam"
        self.source_text.set("Source: Webcam (Device 0)")
        self.status_text.set("Status: Webcam selected. Click 'Start Processing'.")

    def open_video_dialog(self) -> None:
        """Open file dialog for video selection."""
        if self.is_processing:
            self.stop_processing()

        filetypes = [
            ("Video Files", "*.avi *.mp4 *.mov *.mkv *.wmv"),
            ("AVI Videos", "*.avi"),
            ("MP4 Videos", "*.mp4"),
            ("All Files", "*.*"),
        ]
        initial_dir = os.path.abspath(os.path.join("data", "input"))
        if not os.path.exists(initial_dir):
            initial_dir = os.getcwd()

        chosen_path = filedialog.askopenfilename(
            title="Select Video File for CV-MotionTrack",
            initialdir=initial_dir,
            filetypes=filetypes,
        )

        if chosen_path:
            self.current_source = chosen_path
            self.source_type = "file"
            base_name = os.path.basename(chosen_path)
            self.source_text.set(f"Source: {base_name}")
            self.status_text.set(f"Status: Loaded '{base_name}'. Click 'Start Processing'.")

    def start_processing(self) -> None:
        """Initialize pipeline subsystems and launch background processing thread."""
        if self.is_processing:
            return

        # Prepare VideoProcessor
        self.config.video.source = self.current_source
        self.video_processor = VideoProcessor(config=self.config.video)

        try:
            if not self.video_processor.open():
                messagebox.showerror(
                    "Video Source Error",
                    f"Unable to open video source: {self.current_source}\n"
                    "If using a webcam, ensure it is plugged in and not in use by another app.",
                )
                self.status_text.set("Status: Error opening video source.")
                return
        except FileNotFoundError as fnf_err:
            messagebox.showerror("File Not Found", str(fnf_err))
            self.status_text.set("Status: Video file not found.")
            return
        except Exception as e:
            messagebox.showerror("Initialization Error", f"Unexpected error opening source: {e}")
            return

        # Synchronize FPS with MotionAnalyzer
        stream_fps = self.video_processor.fps
        self.motion_analyzer.config.fps = stream_fps

        # Update button states
        self.is_processing = True
        self.is_paused = False
        self.stop_event.clear()

        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_pause.configure(state=tk.NORMAL)
        self.btn_resume.configure(state=tk.DISABLED)
        self.status_text.set("Status: Processing active...")

        # Spawn background processing worker
        self.worker_thread = threading.Thread(target=self._processing_worker, daemon=True)
        self.worker_thread.start()

    def pause_processing(self) -> None:
        """Pause video processing loop."""
        if self.is_processing and not self.is_paused:
            self.is_paused = True
            self.btn_pause.configure(state=tk.DISABLED)
            self.btn_resume.configure(state=tk.NORMAL)
            self.status_text.set("Status: Processing PAUSED. Press 'Resume' (or P).")

    def resume_processing(self) -> None:
        """Resume video processing loop."""
        if self.is_processing and self.is_paused:
            self.is_paused = False
            self.btn_pause.configure(state=tk.NORMAL)
            self.btn_resume.configure(state=tk.DISABLED)
            self.status_text.set("Status: Processing resumed.")

    def toggle_pause(self) -> None:
        """Toggle pause/resume state."""
        if self.is_paused:
            self.resume_processing()
        else:
            self.pause_processing()

    def stop_processing(self) -> None:
        """Safely terminate processing thread and release video hardware."""
        if not self.is_processing:
            return

        self.status_text.set("Status: Stopping pipeline...")
        self.stop_event.set()

        if self.worker_thread is not None and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None

        if self.video_processor is not None:
            self.video_processor.release()
            self.video_processor = None

        self.is_processing = False
        self.is_paused = False

        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_pause.configure(state=tk.DISABLED)
        self.btn_resume.configure(state=tk.DISABLED)
        self.status_text.set("Status: Processing stopped.")

    def reset_pipeline(self) -> None:
        """
        Clear algorithmic state across tracker, motion analyzer, and optical flow.

        Keeps loaded video/webcam source active so user can re-process.
        """
        self.tracker.reset()
        self.motion_analyzer.reset()
        self.optical_flow.reset()
        self.evaluator.reset()

        # Clear telemetry displays
        for v in self.stat_vars.values():
            v.set("0")
        self.stat_vars["fps"].set("0.0 FPS")
        self.stat_moving_stat_var.set("0 / 0")

        # Clear Treeview
        for row in self.objects_tree.get_children():
            self.objects_tree.delete(row)

        self.status_text.set("Status: Pipeline state reset (Trackers & Kinematics cleared).")

    # ---------------------------------------------------------------------- #
    # Frame Capture & Export                                                 #
    # ---------------------------------------------------------------------- #

    def save_current_frame(self) -> Optional[str]:
        """
        Save currently displayed annotated video frame to results/screenshots/.

        Filename format: motiontrack_YYYY_MM_DD_HHMMSS.png

        Returns:
            Optional[str]: Saved file path if successful, None otherwise.
        """
        if self.latest_annotated_frame is None:
            # No active frame available; update status bar silently
            # (avoid blocking messagebox so callers/tests are not hung)
            try:
                self.status_text.set("Status: No active frame to save.")
            except Exception:
                pass
            return None

        out_dir = os.path.join("results", "screenshots")
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        filename = f"motiontrack_{timestamp}.png"
        filepath = os.path.join(out_dir, filename)

        try:
            cv2.imwrite(filepath, self.latest_annotated_frame)
            self.status_text.set(f"Status: Screenshot saved to {filename}")
            return filepath
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save image: {e}")
            return None

    # ---------------------------------------------------------------------- #
    # Threading & Processing Worker                                          #
    # ---------------------------------------------------------------------- #

    def _processing_worker(self) -> None:
        """
        Worker thread executing the 8-stage Computer Vision pipeline sequentially.

        Places composite annotated frames and metrics into self.frame_queue.
        """
        frame_idx = 0
        fps_estimate = 0.0

        try:
            while not self.stop_event.is_set():
                if self.is_paused:
                    time.sleep(0.03)
                    continue

                t0 = time.perf_counter()

                # Step 1: Read frame
                if self.video_processor is None or not self.video_processor.is_opened():
                    break

                ret, frame = self.video_processor.read_frame()
                if not ret or frame is None:
                    # End of stream reached
                    break

                frame_idx += 1

                # Step 2: Preprocess
                gray_frame = self.preprocessor.process(frame)

                # Step 3, 4, 5: Detect
                detections, fg_mask = self.detector.detect(gray_frame)

                # Step 6: Track
                tracked_objects = self.tracker.update(detections)

                # Step 7: Optical Flow
                flow_points = self.optical_flow.update(gray_frame)

                # Step 8: Motion Analysis
                motion_data = self.motion_analyzer.update(tracked_objects, flow_points)

                # Throughput measurement
                elapsed = max(1e-6, time.perf_counter() - t0)
                fps_val = 1.0 / elapsed
                fps_estimate = 0.9 * fps_estimate + 0.1 * fps_val if fps_estimate > 0 else fps_val

                # Step 9: Render annotations
                stats = self.motion_analyzer.get_summary_statistics()
                annotated = self.visualizer.render(
                    frame=frame,
                    detections=detections,
                    tracked_objects=tracked_objects,
                    motion_data=motion_data,
                    optical_flow_points=flow_points,
                    active_count=len(tracked_objects),
                    fps=fps_estimate,
                    stats=stats,
                )

                # Cache latest frame for screenshot exporter
                self.latest_annotated_frame = annotated

                # Extract object rows for the GUI Treeview
                object_rows = []
                for track in tracked_objects:
                    oid = track.object_id
                    m = motion_data.get(oid)
                    direction = m.direction if m else "STATIONARY"
                    disp = f"{m.displacement:.1f}" if m else "0.0"
                    vel = f"{m.approximate_velocity:.1f}" if (m and m.approximate_velocity > 0) else (f"{m.velocity:.1f}" if m else "0.0")
                    trail_len = len(track.trajectory) if track.trajectory else 1
                    object_rows.append((oid, direction, disp, vel, trail_len))

                # Queue frame for main GUI thread (drop old frame if queue full)
                payload = {
                    "annotated_frame": annotated,
                    "frame_idx": frame_idx,
                    "fps": fps_estimate,
                    "detections_count": len(detections),
                    "tracks_count": len(tracked_objects),
                    "flow_count": len(flow_points),
                    "moving_count": stats.get("moving_objects", 0),
                    "stationary_count": stats.get("stationary_objects", 0),
                    "object_rows": object_rows,
                }

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.frame_queue.put(payload)

        except Exception as e:
            print(f"[ERROR in CV Worker Thread] {e}", file=sys.stderr)
        finally:
            # Notify main thread that stream finished
            self.root.after(0, self._on_worker_finished)

    def _on_worker_finished(self) -> None:
        """Handle end-of-stream cleanup on the main Tkinter thread."""
        if self.is_processing:
            self.is_processing = False
            self.btn_start.configure(state=tk.NORMAL)
            self.btn_stop.configure(state=tk.DISABLED)
            self.btn_pause.configure(state=tk.DISABLED)
            self.btn_resume.configure(state=tk.DISABLED)
            self.status_text.set("Status: Video playback completed or source ended.")

    # ---------------------------------------------------------------------- #
    # GUI Polling and Display                                                #
    # ---------------------------------------------------------------------- #

    def _poll_queue(self) -> None:
        """Periodic main-thread polling loop to consume frames and update widgets."""
        try:
            payload = None
            # Get latest available frame from queue
            while not self.frame_queue.empty():
                payload = self.frame_queue.get_nowait()

            if payload is not None:
                # Update Video Canvas with aspect-ratio scaling
                bgr_frame = payload["annotated_frame"]
                self._display_frame(bgr_frame)

                # Update Telemetry Labels
                self.stat_vars["frame"].set(str(payload["frame_idx"]))
                self.stat_vars["fps"].set(f"{payload['fps']:.1f} FPS")
                self.stat_vars["detections"].set(str(payload["detections_count"]))
                self.stat_vars["tracks"].set(str(payload["tracks_count"]))
                self.stat_vars["flow_points"].set(str(payload["flow_count"]))
                self.stat_moving_stat_var.set(f"{payload['moving_count']} / {payload['stationary_count']}")

                # Update Object Kinematics Treeview
                self._update_treeview(payload["object_rows"])

        except Exception as e:
            print(f"[DEBUG GUI Poll Error] {e}", file=sys.stderr)

        # Re-arm polling loop
        self.root.after(25, self._poll_queue)

    def _display_frame(self, bgr_image: np.ndarray) -> None:
        """Resize frame maintaining aspect ratio and blit onto video_label."""
        canvas_w = max(100, self.video_container.winfo_width())
        canvas_h = max(100, self.video_container.winfo_height())

        img_h, img_w = bgr_image.shape[:2]
        if img_h == 0 or img_w == 0:
            return

        scale = min(canvas_w / img_w, canvas_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))

        resized = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)

        self.photo_image = ImageTk.PhotoImage(image=pil_img)
        self.video_label.configure(image=self.photo_image, text="")

    def _update_treeview(self, object_rows: List[Tuple]) -> None:
        """Populate the Tracked Objects table with fresh kinematic records."""
        # Simple differential update: clear and re-insert active tracks
        existing_items = self.objects_tree.get_children()
        for item in existing_items:
            self.objects_tree.delete(item)

        for row in object_rows:
            self.objects_tree.insert("", tk.END, values=row)

    def on_closing(self) -> None:
        """Graceful application shutdown."""
        self.stop_processing()
        self.root.destroy()
