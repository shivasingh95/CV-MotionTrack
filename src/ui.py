"""
Desktop User Interface (GUI) module for CV-MotionTrack.

Academic Computer Vision project for CSE3010 - Step 10 & 11.
Implements a professional, highly readable presentation and demonstration layer:
- Clean Tkinter desktop interface with responsive multi-threaded pipeline execution
- High-contrast Slate theme with clear visual hierarchy and readable typography
- Live aspect-ratio preserving video canvas with idle helper artwork
- Step-by-step control panel: Input Source -> Pipeline Execution -> Live Telemetry
- Large, bold telemetry cards (FPS, Detections, Active Tracks, Flow Points, Velocity)
- Clear Tracked Objects Kinematics table with custom column formatting
- Intuitive vision hyperparameter tuning sliders with helpful plain-English descriptions
- Keyboard shortcuts: Q/ESC=Exit, P=Pause/Resume, R=Reset, S=Save Frame
- Timestamped screenshot capture saved to results/screenshots/
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
        self.root.geometry("1380x880")
        self.root.minsize(1180, 750)

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

        # Thread-safe queue for frame delivery (maxsize 2 drops stale frames)
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
        """Configure high-contrast, modern slate visual styles."""
        # Deep Slate Palette (Tailwind Slate-900 / Slate-800)
        self.bg_dark = "#0f172a"
        self.bg_card = "#1e293b"
        self.bg_card_border = "#334155"
        self.fg_white = "#f8fafc"
        self.fg_muted = "#94a3b8"
        self.accent_cyan = "#38bdf8"
        self.accent_blue = "#0284c7"
        self.accent_emerald = "#34d399"
        self.accent_purple = "#a78bfa"
        self.accent_amber = "#fbbf24"

        self.root.configure(bg=self.bg_dark)
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Base Frame Styles
        style.configure("TFrame", background=self.bg_dark)
        style.configure("Card.TFrame", background=self.bg_card, relief="flat")

        # Label Styles
        style.configure("TLabel", background=self.bg_dark, foreground=self.fg_white, font=("Segoe UI", 10))
        style.configure("Card.TLabel", background=self.bg_card, foreground=self.fg_white, font=("Segoe UI", 10))
        style.configure("HeaderTitle.TLabel", background=self.bg_dark, foreground=self.accent_cyan, font=("Segoe UI", 18, "bold"))
        style.configure("HeaderSub.TLabel", background=self.bg_dark, foreground=self.fg_muted, font=("Segoe UI", 10))
        style.configure("SectionTitle.TLabel", background=self.bg_card, foreground=self.accent_cyan, font=("Segoe UI", 11, "bold"))

        # Stat Card Styles
        style.configure("StatBox.TFrame", background="#090d16", relief="solid", borderwidth=1)
        style.configure("StatTitle.TLabel", background="#090d16", foreground=self.fg_muted, font=("Segoe UI", 9, "bold"))

        # Button Styles
        style.configure("TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.map("TButton", background=[("active", "#334155"), ("!disabled", "#334155")], foreground=[("!disabled", self.fg_white)])

        style.configure("Action.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.map("Action.TButton", background=[("active", "#0369a1"), ("!disabled", self.accent_blue)], foreground=[("!disabled", self.fg_white)])

        style.configure("Stop.TButton", font=("Segoe UI", 9, "bold"), padding=6)
        style.map("Stop.TButton", background=[("active", "#be123c"), ("!disabled", "#e11d48")], foreground=[("!disabled", self.fg_white)])

        # LabelFrame Styles
        style.configure("TLabelframe", background=self.bg_card, foreground=self.accent_cyan, borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=self.bg_card, foreground=self.accent_cyan, font=("Segoe UI", 10, "bold"))

        # Treeview Styles (Tracked Objects Table)
        style.configure("Treeview", background="#090d16", foreground=self.fg_white, fieldbackground="#090d16", font=("Consolas", 10), rowheight=26)
        style.configure("Treeview.Heading", background="#334155", foreground=self.accent_cyan, font=("Segoe UI", 9, "bold"))
        style.map("Treeview", background=[("selected", "#0284c7")], foreground=[("selected", self.fg_white)])

        # Scrollbar
        style.configure("TScrollbar", background=self.bg_card, troughcolor="#090d16", borderwidth=0)

    def _create_widgets(self) -> None:
        """Construct the intuitive, readable responsive UI layout."""
        # Top Header Bar
        header_frame = ttk.Frame(self.root, padding="15 12 15 8")
        header_frame.pack(side=tk.TOP, fill=tk.X)

        title_box = ttk.Frame(header_frame)
        title_box.pack(side=tk.LEFT)

        title_label = ttk.Label(title_box, text="CV-MotionTrack", style="HeaderTitle.TLabel")
        title_label.pack(anchor=tk.W)

        subtitle_label = ttk.Label(
            title_box,
            text="Real-Time Object Detection, Multi-Target Tracking & Motion Kinematics System (CSE3010)",
            style="HeaderSub.TLabel",
        )
        subtitle_label.pack(anchor=tk.W)

        # Status Badge Indicator at Top Right
        self.badge_frame = tk.Frame(header_frame, bg="#334155", padx=12, pady=6)
        self.badge_frame.pack(side=tk.RIGHT)

        self.status_badge_var = tk.StringVar(value="● READY")
        self.status_badge_lbl = tk.Label(
            self.badge_frame,
            textvariable=self.status_badge_var,
            bg="#334155",
            fg="#38bdf8",
            font=("Segoe UI", 11, "bold"),
        )
        self.status_badge_lbl.pack()

        # Horizontal Divider
        sep = ttk.Separator(self.root, orient=tk.HORIZONTAL)
        sep.pack(fill=tk.X, padx=15, pady=4)

        # Main Body Splitter
        body_container = ttk.Frame(self.root, padding="15 5 15 15")
        body_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ------------------------------------------------------------------ #
        # LEFT COLUMN: Live Video View & Stream Info                          #
        # ------------------------------------------------------------------ #
        left_frame = ttk.Frame(body_container)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 12))

        # Video Canvas Container
        self.video_container = tk.Frame(left_frame, bg="#020617", bd=2, relief=tk.SOLID, highlightbackground="#334155", highlightthickness=1)
        self.video_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(
            self.video_container,
            text="🎥  CV-MotionTrack Ready\n\n1. Select Input Source ('Start Webcam' or 'Open Video')\n2. Click '▶ Start Processing' to run live motion analysis",
            bg="#020617",
            fg="#94a3b8",
            font=("Segoe UI", 13),
            justify=tk.CENTER,
        )
        self.video_label.pack(fill=tk.BOTH, expand=True)

        # Stream Status Footer beneath video
        status_bar = ttk.Frame(left_frame, padding="6 6 6 0")
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.StringVar(value="Status: System initialized. Select video source.")
        self.status_label = ttk.Label(status_bar, textvariable=self.status_text, font=("Segoe UI", 10, "italic"), foreground="#34d399")
        self.status_label.pack(side=tk.LEFT)

        self.source_text = tk.StringVar(value="Source: Webcam (Device 0)")
        self.source_label = tk.Label(status_bar, textvariable=self.source_text, font=("Segoe UI", 10, "bold"), bg=self.bg_dark, fg="#fde047")
        self.source_label.pack(side=tk.RIGHT)

        # ------------------------------------------------------------------ #
        # RIGHT COLUMN: Control Panel, Live Telemetry Cards & Vision Settings #
        # ------------------------------------------------------------------ #
        right_frame = ttk.Frame(body_container, width=480)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(4, 0))
        right_frame.pack_propagate(False)

        # ------------------------------------------------------------------ #
        # 1. INPUT SOURCE & EXECUTION CONTROL PANEL                         #
        # ------------------------------------------------------------------ #
        controls_group = ttk.LabelFrame(right_frame, text=" 1. Input Source & Pipeline Control ", padding=10)
        controls_group.pack(fill=tk.X, pady=(0, 10))

        # Row 1: Source Selection
        src_frame = ttk.Frame(controls_group)
        src_frame.pack(fill=tk.X, pady=(0, 6))

        self.btn_webcam = ttk.Button(src_frame, text="🎥 Use Webcam", command=self.start_webcam)
        self.btn_webcam.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_open_file = ttk.Button(src_frame, text="📁 Open Video File...", command=self.open_video_dialog)
        self.btn_open_file.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Row 2: Primary Start / Stop Execution
        exec_frame = ttk.Frame(controls_group)
        exec_frame.pack(fill=tk.X, pady=(0, 6))

        self.btn_start = ttk.Button(exec_frame, text="▶ START PROCESSING", style="Action.TButton", command=self.start_processing)
        self.btn_start.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 4))

        self.btn_stop = ttk.Button(exec_frame, text="⏹ STOP", style="Stop.TButton", command=self.stop_processing, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(4, 0))

        # Row 3: Live Control Actions
        action_frame = ttk.Frame(controls_group)
        action_frame.pack(fill=tk.X)

        self.btn_pause = ttk.Button(action_frame, text="⏸ Pause", command=self.pause_processing, state=tk.DISABLED)
        self.btn_pause.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.btn_resume = ttk.Button(action_frame, text="⏯ Resume", command=self.resume_processing, state=tk.DISABLED)
        self.btn_resume.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_reset = ttk.Button(action_frame, text="🔄 Reset", command=self.reset_pipeline)
        self.btn_reset.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)

        self.btn_save_frame = ttk.Button(action_frame, text="📸 Save Frame", command=self.save_current_frame)
        self.btn_save_frame.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(2, 0))

        # ------------------------------------------------------------------ #
        # 2. LIVE PIPELINE TELEMETRY STATS HUD CARDS                        #
        # ------------------------------------------------------------------ #
        stats_group = ttk.LabelFrame(right_frame, text=" 2. Live Telemetry HUD ", padding=10)
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
        self.stat_moving_stat_var = tk.StringVar(value="0 / 0")

        # 6 Card Stat Boxes Grid
        stat_grid = ttk.Frame(stats_group)
        stat_grid.pack(fill=tk.X)

        stat_cards = [
            ("THROUGHPUT", "fps", self.stat_vars["fps"], self.accent_emerald, 0, 0),
            ("FRAME NO.", "frame", self.stat_vars["frame"], self.accent_cyan, 0, 1),
            ("DETECTIONS", "detections", self.stat_vars["detections"], self.accent_purple, 0, 2),
            ("ACTIVE TRACKS", "tracks", self.stat_vars["tracks"], "#f472b6", 1, 0),
            ("OPTICAL FLOW", "flow_points", self.stat_vars["flow_points"], self.accent_amber, 1, 1),
            ("MOVING / STAT", "moving_stat", self.stat_moving_stat_var, "#2dd4bf", 1, 2),
        ]

        for title, key, var, color, r, c in stat_cards:
            box = tk.Frame(stat_grid, bg="#090d16", bd=1, relief=tk.SOLID, highlightbackground="#334155", highlightthickness=1, padx=8, pady=5)
            box.grid(row=r, column=c, padx=3, pady=3, sticky="nsew")

            lbl_title = tk.Label(box, text=title, bg="#090d16", fg=self.fg_muted, font=("Segoe UI", 8, "bold"))
            lbl_title.pack(anchor=tk.W)

            lbl_val = tk.Label(box, textvariable=var, bg="#090d16", fg=color, font=("Consolas", 11, "bold"))
            lbl_val.pack(anchor=tk.E, pady=(2, 0))

        for c in range(3):
            stat_grid.columnconfigure(c, weight=1)

        # ------------------------------------------------------------------ #
        # 3. TRACKED OBJECT KINEMATICS TABLE                                 #
        # ------------------------------------------------------------------ #
        table_group = ttk.LabelFrame(right_frame, text=" 3. Tracked Object Kinematics ", padding=8)
        table_group.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        columns = ("id", "direction", "disp", "velocity", "length")
        self.objects_tree = ttk.Treeview(table_group, columns=columns, show="headings", height=5)

        self.objects_tree.heading("id", text="ID")
        self.objects_tree.heading("direction", text="Heading Direction")
        self.objects_tree.heading("disp", text="Disp (px)")
        self.objects_tree.heading("velocity", text="Vel (px/s)")
        self.objects_tree.heading("length", text="Trail")

        self.objects_tree.column("id", width=45, anchor="center")
        self.objects_tree.column("direction", width=120, anchor="center")
        self.objects_tree.column("disp", width=80, anchor="e")
        self.objects_tree.column("velocity", width=95, anchor="e")
        self.objects_tree.column("length", width=60, anchor="center")

        tree_scroll = ttk.Scrollbar(table_group, orient=tk.VERTICAL, command=self.objects_tree.yview)
        self.objects_tree.configure(yscrollcommand=tree_scroll.set)

        self.objects_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        # ------------------------------------------------------------------ #
        # 4. TUNABLE VISION HYPERPARAMETERS PANEL                            #
        # ------------------------------------------------------------------ #
        settings_group = ttk.LabelFrame(right_frame, text=" 4. Real-Time Vision Settings ", padding=8)
        settings_group.pack(fill=tk.X)

        self.setting_vars = {
            "min_area": tk.DoubleVar(value=float(self.config.detector.min_contour_area)),
            "max_dist": tk.DoubleVar(value=float(self.config.tracker.max_distance)),
            "max_disp": tk.IntVar(value=int(self.config.tracker.max_disappeared)),
            "flow_corners": tk.IntVar(value=int(self.config.optical_flow.max_corners)),
            "var_threshold": tk.DoubleVar(value=float(self.config.detector.var_threshold)),
        }

        settings_specs = [
            ("Min Object Area (px²):", "min_area", 50.0, 3000.0, "Filter out small noise blobs"),
            ("Tracker Match Dist (px):", "max_dist", 10.0, 150.0, "Max distance to associate object"),
            ("Max Disappeared (frames):", "max_disp", 2, 40, "Frames before removing lost track"),
            ("Optical Flow Corners:", "flow_corners", 20, 250, "Max Shi-Tomasi feature points"),
            ("MOG2 Var Threshold:", "var_threshold", 4.0, 64.0, "Background subtractor sensitivity"),
        ]

        for row_idx, (label_txt, var_key, v_min, v_max, desc_txt) in enumerate(settings_specs):
            row_frame = ttk.Frame(settings_group)
            row_frame.pack(fill=tk.X, pady=2)

            lbl_title = ttk.Label(row_frame, text=label_txt, font=("Segoe UI", 9, "bold"))
            lbl_title.pack(side=tk.LEFT)

            val_lbl = tk.Label(row_frame, bg=self.bg_card, fg=self.accent_cyan, font=("Consolas", 9, "bold"), width=6, anchor="e")
            val_lbl.pack(side=tk.RIGHT)

            scale = ttk.Scale(
                row_frame,
                from_=v_min,
                to=v_max,
                variable=self.setting_vars[var_key],
                orient=tk.HORIZONTAL,
                command=lambda val, k=var_key: self._on_setting_changed(k, val),
            )
            scale.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=8)

            # Link live value label
            if isinstance(self.setting_vars[var_key], tk.IntVar):
                val_lbl.configure(textvariable=self.setting_vars[var_key])
            else:
                def _update_float_lbl(*args, lbl=val_lbl, k=var_key):
                    lbl.configure(text=f"{self.setting_vars[k].get():.0f}")
                self.setting_vars[var_key].trace_add("write", _update_float_lbl)
                val_lbl.configure(text=f"{self.setting_vars[var_key].get():.0f}")

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
        self.status_text.set("Status: Webcam selected. Click '▶ START PROCESSING'.")

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
            self.status_text.set(f"Status: Loaded '{base_name}'. Click '▶ START PROCESSING'.")

    def start_processing(self) -> None:
        """Initialize pipeline subsystems and launch background processing thread."""
        if self.is_processing:
            return

        try:
            self.video_processor = VideoProcessor(source=self.current_source, config=self.config.video)
        except Exception as e:
            messagebox.showerror("Stream Error", f"Failed to open video source:\n{e}")
            self.status_text.set("Status: Error opening video source.")
            return

        self.stop_event.clear()
        self.is_processing = True
        self.is_paused = False

        # Update Button States & Badge
        self.btn_start.configure(state=tk.DISABLED)
        self.btn_stop.configure(state=tk.NORMAL)
        self.btn_pause.configure(state=tk.NORMAL)
        self.btn_resume.configure(state=tk.DISABLED)

        self.status_badge_var.set("● PROCESSING")
        self.status_badge_lbl.configure(fg="#10b981")
        self.status_text.set("Status: Processing pipeline active...")

        # Launch Worker Thread
        self.worker_thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        self.worker_thread.start()

    def stop_processing(self) -> None:
        """Signal worker thread to terminate and release resources."""
        if not self.is_processing:
            return

        self.stop_event.set()
        self.is_processing = False
        self.is_paused = False

        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

        if self.video_processor:
            self.video_processor.release()
            self.video_processor = None

        # Reset Controls & Badge
        self.btn_start.configure(state=tk.NORMAL)
        self.btn_stop.configure(state=tk.DISABLED)
        self.btn_pause.configure(state=tk.DISABLED)
        self.btn_resume.configure(state=tk.DISABLED)

        self.status_badge_var.set("⏹ STOPPED")
        self.status_badge_lbl.configure(fg="#f43f5e")
        self.status_text.set("Status: Pipeline stopped.")

    def pause_processing(self) -> None:
        """Pause frame ingestion while preserving tracking states."""
        if not self.is_processing or self.is_paused:
            return
        self.is_paused = True
        self.btn_pause.configure(state=tk.DISABLED)
        self.btn_resume.configure(state=tk.NORMAL)

        self.status_badge_var.set("⏸ PAUSED")
        self.status_badge_lbl.configure(fg="#fbbf24")
        self.status_text.set("Status: Processing PAUSED. Click 'Resume' or press 'P'.")

    def resume_processing(self) -> None:
        """Resume active frame ingestion."""
        if not self.is_processing or not self.is_paused:
            return
        self.is_paused = False
        self.btn_pause.configure(state=tk.NORMAL)
        self.btn_resume.configure(state=tk.DISABLED)

        self.status_badge_var.set("● PROCESSING")
        self.status_badge_lbl.configure(fg="#10b981")
        self.status_text.set("Status: Processing resumed.")

    def toggle_pause(self) -> None:
        """Toggle pause state via shortcut."""
        if self.is_paused:
            self.resume_processing()
        else:
            self.pause_processing()

    def reset_pipeline(self) -> None:
        """Reset internal states of tracker, motion analyzer, and optical flow."""
        self.tracker.reset()
        self.motion_analyzer.reset()
        self.optical_flow.reset()
        self.evaluator.reset()

        # Clear UI table and stats
        for item in self.objects_tree.get_children():
            self.objects_tree.delete(item)

        self.stat_vars["fps"].set("0.0 FPS")
        self.stat_vars["frame"].set("0")
        self.stat_vars["detections"].set("0")
        self.stat_vars["tracks"].set("0")
        self.stat_vars["flow_points"].set("0")
        self.stat_moving_stat_var.set("0 / 0")

        self.status_text.set("Status: Pipeline state reset by user.")

    def save_current_frame(self) -> Optional[str]:
        """Save the latest annotated frame to results/screenshots/."""
        if self.latest_annotated_frame is None:
            messagebox.showinfo("Save Screenshot", "No active frame available to save.")
            return None

        out_dir = os.path.abspath(os.path.join("results", "screenshots"))
        os.makedirs(out_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        filename = f"motiontrack_{timestamp}.png"
        filepath = os.path.join(out_dir, filename)

        cv2.imwrite(filepath, self.latest_annotated_frame)
        self.status_text.set(f"Status: Screenshot saved to 'results/screenshots/{filename}'.")
        return filepath

    # ---------------------------------------------------------------------- #
    # Worker Thread Processing Loop                                          #
    # ---------------------------------------------------------------------- #

    def _pipeline_worker(self) -> None:
        """Isolated background worker thread running the CV pipeline."""
        frame_idx = 0
        fps_smoothing_window = 10
        recent_latencies: List[float] = []

        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.05)
                continue

            t_start = time.perf_counter()

            if self.video_processor is None:
                break

            ret, frame = self.video_processor.read_frame()
            if not ret or frame is None:
                # End of stream reached
                self.root.after(0, lambda: self.status_text.set("Status: End of video stream reached."))
                self.root.after(0, self.stop_processing)
                break

            frame_idx += 1

            # 1. Preprocessing
            prep_frame = self.preprocessor.process(frame)

            # 2. Object Detection
            detections, fg_mask = self.detector.detect(prep_frame)

            # 3. Object Tracking
            tracked_objects = self.tracker.update(detections)

            # 4. Optical Flow
            optical_flow_points = self.optical_flow.update(prep_frame)

            # 5. Motion Analysis
            motion_data = self.motion_analyzer.update(
                tracked_objects=tracked_objects,
                optical_flow_points=optical_flow_points,
            )

            # Record metrics
            self.evaluator.record_frame(
                detections=detections,
                tracked_objects=tracked_objects,
                optical_flow_points=optical_flow_points,
                motion_data=motion_data,
            )

            # Calculate instantaneous smoothed FPS
            t_elapsed = time.perf_counter() - t_start
            recent_latencies.append(t_elapsed)
            if len(recent_latencies) > fps_smoothing_window:
                recent_latencies.pop(0)
            avg_latency = float(np.mean(recent_latencies)) if recent_latencies else 0.033
            current_fps = 1.0 / max(0.0001, avg_latency)

            # 6. Visualization Overlay
            stats = self.motion_analyzer.get_summary_statistics()
            annotated_frame = self.visualizer.render(
                frame=frame,
                detections=detections,
                tracked_objects=tracked_objects,
                motion_data=motion_data,
                optical_flow_points=optical_flow_points,
                active_count=len(tracked_objects),
                fps=current_fps,
                stats=stats,
            )

            self.latest_annotated_frame = annotated_frame.copy()

            # Format Object Kinematics table rows
            object_rows = []
            for track in tracked_objects:
                m_info = motion_data.get(track.object_id)
                direction_str = m_info.direction.value if m_info else "STATIONARY"
                disp_val = f"{m_info.displacement:.1f}" if m_info else "0.0"
                vel_val = f"{m_info.velocity_sec:.1f}" if m_info else "0.0"
                trail_len = len(track.trajectory)
                object_rows.append((track.object_id, direction_str, disp_val, vel_val, trail_len))

            payload = {
                "annotated_frame": annotated_frame,
                "frame_idx": frame_idx,
                "fps": current_fps,
                "detections_count": len(detections),
                "tracks_count": len(tracked_objects),
                "flow_count": len(optical_flow_points),
                "moving_count": stats.get("moving_count", 0),
                "stationary_count": stats.get("stationary_count", 0),
                "object_rows": object_rows,
            }

            # Offer payload to UI main-thread polling queue (drop stale frame if full)
            try:
                self.frame_queue.put_nowait(payload)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(payload)
                except queue.Empty:
                    pass

            # Target ~30 FPS UI display update rate
            time.sleep(0.005)

    # ---------------------------------------------------------------------- #
    # Main-Thread UI Queue Consumer                                         #
    # ---------------------------------------------------------------------- #

    def _poll_queue(self) -> None:
        """Periodic main-thread polling loop to consume frames and update widgets."""
        try:
            payload = None
            while not self.frame_queue.empty():
                payload = self.frame_queue.get_nowait()

            if payload is not None:
                # Update Video Canvas
                bgr_frame = payload["annotated_frame"]
                self._display_frame(bgr_frame)

                # Update Telemetry Labels
                self.stat_vars["frame"].set(str(payload["frame_idx"]))
                self.stat_vars["fps"].set(f"{payload['fps']:.1f} FPS")
                self.stat_vars["detections"].set(str(payload["detections_count"]))
                self.stat_vars["tracks"].set(str(payload["tracks_count"]))
                self.stat_vars["flow_points"].set(str(payload["flow_count"]))
                self.stat_moving_stat_var.set(f"{payload['moving_count']} / {payload['stationary_count']}")

                # Update Object Kinematics Table
                self._update_treeview(payload["object_rows"])

        except Exception as e:
            print(f"[DEBUG GUI Poll Error] {e}", file=sys.stderr)

        # Re-arm polling loop
        self.root.after(20, self._poll_queue)

    def _display_frame(self, bgr_image: np.ndarray) -> None:
        """Resize frame maintaining aspect ratio and render onto video canvas."""
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
        existing_items = self.objects_tree.get_children()
        for item in existing_items:
            self.objects_tree.delete(item)

        for row in object_rows:
            self.objects_tree.insert("", tk.END, values=row)

    def on_closing(self) -> None:
        """Graceful application shutdown."""
        self.stop_processing()
        self.root.destroy()
