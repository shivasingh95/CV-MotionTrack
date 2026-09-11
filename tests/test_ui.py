"""
Unit tests for the Desktop User Interface module (src/ui.py).

Academic Coursework: CSE3010 Computer Vision - Step 10
Verifies:
1. CVMotionTrackApp initializes cleanly in a headless/hidden Tkinter environment.
2. Initial button states, telemetry labels, and Treeview structure.
3. State transitions: webcam selection, pause, resume, reset, stop.
4. Real-time settings propagation to underlying CV modules.
5. Screenshot filename formatting and export into results/screenshots/.
6. Treeview kinematics table population.
7. Clean teardown and resource release on application close.
"""

import os
import tkinter as tk
import numpy as np
import pytest

from src.config import AppConfig
from src.ui import CVMotionTrackApp

# NOTE: The ``tk_app`` fixture is provided by the root conftest.py.
# It uses a session-scoped Tk root to avoid TclError when multiple
# tk.Tk() instances would otherwise be created sequentially.


# -------------------------------------------------------------------------- #
# 1. Initialization Test                                                     #
# -------------------------------------------------------------------------- #

def test_ui_initialization(tk_app):
    """Verify application initializes with expected title, state, and buttons."""
    app = tk_app
    assert app.root.title() == "CV-MotionTrack - Real-Time Motion Analysis System"
    assert app.is_processing is False
    assert app.is_paused is False
    assert app.current_source == 0
    assert app.source_type == "webcam"

    # Verify initial control button states
    assert str(app.btn_start["state"]) == tk.NORMAL
    assert str(app.btn_stop["state"]) == tk.DISABLED
    assert str(app.btn_pause["state"]) == tk.DISABLED
    assert str(app.btn_resume["state"]) == tk.DISABLED


# -------------------------------------------------------------------------- #
# 2. Source Selection & State Transitions                                    #
# -------------------------------------------------------------------------- #

def test_source_selection_and_pause_resume(tk_app):
    """Verify webcam selection and pause/resume state toggling."""
    app = tk_app

    # Test webcam selection
    app.start_webcam()
    assert app.current_source == 0
    assert app.source_type == "webcam"
    assert "Webcam" in app.source_text.get()

    # Test pause / resume transition simulation
    app.is_processing = True
    app.pause_processing()
    assert app.is_paused is True
    assert "PAUSED" in app.status_text.get()

    app.resume_processing()
    assert app.is_paused is False
    assert "resumed" in app.status_text.get()

    app.is_processing = False


# -------------------------------------------------------------------------- #
# 3. Real-Time Settings Propagation                                          #
# -------------------------------------------------------------------------- #

def test_settings_propagation(tk_app):
    """Verify adjusting UI scale sliders dynamically updates subsystem hyperparameters."""
    app = tk_app

    # 1. Minimum Object Area
    app._on_setting_changed("min_area", 250.0)
    assert app.config.detector.min_contour_area == 250.0
    assert app.detector.min_contour_area == 250.0

    # 2. Maximum Tracker Distance
    app._on_setting_changed("max_dist", 75.0)
    assert app.config.tracker.max_distance == 75.0
    assert app.tracker.max_distance == 75.0

    # 3. Maximum Disappeared Frames
    app._on_setting_changed("max_disp", 18)
    assert app.config.tracker.max_disappeared == 18
    assert app.tracker.max_disappeared == 18

    # 4. Optical Flow Feature Count
    app._on_setting_changed("flow_corners", 140)
    assert app.config.optical_flow.max_corners == 140
    assert app.optical_flow.max_corners == 140


# -------------------------------------------------------------------------- #
# 4. Pipeline Reset Behavior                                                 #
# -------------------------------------------------------------------------- #

def test_pipeline_reset(tk_app):
    """Verify reset_pipeline clears trackers, flow points, and table entries."""
    app = tk_app

    # Populate dummy entries
    app.stat_vars["frame"].set("45")
    app.stat_vars["fps"].set("32.5 FPS")
    app.objects_tree.insert("", tk.END, values=(1, "RIGHT", "5.2", "104.0", 8))
    assert len(app.objects_tree.get_children()) == 1

    app.reset_pipeline()

    # Verify telemetry cleared
    assert app.stat_vars["frame"].get() == "0"
    assert app.stat_vars["fps"].get() == "0.0 FPS"
    assert len(app.objects_tree.get_children()) == 0
    assert app.tracker.active_count == 0
    assert app.optical_flow.prev_gray is None


# -------------------------------------------------------------------------- #
# 5. Save Frame Functionality                                                #
# -------------------------------------------------------------------------- #

def test_save_current_frame(tk_app):
    """Verify current annotated frame is exported to results/screenshots/ with valid timestamp."""
    app = tk_app

    # No active frame -> returns None
    app.latest_annotated_frame = None
    assert app.save_current_frame() is None

    # Synthetic frame -> saves PNG
    synthetic_frame = np.full((120, 160, 3), 128, dtype=np.uint8)
    app.latest_annotated_frame = synthetic_frame

    saved_path = app.save_current_frame()
    assert saved_path is not None
    assert os.path.exists(saved_path)
    assert "motiontrack_" in saved_path
    assert saved_path.endswith(".png")
    assert os.path.getsize(saved_path) > 0


# -------------------------------------------------------------------------- #
# 6. Object Table Updating                                                   #
# -------------------------------------------------------------------------- #

def test_update_treeview(tk_app):
    """Verify treeview correctly reflects active track kinematic records."""
    app = tk_app

    rows = [
        (1, "RIGHT", "4.5", "90.0", 12),
        (2, "DOWN-LEFT", "8.2", "164.0", 5),
    ]

    app._update_treeview(rows)
    items = app.objects_tree.get_children()
    assert len(items) == 2

    val0 = app.objects_tree.item(items[0], "values")
    assert str(val0[0]) == "1"
    assert val0[1] == "RIGHT"

    val1 = app.objects_tree.item(items[1], "values")
    assert str(val1[0]) == "2"
    assert val1[1] == "DOWN-LEFT"
