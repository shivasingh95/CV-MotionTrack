"""
Root pytest configuration for CV-MotionTrack test suite.

Sets the Tcl/Tk library environment variables before any tkinter import
to resolve the Windows TclError caused by missing ttk theme file path
lookups on some Python 3.10 installations.

Provides the shared ``tk_app`` fixture used by test_ui.py.
"""

import os
import sys
import pytest

# ---------------------------------------------------------------------------
# Tcl/Tk path fix (must happen before any tkinter import).
# ---------------------------------------------------------------------------
_PYTHON_HOME = os.path.dirname(sys.executable)
_TCL_LIB = os.path.join(_PYTHON_HOME, "tcl", "tcl8.6")
_TK_LIB = os.path.join(_PYTHON_HOME, "tcl", "tk8.6")

if os.path.isdir(_TCL_LIB):
    os.environ.setdefault("TCL_LIBRARY", _TCL_LIB)
if os.path.isdir(_TK_LIB):
    os.environ.setdefault("TK_LIBRARY", _TK_LIB)


# ---------------------------------------------------------------------------
# Session-scoped Tkinter root (one Tk() per test session avoids TclError
# that occurs when a second Tk() is instantiated after one is destroyed).
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tk_root():
    """
    Single Tkinter root window shared across the entire test session.

    Using a session scope prevents the TclError that occurs when multiple
    Tk() instances are created sequentially in the same process.
    """
    import tkinter as tk

    root = tk.Tk()
    root.withdraw()          # Hidden -- never shown during tests
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def tk_app(tk_root):
    """
    Provide a fresh CVMotionTrackApp backed by the shared Tk root.

    Each test gets a newly constructed app instance while reusing the
    same underlying Tk interpreter, avoiding repeated Tk() creation.
    """
    from src.ui import CVMotionTrackApp

    # Wipe any widgets the previous test may have left on the root
    for child in tk_root.winfo_children():
        try:
            child.destroy()
        except Exception:
            pass

    app = CVMotionTrackApp(tk_root)
    yield app

    try:
        app.stop_processing()
    except Exception:
        pass
