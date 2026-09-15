"""
Main entry point for CV-MotionTrack.

Academic Computer Vision project for CSE3010.
Delegates command-line execution to src/cli.py or launches the desktop GUI
when requested (--gui or --ui).
"""

import sys
from src.cli import parse_cli_args, run_cli_pipeline


def launch_gui() -> None:
    """Launch the optional Tkinter desktop GUI application."""
    import tkinter as tk
    from src.ui import CVMotionTrackApp

    root = tk.Tk()
    app = CVMotionTrackApp(root)
    root.mainloop()


def main() -> int:
    """
    Main application entry point.

    Delegates argument parsing to src/cli.py and triggers CLI pipeline
    or desktop GUI accordingly.
    """
    # Check if user requested GUI mode via --gui or --ui flag
    if "--gui" in sys.argv or "--ui" in sys.argv:
        launch_gui()
        return 0

    args = parse_cli_args()

    if args.gui:
        launch_gui()
        return 0

    return run_cli_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())
