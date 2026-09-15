"""
Unit tests for the Command-Line Interface module (src/cli.py).

Academic Coursework: CSE3010 Computer Vision - Step 11
Verifies:
1. Argument parsing for --source webcam, --source video --input <path>, --min-area, --output, --headless.
2. Handling of missing video files with clean exit code without raw exception crashes.
3. Execution of CLI runner on valid synthetic video stream.
"""

import os
import sys
import pytest

from src.cli import parse_cli_args, run_cli_pipeline


def test_cli_argument_parsing():
    """Verify parse_cli_args properly maps flags to Namespace attributes."""
    args = parse_cli_args(["--source", "video", "--input", "data/input/synthetic_demo.avi", "--min-area", "250", "--headless"])
    assert args.source == "video"
    assert args.input == "data/input/synthetic_demo.avi"
    assert args.min_area == 250.0
    assert args.headless is True

    # Default source determination when --input is provided
    args_implicit = parse_cli_args(["--input", "data/input/synthetic_demo.avi"])
    assert args_implicit.source == "video"


def test_cli_invalid_input_file():
    """Verify clean error code 1 returned for non-existent video files."""
    args = parse_cli_args(["--source", "video", "--input", "data/input/nonexistent_file_123.mp4"])
    ret_code = run_cli_pipeline(args)
    assert ret_code == 1


def test_cli_execution_synthetic_video():
    """Verify end-to-end CLI execution on synthetic video file."""
    video_path = os.path.abspath(os.path.join("data", "input", "synthetic_demo.avi"))
    assert os.path.exists(video_path)

    args = parse_cli_args([
        "--source", "video",
        "--input", video_path,
        "--headless",
        "--max-frames", "5",
        "--save-results"
    ])
    ret_code = run_cli_pipeline(args)
    assert ret_code == 0
