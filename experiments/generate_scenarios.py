"""
Deterministic synthetic scenario generator for CV-MotionTrack evaluation (Step 9).

Generates 6 reproducible video sequences in data/input/:
1. scenario_1_single_object.avi: Single moving object across frame.
2. scenario_2_multi_object.avi: Two objects moving on independent paths.
3. scenario_3_slow_motion.avi: Slow-moving object (displacement ~ 0.8 px/frame).
4. scenario_4_fast_motion.avi: Fast-moving object (displacement ~ 18 px/frame).
5. scenario_5_reentry.avi: Object moves across, exits scene, stays absent, then re-enters.
6. scenario_6_background_variation.avi: Moving object with illumination fluctuation and shadow.
"""

import os
import cv2
import numpy as np


def ensure_input_dir() -> str:
    """Ensure data/input directory exists."""
    input_dir = os.path.join("data", "input")
    os.makedirs(input_dir, exist_ok=True)
    return input_dir


def create_scenario_1_single_object(input_dir: str, width: int = 320, height: int = 240, fps: float = 20.0) -> str:
    """Scenario 1: Single object moving steadily from top-left to bottom-right."""
    path = os.path.join(input_dir, "scenario_1_single_object.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    # Background canvas
    bg = np.full((height, width, 3), 35, dtype=np.uint8)
    # Add static background landmarks for optical flow
    cv2.circle(bg, (40, 200), 10, (70, 70, 70), -1)
    cv2.circle(bg, (280, 40), 12, (70, 70, 70), -1)

    total_frames = 45
    # 10 frames background warmup + 35 frames movement
    for i in range(total_frames):
        frame = bg.copy()
        if i >= 8:
            step = i - 8
            x = int(30 + step * 6.5)
            y = int(30 + step * 4.5)
            cv2.rectangle(frame, (x, y), (x + 35, y + 35), (210, 210, 210), -1)
            # Add interior texture to moving object for feature tracking
            cv2.circle(frame, (x + 10, y + 10), 3, (40, 40, 40), -1)
            cv2.circle(frame, (x + 25, y + 25), 3, (40, 40, 40), -1)
        writer.write(frame)

    writer.release()
    return path


def create_scenario_2_multi_object(input_dir: str, width: int = 320, height: int = 240, fps: float = 20.0) -> str:
    """Scenario 2: Multiple objects moving simultaneously on distinct trajectories."""
    path = os.path.join(input_dir, "scenario_2_multi_object.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    bg = np.full((height, width, 3), 35, dtype=np.uint8)
    cv2.rectangle(bg, (140, 100), (180, 140), (60, 60, 60), 2)

    total_frames = 50
    for i in range(total_frames):
        frame = bg.copy()
        if i >= 8:
            step = i - 8
            # Object 1: moves left to right
            x1 = int(20 + step * 5.5)
            y1 = int(40 + step * 2.0)
            cv2.rectangle(frame, (x1, y1), (x1 + 30, y1 + 30), (220, 200, 180), -1)
            cv2.circle(frame, (x1 + 15, y1 + 15), 3, (30, 30, 30), -1)

            # Object 2: moves right to left along bottom
            x2 = int(270 - step * 5.0)
            y2 = int(170 - step * 1.5)
            cv2.rectangle(frame, (x2, y2), (x2 + 32, y2 + 32), (180, 220, 200), -1)
            cv2.circle(frame, (x2 + 16, y2 + 16), 3, (30, 30, 30), -1)

        writer.write(frame)

    writer.release()
    return path


def create_scenario_3_slow_motion(input_dir: str, width: int = 320, height: int = 240, fps: float = 20.0) -> str:
    """Scenario 3: Slow-moving object with tiny inter-frame displacement (0.8 px/frame)."""
    path = os.path.join(input_dir, "scenario_3_slow_motion.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    bg = np.full((height, width, 3), 40, dtype=np.uint8)
    cv2.circle(bg, (100, 50), 15, (75, 75, 75), -1)

    total_frames = 50
    for i in range(total_frames):
        frame = bg.copy()
        if i >= 8:
            step = i - 8
            # Displacement ~ 0.8 px per frame horizontally
            x = int(60 + step * 0.8)
            y = int(100 + step * 0.4)
            cv2.rectangle(frame, (x, y), (x + 36, y + 36), (230, 230, 230), -1)
            cv2.circle(frame, (x + 18, y + 18), 4, (30, 30, 30), -1)
        writer.write(frame)

    writer.release()
    return path


def create_scenario_4_fast_motion(input_dir: str, width: int = 320, height: int = 240, fps: float = 20.0) -> str:
    """Scenario 4: Fast-moving object with large displacement (~18 px/frame)."""
    path = os.path.join(input_dir, "scenario_4_fast_motion.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    bg = np.full((height, width, 3), 35, dtype=np.uint8)

    total_frames = 35
    for i in range(total_frames):
        frame = bg.copy()
        if i >= 8:
            step = i - 8
            # Large displacement: 18 px/frame
            x = int(10 + step * 18.0)
            y = int(30 + step * 6.0)
            if x + 40 < width:
                cv2.rectangle(frame, (x, y), (x + 40, y + 40), (240, 240, 240), -1)
                cv2.circle(frame, (x + 12, y + 12), 4, (20, 20, 20), -1)
                cv2.circle(frame, (x + 28, y + 28), 4, (20, 20, 20), -1)
        writer.write(frame)

    writer.release()
    return path


def create_scenario_5_reentry(input_dir: str, width: int = 320, height: int = 240, fps: float = 20.0) -> str:
    """Scenario 5: Object moves, exits the field of view, stays absent, then re-enters."""
    path = os.path.join(input_dir, "scenario_5_reentry.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    bg = np.full((height, width, 3), 35, dtype=np.uint8)

    total_frames = 65
    for i in range(total_frames):
        frame = bg.copy()
        # Phase 1: Frame 8 to 28 -> Object enters from left and moves past right boundary (exits)
        if 8 <= i < 28:
            step = i - 8
            x = int(10 + step * 16.0)
            y = 80
            cv2.rectangle(frame, (x, y), (x + 35, y + 35), (210, 210, 210), -1)
            cv2.circle(frame, (x + 17, y + 17), 3, (30, 30, 30), -1)

        # Phase 2: Frame 28 to 40 -> Scene is empty (object deregistered by tracker)

        # Phase 3: Frame 40 to 65 -> Object re-enters from top-left with a new track ID
        elif i >= 40:
            step = i - 40
            x = int(20 + step * 10.0)
            y = int(30 + step * 6.0)
            cv2.rectangle(frame, (x, y), (x + 35, y + 35), (210, 210, 210), -1)
            cv2.circle(frame, (x + 17, y + 17), 3, (30, 30, 30), -1)

        writer.write(frame)

    writer.release()
    return path


def create_scenario_6_background_variation(input_dir: str, width: int = 320, height: int = 240, fps: float = 20.0) -> str:
    """Scenario 6: Moving object with gradual illumination shifts and shadow casting."""
    path = os.path.join(input_dir, "scenario_6_background_variation.avi")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))

    total_frames = 50
    for i in range(total_frames):
        # Background brightness fluctuates slightly across time (illumination drift)
        base_intensity = int(35 + 15 * np.sin(i / 6.0))
        frame = np.full((height, width, 3), base_intensity, dtype=np.uint8)

        # Static landmark
        cv2.circle(frame, (60, 180), 15, (base_intensity + 30, base_intensity + 30, base_intensity + 30), -1)

        if i >= 8:
            step = i - 8
            x = int(25 + step * 6.0)
            y = int(60 + step * 2.5)

            # Simulated shadow region (semi-darkened area beneath object)
            shadow_x1 = max(0, x - 5)
            shadow_y1 = min(height - 1, y + 35)
            shadow_x2 = min(width - 1, x + 40)
            shadow_y2 = min(height - 1, y + 48)
            frame[shadow_y1:shadow_y2, shadow_x1:shadow_x2] = np.clip(
                frame[shadow_y1:shadow_y2, shadow_x1:shadow_x2] * 0.55, 0, 255
            ).astype(np.uint8)

            # Foreground moving object
            cv2.rectangle(frame, (x, y), (x + 35, y + 35), (225, 225, 225), -1)
            cv2.circle(frame, (x + 17, y + 17), 4, (30, 30, 30), -1)

        writer.write(frame)

    writer.release()
    return path


def generate_all_scenarios() -> dict:
    """Generate all 6 benchmark scenarios and return dictionary of paths."""
    input_dir = ensure_input_dir()
    print("=" * 65)
    print("Generating Deterministic CV-MotionTrack Synthetic Benchmark Scenarios")
    print("=" * 65)

    scenarios = {
        "scenario_1_single_object": create_scenario_1_single_object(input_dir),
        "scenario_2_multi_object": create_scenario_2_multi_object(input_dir),
        "scenario_3_slow_motion": create_scenario_3_slow_motion(input_dir),
        "scenario_4_fast_motion": create_scenario_4_fast_motion(input_dir),
        "scenario_5_reentry": create_scenario_5_reentry(input_dir),
        "scenario_6_background_variation": create_scenario_6_background_variation(input_dir),
    }

    for name, path in scenarios.items():
        print(f"[CREATED] {name:32s} -> {path}")
    print("=" * 65)
    return scenarios


if __name__ == "__main__":
    generate_all_scenarios()
