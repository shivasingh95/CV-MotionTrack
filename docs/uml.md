# UML Diagrams — CV-MotionTrack

**Academic Coursework: CSE3010 Computer Vision**

This document contains three UML diagrams for the CV-MotionTrack system:
1. Use Case Diagram — system interactions from the user's perspective
2. Class Diagram — the actual Python class structure and relationships
3. Sequence Diagram — the runtime processing loop interaction between components

All diagrams reflect the **actual** implementation. No speculative features are included.

---

## A. Use Case Diagram

The single external actor is the **User**, who interacts with the system through the Tkinter
graphical user interface. The processing pipeline and its CV modules are internal actors.

```mermaid
flowchart TD
    User(["👤 User"])

    subgraph CV-MotionTrack System
        UC1["Start Application"]
        UC2["Select Video Source\n(Webcam or File)"]
        UC3["Start Processing"]
        UC4["Pause Processing"]
        UC5["Resume Processing"]
        UC6["Reset Pipeline"]
        UC7["View Detected Objects\n(Bounding Boxes + IDs)"]
        UC8["View Tracking Information\n(Trajectories + Kinematics Table)"]
        UC9["View Motion Statistics\n(FPS, Flow Points, Velocity)"]
        UC10["Save Annotated Frame\n(PNG Screenshot)"]
        UC11["Stop Processing"]
    end

    User --> UC1
    User --> UC2
    User --> UC3
    User --> UC4
    User --> UC5
    User --> UC6
    User --> UC7
    User --> UC8
    User --> UC9
    User --> UC10
    User --> UC11

    UC3 --> UC7
    UC3 --> UC8
    UC3 --> UC9
    UC4 -.->|"requires"| UC3
    UC5 -.->|"requires"| UC4
    UC10 -.->|"requires"| UC3
    UC11 --> UC3
```

---

## B. Class Diagram

The class diagram shows the actual Python classes, their key attributes and methods, and
their relationships. Only attributes and methods that exist in the source code are shown.

### Data Models (`src/data_models.py`)

```mermaid
classDiagram
    class BoundingBox {
        +int x
        +int y
        +int width
        +int height
        +top_left() Tuple
        +bottom_right() Tuple
        +center() Tuple
        +area() int
        +as_tuple() Tuple
    }

    class Detection {
        +BoundingBox bbox
        +Tuple centroid
        +float confidence
        +float area
    }

    class TrackedObject {
        +int object_id
        +Tuple current_position
        +Tuple previous_position
        +List trajectory
        +BoundingBox bbox
        +int disappeared_count
    }

    class OpticalFlowPoint {
        +float previous_x
        +float previous_y
        +float current_x
        +float current_y
        +float dx
        +float dy
        +previous_point() Tuple
        +current_point() Tuple
        +vector() Tuple
        +magnitude() float
    }

    class MotionData {
        +int object_id
        +float dx
        +float dy
        +float displacement
        +str direction
        +float angle
        +float velocity
        +float path_length
        +float net_displacement
        +float approximate_velocity
        +float instantaneous_speed
    }

    Detection --> BoundingBox
    TrackedObject --> BoundingBox
```

### Configuration (`src/config.py`)

```mermaid
classDiagram
    class AppConfig {
        +VideoConfig video
        +PreprocessingConfig preprocessing
        +DetectorConfig detector
        +TrackerConfig tracker
        +OpticalFlowConfig optical_flow
        +MotionAnalysisConfig motion
        +VisualizationConfig visualization
        +OutputConfig output
    }

    class VideoConfig {
        +source Union[int, str]
        +target_width int
        +target_height int
        +enforce_target_size bool
        +fps_limit float
    }

    class DetectorConfig {
        +subtractor_type str
        +history int
        +var_threshold float
        +detect_shadows bool
        +min_contour_area float
        +max_contour_area float
        +morph_kernel_size Tuple
    }

    class TrackerConfig {
        +max_distance float
        +max_disappeared int
        +max_trajectory_length int
    }

    class OpticalFlowConfig {
        +win_size Tuple
        +max_level int
        +max_corners int
        +quality_level float
        +min_distance float
        +min_features int
    }

    class MotionAnalysisConfig {
        +fps float
        +smoothing_window int
        +stationary_threshold float
        +history_length int
    }

    AppConfig --> VideoConfig
    AppConfig --> DetectorConfig
    AppConfig --> TrackerConfig
    AppConfig --> OpticalFlowConfig
    AppConfig --> MotionAnalysisConfig
```

### Processing Modules

```mermaid
classDiagram
    class VideoProcessor {
        +VideoConfig config
        +source Union[int, str]
        +capture cv2.VideoCapture
        +frame_counter int
        +open() bool
        +is_opened() bool
        +read_frame() Tuple
        +get_properties() Dict
        +release() None
        +width() int
        +height() int
        +fps() float
    }

    class Preprocessor {
        +PreprocessingConfig config
        +gaussian_kernel_size Tuple
        +gaussian_sigma float
        +to_grayscale(frame) ndarray
        +apply_gaussian_blur(frame) ndarray
        +process(frame) ndarray
    }

    class ObjectDetector {
        +DetectorConfig config
        +history int
        +var_threshold float
        +min_contour_area float
        +max_contour_area float
        +subtractor BackgroundSubtractor
        +get_foreground_mask(frame) ndarray
        +clean_mask(mask) ndarray
        +detect_objects(mask) List
        +detect(frame) Tuple
        +reset() None
    }

    class ObjectTracker {
        +TrackerConfig config
        +max_distance float
        +max_disappeared int
        +objects Dict
        +disappeared Dict
        +next_object_id int
        +active_count int
        +update(detections) List
        +register(detection) None
        +deregister(object_id) None
        +reset() None
    }

    class OpticalFlowAnalyzer {
        +OpticalFlowConfig config
        +max_corners int
        +win_size Tuple
        +max_level int
        +prev_gray ndarray
        +prev_points ndarray
        +reinit_counter int
        +update(gray_frame) List
        +detect_features(gray_frame) ndarray
        +reset() None
    }

    class MotionAnalyzer {
        +MotionAnalysisConfig config
        +fps float
        +stationary_threshold float
        +smoothing_window int
        +position_history Dict
        +speed_history Dict
        +update(tracked_objects, flow_points) Dict
        +get_summary_statistics() Dict
        +reset() None
    }

    class Visualizer {
        +VisualizationConfig config
        +draw_detections(frame, detections) ndarray
        +draw_tracked_objects(frame, tracked_objects, motion_data) ndarray
        +draw_trajectories(frame, tracked_objects) ndarray
        +draw_optical_flow(frame, flow_points) ndarray
        +draw_statistics(frame, active_count, fps, stats) ndarray
        +render(frame, detections, tracked_objects, motion_data, flow_points, ...) ndarray
    }

    class Evaluator {
        +EvaluationSummary summary
        +frame_metrics List
        +start_frame() None
        +end_frame(detections, tracked_objects, flow_points, motion_data) None
        +export_csv(path) None
        +export_json(path) None
        +print_summary() None
        +reset() None
    }

    VideoProcessor --> Preprocessor : provides frames
    Preprocessor --> ObjectDetector : grayscale frame
    ObjectDetector --> ObjectTracker : List~Detection~
    ObjectTracker --> MotionAnalyzer : List~TrackedObject~
    OpticalFlowAnalyzer --> MotionAnalyzer : List~OpticalFlowPoint~
    MotionAnalyzer --> Visualizer : Dict~MotionData~
    ObjectDetector --> Visualizer : List~Detection~
    ObjectTracker --> Visualizer : List~TrackedObject~
    OpticalFlowAnalyzer --> Visualizer : List~OpticalFlowPoint~
    Visualizer --> CVMotionTrackApp : annotated ndarray
    Evaluator --> CVMotionTrackApp : EvaluationSummary
```

### UI Controller

```mermaid
classDiagram
    class CVMotionTrackApp {
        +tk.Tk root
        +AppConfig config
        +VideoProcessor video_processor
        +Preprocessor preprocessor
        +ObjectDetector detector
        +ObjectTracker tracker
        +OpticalFlowAnalyzer optical_flow
        +MotionAnalyzer motion_analyzer
        +Visualizer visualizer
        +Evaluator evaluator
        +bool is_processing
        +bool is_paused
        +int current_source
        +str source_type
        +ndarray latest_annotated_frame
        +Queue frame_queue
        +start_webcam() None
        +open_video_dialog() None
        +start_processing() None
        +pause_processing() None
        +resume_processing() None
        +reset_pipeline() None
        +stop_processing() None
        +save_current_frame() Optional[str]
        +_processing_worker() None
        +_poll_queue() None
        +_display_frame(frame) None
        +_update_treeview(rows) None
        +_on_setting_changed(key, val) None
        +on_closing() None
    }

    CVMotionTrackApp --> VideoProcessor : owns
    CVMotionTrackApp --> Preprocessor : owns
    CVMotionTrackApp --> ObjectDetector : owns
    CVMotionTrackApp --> ObjectTracker : owns
    CVMotionTrackApp --> OpticalFlowAnalyzer : owns
    CVMotionTrackApp --> MotionAnalyzer : owns
    CVMotionTrackApp --> Visualizer : owns
    CVMotionTrackApp --> Evaluator : owns
    CVMotionTrackApp --> AppConfig : configures
```

---

## C. Sequence Diagram

The sequence diagram shows the runtime interaction between components for a single
processed frame. The processing loop runs in a background thread (`_processing_worker`)
to keep the Tkinter UI responsive.

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant UI as CVMotionTrackApp<br/>(Tkinter Thread)
    participant Worker as _processing_worker<br/>(Background Thread)
    participant VP as VideoProcessor
    participant PP as Preprocessor
    participant DET as ObjectDetector
    participant TRK as ObjectTracker
    participant OF as OpticalFlowAnalyzer
    participant MA as MotionAnalyzer
    participant VIS as Visualizer
    participant Q as frame_queue

    User->>UI: Click "Start Processing"
    UI->>Worker: start Thread (stop_event cleared)
    UI->>VP: open(source)
    VP-->>UI: True (stream opened)

    loop Every Frame
        Worker->>VP: read_frame()
        VP-->>Worker: (True, bgr_frame)

        Worker->>PP: process(bgr_frame)
        PP-->>Worker: gray_frame

        Worker->>DET: detect(gray_frame)
        DET-->>Worker: (List[Detection], fg_mask)

        Worker->>TRK: update(List[Detection])
        TRK-->>Worker: List[TrackedObject]

        Worker->>OF: update(gray_frame)
        OF-->>Worker: List[OpticalFlowPoint]

        Worker->>MA: update(tracked_objects, flow_points)
        MA-->>Worker: Dict[int, MotionData]

        Worker->>VIS: render(frame, detections, tracked_objects, motion_data, flow_points, ...)
        VIS-->>Worker: annotated_frame (BGR ndarray)

        Worker->>Q: put(payload dict)
        Q-->>UI: payload (via _poll_queue every 25ms)

        UI->>UI: _display_frame(annotated_frame)
        UI->>UI: update telemetry labels
        UI->>UI: _update_treeview(object_rows)
    end

    User->>UI: Click "Pause"
    UI->>Worker: is_paused = True
    Note over Worker: sleep(0.03) — no processing

    User->>UI: Click "Resume"
    UI->>Worker: is_paused = False
    Note over Worker: Processing loop resumes

    User->>UI: Press "S" / Click "Save Frame"
    UI->>UI: save_current_frame()
    UI-->>User: PNG saved to results/screenshots/

    User->>UI: Click "Stop" / Press Q
    UI->>Worker: stop_event.set()
    Worker->>VP: (stream exits loop)
    UI->>VP: release()
    UI-->>User: Status bar updated
```

---

## Design Patterns Observed

| Pattern | Where Used |
|---|---|
| **Strategy** | `ObjectDetector` accepts `subtractor_type` ("MOG2"/"KNN") at construction |
| **Observer (polling)** | `_poll_queue` + `frame_queue` decouples worker thread from GUI thread |
| **Facade** | `CVMotionTrackApp` coordinates 8 CV modules behind a single interface |
| **Data Transfer Object** | `Detection`, `TrackedObject`, `MotionData`, `OpticalFlowPoint` dataclasses |
| **Configuration Object** | `AppConfig` aggregates all subsystem configs centrally |
| **Context Manager** | `VideoProcessor` implements `__enter__`/`__exit__` for resource safety |
