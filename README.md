

Terminal 1: 
docker exec -it skeleton_humble bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py

Terminal 2:
docker exec -it skeleton_humble bash
source /opt/ros/humble/setup.bash
ros2 run rqt_gui rqt_gui


# Skeleton Detection Workspace

Real-time multi-person skeleton detection and tracking on an NVIDIA DGX Spark using an Intel RealSense D456, RTMO-M, ROS 2 Humble, and optional BoT-SORT + OSNet ReID tracking.

The live perception path is intentionally kept inside **one ROS node / one Python process** so that the high-bandwidth RealSense RGB stream never needs to pass through DDS before inference or tracking.

## Overview

The workspace provides:

- RealSense D456 RGB capture through `pyrealsense2`
- RTMO-M multi-person 2D pose detection
- COCO-17 keypoints and skeleton connections
- Per-person bounding boxes and confidence scores
- Optional BoT-SORT tracking
- Optional OSNet ReID for appearance-based association
- Persistent `person_id` when tracking is enabled
- A custom robot-facing `SkeletonFrame` ROS message
- A separate human-facing visualization image topic
- Offline image testing and annotated image output
- Runtime performance statistics for capture, RTMO, tracking, and total pipeline time

Current model assets are baked into the Docker image:

```text
/opt/models/
├── rtmo/
│   ├── rtmo-m.py
│   └── rtmo-m.pth
└── reid/
    └── osnet_x0_25_msmt17.pt
```

---

## Data Flow

```text
Intel RealSense D456
        │
        │ pyrealsense2
        ▼
848×480 BGR NumPy frame
        │
        │ same Python process
        ▼
      RTMO-M
        │
        ├── bbox (xyxy)
        ├── person confidence
        └── COCO-17 keypoints
        │
        ▼
  BoT-SORT Tracking          optional
        │
        ├── motion / IoU
        └── OSNet ReID       optional
              │
              └── crops people directly from the SAME BGR frame
        │
        ▼
 persistent track_id
        │
        ▼
 SkeletonFrame ROS message
        │
        ├──► /skeleton_detection/frame
        │       robot-facing structured output
        │
        └──► visualization overlay
                │
                ▼
        /skeleton_detection/visualization_image
                human-facing image output
```

There is **no ROS image topic between RealSense, RTMO, and BoT-SORT** in live RealSense mode. DDS starts only at the final outputs.

---

## Package Structure

```text
skeleton_detection/
├── rtmo_node.py
├── realsense_capture.py
├── rtmo_inference.py
├── tracking.py
├── message_builder.py
├── pipeline_stats.py
├── visualization.py
├── coco_keypoints.py
└── image_publisher.py
```

Main responsibilities:

| File | Responsibility |
|---|---|
| `rtmo_node.py` | ROS parameters, publishers, threads, lifecycle, and high-level pipeline orchestration |
| `realsense_capture.py` | D456 capture and latest-frame-wins buffering |
| `rtmo_inference.py` | RTMO model loading, inference, and parsing |
| `tracking.py` | BoT-SORT + OSNet ReID tracking |
| `message_builder.py` | Converts internal detections into ROS messages |
| `pipeline_stats.py` | FPS and latency statistics |
| `visualization.py` | Bounding box, skeleton, ID, score, and visualization rendering |
| `coco_keypoints.py` | COCO-17 keypoint names and connections |
| `image_publisher.py` | Offline/local-image test input |

---

# Quick Start

## 1. Start the Docker container

From the DGX Spark host:

```bash
cd /home/aims/Desktop/samd/skeleton_detection

docker run -d --name skeleton_humble \
  --privileged \
  --network host \
  --ipc host \
  --gpus all \
  -v /home/aims/Desktop/samd/skeleton_detection:/ros2_ws/src/skeleton_detection \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  skeleton_humble_dev \
  sleep infinity
```

If the container already exists:

```bash
docker start skeleton_humble
```

Enter the container:

```bash
docker exec -it skeleton_humble bash
```

---

## 2. Build the ROS package

Inside the container:

```bash
source /opt/ros/humble/setup.bash

cd /ros2_ws

colcon build --packages-select skeleton_detection

source /ros2_ws/install/setup.bash
```

After changing Python or ROS source files, rebuild with:

```bash
cd /ros2_ws
colcon build --packages-select skeleton_detection
source /ros2_ws/install/setup.bash
```

---

# Running the Live RealSense Pipeline

All commands below are run **inside the container** after:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
```

## Basic RTMO only

Tracking disabled:

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py
```

Equivalent:

```bash
ros2 run skeleton_detection rtmo_node --ros-args \
  -p input_mode:=realsense
```

In this mode:

```text
RealSense → RTMO → SkeletonFrame
```

`person_id` is only a frame-local detection index.

---

## RTMO + BoT-SORT, without ReID

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=false
```

Pipeline:

```text
RealSense → RTMO → BoT-SORT motion/IoU tracking → SkeletonFrame
```

This provides persistent track IDs with very small tracking overhead.

---

## RTMO + BoT-SORT + ReID

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=true
```

Pipeline:

```text
RealSense
   ↓
RTMO
   ↓
BoT-SORT
   ├── motion / IoU
   └── OSNet ReID
   ↓
persistent person_id
   ↓
SkeletonFrame
```

With tracking enabled, `PersonSkeleton.person_id` is the persistent BoT-SORT track ID.

---

# Visualization

The visualization output is separate from the robot-facing structured output.

Topic:

```text
/skeleton_detection/visualization_image
```

Type:

```text
sensor_msgs/msg/Image
```

Default visualization size:

```text
424 × 240
```

Default rate:

```text
10 Hz
```

## Enable visualization

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  publish_visualization_image:=true
```

## Tracking + ReID + visualization

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=true \
  publish_visualization_image:=true
```

## Visualization at 30 Hz

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=true \
  publish_visualization_image:=true \
  visualization_fps:=30.0
```

## Visualization at 50 Hz

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=true \
  publish_visualization_image:=true \
  visualization_fps:=50.0
```

The visualization rate is independent of the robot-facing `SkeletonFrame` publish rate.

---

# View the Visualization with RQT

On the Spark with a working graphical display:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

ros2 run rqt_image_view rqt_image_view \
  /skeleton_detection/visualization_image
```

Or:

```bash
ros2 run rqt_image_view rqt_image_view
```

and select:

```text
/skeleton_detection/visualization_image
```

The visualization contains:

- original RGB image
- person bounding boxes
- COCO-17 joints
- skeleton connections
- person confidence
- person ID

When tracking is enabled, the displayed ID is the persistent BoT-SORT track ID.

---

# ROS Outputs

## Robot-facing skeleton output

```text
/skeleton_detection/frame
```

Type:

```text
skeleton_detection/msg/SkeletonFrame
```

Each frame contains one or more `PersonSkeleton` messages.

Important `PersonSkeleton` fields include:

```text
person_id
score
bbox
joints
connections
position
```

Current conventions:

```text
bbox:
[x, y, width, height]
in original camera-image pixels

joints:
[x0, y0, confidence0,
 x1, y1, confidence1,
 ...]
17 COCO keypoints → 51 floats

connections:
19 COCO skeleton edges

position:
currently (0,0,0)
3D localization is not implemented yet
```

### `person_id` semantics

Tracking disabled:

```text
person_id = frame-local detection index
```

Tracking enabled:

```text
person_id = persistent BoT-SORT track ID
```

---

## Human-facing visualization output

```text
/skeleton_detection/visualization_image
```

Type:

```text
sensor_msgs/msg/Image
```

Default QoS:

```text
BEST_EFFORT
KEEP_LAST
depth = 1
VOLATILE
```

The visualization topic is intended only for inspection, not as a machine-readable perception contract.

---

# ROS Debug / Inspection Commands

## List nodes

```bash
ros2 node list
```

In live RealSense mode, the main perception pipeline should appear as one node:

```text
/rtmo_node
```

---

## List topics

```bash
ros2 topic list
```

Typical live outputs:

```text
/skeleton_detection/frame
/skeleton_detection/visualization_image
```

The visualization topic appears only when visualization publishing is enabled.

---

## Check SkeletonFrame frequency

```bash
ros2 topic hz /skeleton_detection/frame
```

---

## Check visualization frequency

```bash
ros2 topic hz /skeleton_detection/visualization_image
```

---

## Inspect one skeleton message

```bash
ros2 topic echo /skeleton_detection/frame --once
```

Use this to inspect:

- `person_id`
- confidence
- bbox
- joints
- connections

---

## Inspect visualization topic information

```bash
ros2 topic info /skeleton_detection/visualization_image --verbose
```

---

## Inspect the node

```bash
ros2 node info /rtmo_node
```

---

# Important ROS Parameters

## Pipeline

| Parameter | Default | Description |
|---|---:|---|
| `input_mode` | `ros_topic` / launch-specific | `realsense` for live D456 or `ros_topic` for offline ROS image input |
| `model_config` | `/opt/models/rtmo/rtmo-m.py` | RTMO config |
| `checkpoint` | `/opt/models/rtmo/rtmo-m.pth` | RTMO checkpoint |
| `device` | `cuda:0` | RTMO compute device |

---

## Tracking

| Parameter | Default | Description |
|---|---:|---|
| `enable_tracking` | `false` | Enable BoT-SORT |
| `with_reid` | `true` | Enable OSNet appearance ReID when tracking is on |
| `reid_checkpoint` | `/opt/models/reid/osnet_x0_25_msmt17.pt` | ReID checkpoint |
| `tracking_frame_rate` | `0.0` | `<=0` uses the configured fallback processing rate |
| `cmc_method` | `none` | Camera-motion compensation method |

BoT-SORT tuning parameters:

| Parameter | Default |
|---|---:|
| `track_high_thresh` | `0.5` |
| `new_track_thresh` | `0.6` |
| `track_buffer` | `30` |
| `match_thresh` | `0.8` |
| `appearance_thresh` | `0.25` |
| `proximity_thresh` | `0.5` |

For the current static-camera setup:

```text
cmc_method = none
```

is the recommended baseline.

For a moving/robot-mounted camera, camera-motion compensation should be re-evaluated.

---

## Visualization

| Parameter | Default | Description |
|---|---:|---|
| `publish_visualization_image` | `false` | Enable live visualization topic |
| `visualization_topic` | `/skeleton_detection/visualization_image` | Output topic |
| `visualization_width` | `424` | Output width |
| `visualization_height` | `240` | Output height |
| `visualization_fps` | `10.0` | Visualization target FPS |
| `visualization_reliability` | `best_effort` | ROS QoS reliability |
| `joint_score_threshold` | `0.3` | Minimum joint confidence to draw |
| `draw_joint_scores` | `false` | Draw numerical joint scores |
| `save_visualization_images` | `false` | Save annotated frames locally |
| `visualization_output_dir` | `output/visualizations` | Saved-image directory |
| `visualization_image_format` | `jpg` | Saved-image format |

Example custom visualization:

```bash
ros2 run skeleton_detection rtmo_node --ros-args \
  -p input_mode:=realsense \
  -p publish_visualization_image:=true \
  -p visualization_width:=640 \
  -p visualization_height:=360 \
  -p visualization_fps:=30.0
```

---

# Useful Launch Configurations

## Maximum throughput

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py
```

Tracking and visualization are off.

---

## Tracking without ReID

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=false
```

Recommended when maximum FPS matters more than appearance-based re-association.

---

## Full tracking with ReID

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=true
```

---

## Full tracking with live visualization

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true \
  with_reid:=true \
  publish_visualization_image:=true \
  visualization_fps:=30.0
```

---

# Offline Image Test

The original local-image RTMO path remains available.

```bash
ros2 launch skeleton_detection milestone1.launch.py
```

This can:

```text
local image
   ↓
ROS Image
   ↓
RTMO
   ↓
SkeletonFrame
   +
annotated visualization
```

Saved visualization images are written under:

```text
output/visualizations/
```

when enabled.

---

# Performance Notes

Typical measured behavior on the DGX Spark:

```text
RealSense capture:
~60 FPS

RTMO only:
~50–55 FPS

RTMO + BoT-SORT, ReID disabled:
~50 FPS with people

RTMO + BoT-SORT + ReID:
~38 FPS with 1–2 people

Visualization:
10 / 30 / 50 Hz tested successfully at 424×240
```

The largest additional cost currently comes from OSNet ReID.

Motion-only BoT-SORT adds very little overhead.

The capture path uses a **latest-frame-wins** buffer. If processing falls behind the 60 FPS camera, stale camera frames are intentionally dropped instead of building latency.

---

# Docker Image

Build:

```bash
cd /home/aims/Desktop/samd/skeleton_detection

docker build -t skeleton_humble_dev .
```

The Docker build:

- installs the pinned RTMO/OpenMMLab stack
- installs `boxmot==19.0.0`
- preserves `numpy==1.23.5`
- downloads and verifies model assets
- places them under `/opt/models`
- verifies important imports and package versions

Relevant Docker helper files:

```text
docker/
├── fetch_models.py
├── verify_image.py
└── mmcv_ext_stub.py
```

`fetch_models.py` prepares and checksum-verifies the model assets.

`verify_image.py` acts as a build-time regression gate for the Python/CUDA/OpenMMLab/BoxMOT environment.

---

# Current Limitations

- Real-person tracking and occlusion behavior still needs more live validation.
- ReID currently reduces pipeline throughput to roughly 38–40 FPS with people.
- `tracking_frame_rate` is fixed when BoT-SORT is constructed and does not dynamically adapt to measured runtime FPS.
- `cmc_method=none` assumes a static-camera baseline.
- `position` is not yet populated with depth-derived XYZ.
- No 3D person localization is implemented yet.
- No TensorRT optimization is currently used.

---

# Planned Next Step

The next planned perception stage is:

```text
tracked person
    ↓
aligned RealSense depth
    ↓
robust body anchor
    ↓
3D deprojection
    ↓
PersonSkeleton.position
```

The tracking ID will provide the persistent person identity to which future 3D location information can be attached.
