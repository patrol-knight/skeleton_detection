

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

All model assets are baked into the Docker image, so a fresh
`git clone` + `docker build` gives a fully runnable container with **no manual
file copying** and no `docker cp`:

```text
/opt/models/
├── rtmo/
│   ├── rtmo-m.py            RTMO-M config          ← tracked in git (models/rtmo/)
│   ├── default_runtime.py   its `_base_`           ← tracked in git (models/rtmo/)
│   └── rtmo-m.pth           RTMO-M Body7 weights   ← downloaded during docker build
└── reid/
    └── osnet_x0_25_msmt17.pt  OSNet ReID weights   ← downloaded during docker build
```

The **configs live in git**; the **weights do not** (they are binaries, and
`.gitignore` blocks `*.pth` / `*.pt`). Both end up in an image layer, so
deleting and recreating the container never loses them.

---

## Data Flow

```text
Intel RealSense D456
        │
        │ pyrealsense2
        │ rs.align(rs.stream.color)
        ▼
848×480 BGR NumPy frame  +  aligned Z16 depth frame
        │                              │
        │ same Python process          │
        ▼                              │
      RTMO-M                           │
        │                              │
        ├── bbox (xyxy)                │
        ├── person confidence          │
        └── COCO-17 keypoints          │
        │                              │
        ▼                              │
  BoT-SORT Tracking          optional  │
        │                              │
        ├── motion / IoU               │
        └── OSNet ReID       optional  │
              │                        │
              └── crops people directly from the SAME BGR frame
        │                              │
        ▼                              ▼
 persistent track_id           person_depth module
        │                       two-stage median
        │                              │
        └──────────────┬───────────────┘
                       ▼
                person depth (m)
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
├── occlusion_tracking.py
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
| `occlusion_tracking.py` | Experimental occlusion-aware Kalman/ReID state protection |
| `message_builder.py` | Converts internal detections into ROS messages |
| `pipeline_stats.py` | FPS and latency statistics |
| `visualization.py` | Bounding box, skeleton, ID, score, and visualization rendering |
| `coco_keypoints.py` | COCO-17 keypoint names and connections |
| `image_publisher.py` | Offline/local-image test input |

Model configs are version-controlled alongside the source:

```text
models/
└── rtmo/
    ├── rtmo-m.py           MMPose RTMO-M (Body7, 640×640) config
    └── default_runtime.py  the `_base_` rtmo-m.py inherits from
```

Both are vendored from `open-mmlab/mmpose` v1.3.2. Upstream, `rtmo-m.py`
declares `_base_ = ['../../../_base_/default_runtime.py']`, which only resolves
inside the MMPose config tree; the sole edit made here is to point that at the
sibling `./default_runtime.py`. That base file is not optional — it supplies
`default_scope = 'mmpose'`, without which `init_model` cannot resolve the model
registry.

---

# Quick Start

Four steps from nothing to a running pipeline. No model file is ever copied by
hand.

## 0. Clone

```bash
git clone <this-repo-url> skeleton_detection
cd skeleton_detection
```

Every command below assumes you are in that clone; `$(pwd)` is used instead of
any absolute host path, so the workflow is not tied to one machine.

## 1. Build the Docker image

```bash
docker build -t skeleton_humble_dev .
```

This installs the pinned OpenMMLab/CUDA stack, copies `models/rtmo/` to
`/opt/models/rtmo/`, downloads and sha256-verifies the RTMO-M and ReID
checkpoints, and runs a build-time verification gate. Expect a long first
build; the checkpoint download needs network access.

## 2. Start the Docker container

```bash
docker run -d --name skeleton_humble \
  --privileged \
  --network host \
  --ipc host \
  --gpus all \
  -v "$(pwd)":/ros2_ws/src/skeleton_detection \
  -v /dev:/dev \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  skeleton_humble_dev \
  sleep infinity
```

The bind mount covers **source code only** — models come from the image, not
the mount, so editing code on the host still takes effect immediately without
putting weights on the host.

If the container already exists:

```bash
docker start skeleton_humble
```

Enter the container:

```bash
docker exec -it skeleton_humble bash
```

Sanity-check the baked model assets at any time:

```bash
docker exec skeleton_humble ls -lh /opt/models/rtmo /opt/models/reid
```

---

## 3. Build the ROS package

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

## 4. Verify the RTMO node starts

Inside the container, with a RealSense D456 attached:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  run_duration_sec:=10.0
```

A healthy start-up logs the resolved model paths, e.g.:

```text
[rtmo_node]: RTMO model loaded (config=/opt/models/rtmo/rtmo-m.py, checkpoint=/opt/models/rtmo/rtmo-m.pth)
```

then periodic pipeline statistics, and exits after 10 seconds.

With no camera attached, the model-loading half can still be checked on its own
— this loads the config and checkpoint and then fails at camera open, which is
enough to prove the baked assets are correct:

```bash
ros2 run skeleton_detection rtmo_node --ros-args -p input_mode:=realsense
```

Confirm the topic is publishing from a second shell:

```bash
docker exec -it skeleton_humble bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
   ros2 topic hz /skeleton_detection/frame'
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
depth
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
full 3D deprojection is not implemented yet

depth:
estimated distance of the person from the camera,
in METERS
NaN = not available (never 0)
```

### `depth` semantics

`depth` is a single scalar per person, in meters, estimated from the RealSense
depth image aligned to the colour frame. It is computed as a **two-stage
median**:

```text
for every VISIBLE keypoint (joint confidence >= depth_keypoint_score_threshold):
    3x3 depth neighbourhood around (u, v)
        drop zero / NaN / inf samples
        median  ->  joint_depth

drop keypoints with no valid joint_depth

median of the remaining joint_depth values  ->  person.depth
```

The median is used at both stages on purpose: depth around a person is
bimodal (person surface vs. background), and a mean would place the person in
the empty space between the two. It must not be replaced by a mean.

If no visible keypoint yields a valid depth — no depth stream, the person is
out of the depth range, or the depth is all holes — `depth` is `NaN`, never
`0`. Consumers must test with `math.isnan()`; the visualization prints
`Depth: N/A`.

The implementation lives in `skeleton_detection/person_depth.py`, which is
pure NumPy (no ROS, no pyrealsense2) and is covered by
`test/test_person_depth.py`.

### Depth-to-colour alignment

RTMO runs on the COLOUR image, so its keypoints are colour pixels. The raw
depth frame lives in the depth sensor's own pixel grid, so it is passed
through `rs.align(rs.stream.color)` in the capture thread
(`skeleton_detection/realsense_capture.py`, `_capture_loop`) before the array
is handed on. That is the single place alignment happens; `person_depth.py`
assumes its input is already aligned. The device depth scale (meters per raw
Z16 unit) is read once in `RealSenseCapture.start()` and applied inside the
depth module, so nothing is ever scaled twice.

Depth is on by default and can be turned off with
`realsense_enable_depth:=false`, in which case every `depth` is `NaN`. In
`input_mode=ros_topic` there is no depth stream at all, so `depth` is always
`NaN` on that path.

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

Each drawn person is labelled with its ID, its detection score and its depth:

```text
ID 7  score=0.91 | Depth: 3.24 m
ID 8  score=0.84 | Depth: N/A
```

`Depth: N/A` means the depth for that person is `NaN` (no depth stream, or no
visible keypoint with a valid depth sample).

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
- `depth` (meters; `nan` when unavailable)

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
| `track_buffer` | `90` |
| `match_thresh` | `0.8` |
| `appearance_thresh` | `0.25` |
| `proximity_thresh` | `0.70` |

#### `proximity_thresh` operates on IoU *distance*, not raw IoU

BoT-SORT discards the appearance distance for a pair whenever

```text
iou_dist > proximity_thresh        with    iou_dist = 1 - IoU
```

so the threshold is an **IoU-distance** value and **raising** it makes the gate
**more permissive**:

| `proximity_thresh` | ReID usable for |
|---:|---|
| `0.50` (BoxMOT default) | `IoU >= 0.50` |
| **`0.70` (this package)** | **`IoU >= 0.30`** |
| `0.30` | `IoU >= 0.70` (tighter — not what you want) |

Raised from `0.50` to `0.70` because the debug log repeatedly showed strong
appearance matches being thrown away by the spatial gate:

```text
IoU 0.146   RAW ReID 0.152    appearance PASS, proximity FAIL -> masked to 1.0
IoU 0.164   RAW ReID 0.218    appearance PASS, proximity FAIL -> masked to 1.0
IoU 0.159   RAW ReID 0.087    appearance PASS, proximity FAIL -> masked to 1.0
```

The gate is relaxed, not removed: below `IoU 0.30` appearance is still
discarded and association falls back to IoU alone. `appearance_thresh` stays
at `0.25`, so a weak appearance match is still rejected on its own merits.

`track_buffer` was raised from `30` to `90` for the ID-switch investigation.
BoxMOT 19.0.0 scales it by the frame rate:

```text
max_time_lost = int(frame_rate / 30.0 * track_buffer)
```

so at `tracking_frame_rate = 55` a lost track now survives **165 frames
(~3.0 s)** instead of 55 frames (~1.0 s). `track_buffer` and
`proximity_thresh` are the only two association-affecting values that differ
from the BoxMOT defaults.

### Occlusion-aware tracking (experimental)

Off by default. When on, every matched observation is classified into exactly
two states, `NORMAL` or `OCCLUDED`, and an `OCCLUDED` observation is prevented
from corrupting the track's motion and appearance state:

| | `NORMAL` | `OCCLUDED` |
|---|---|---|
| Kalman prediction | runs | **runs** (unchanged) |
| Kalman measurement update | applied | **skipped** |
| Velocity `mean[4:8]` | updated | **preserved, never reset or zeroed** |
| ReID `curr_feat` / `smooth_feat` / history | updated | **frozen** |
| NORMAL bbox-width history | appended | **not appended** |
| Track lifetime (`frame_id`, `tracklet_len`, `state`, `conf`, `cls`, `det_ind`) | updated | updated |

The detection still receives the track id in both states, so `person_id` is
unaffected. No association logic changes: IoU distance, the proximity gate,
ReID distance, the appearance gate, the Hungarian assignment, the track buffer
and the four association stages are stock BoxMOT, and none of the tuning
parameters above are touched.

The rule has **one** signal:

```text
is_occluded = visibility.available
              and visible_ratio < visible_ratio_threshold
```

where

```text
visible_ratio = (# RTMO keypoints with score >= keypoint_visibility_threshold)
                / (# keypoints)
```

`visible_ratio` uses RTMO's **per-joint** `keypoint_scores`, not the aggregate
person score.

**bbox width is not a signal.** An earlier version also marked a detection
`OCCLUDED` when its width fell below `0.60` of the track's recent median. That
misfired whenever somebody simply turned sideways — 16/17 keypoints visible,
nothing occluding them, box down to ~0.45 of its frontal width. A person's own
pose changes their box width as much as an occluder does, so width cannot tell
the two apart. The per-track width history is still maintained and still
printed in the debug log, labelled `classification use: INFORMATIONAL ONLY`; it
has zero effect on classification or on any tracker state.

| Parameter | Default | Description |
|---|---:|---|
| `occlusion_aware_tracking` | `false` | Master switch; `false` = stock BoT-SORT, zero overhead |
| `keypoint_visibility_threshold` | `0.30` | A joint counts as visible at or above this per-keypoint score |
| `visible_ratio_threshold` | `0.50` | `visible_ratio` below this marks the detection `OCCLUDED` — **the only classification input** |
| `normal_bbox_history_size` | `15` | *Informational only:* length of the per-track `NORMAL` bbox-width ring buffer shown in the debug log |
| `min_normal_width_samples` | `5` | *Informational only:* `NORMAL` widths needed before the debug log prints a width ratio |

Edge cases:

- If RTMO's keypoint scores are missing or non-finite, the visible-ratio
  criterion is **skipped** (the detection stays `NORMAL`) and a throttled
  warning is logged. Missing keypoints are never read as zero confidence.
- A `LOST` track re-activated on an `OCCLUDED` detection gets the same
  protection: it is re-activated (id preserved, `new_id=False`) but the partial
  box does not move the filter and does not update the appearance feature.

ON:

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true occlusion_aware_tracking:=true \
  tracking_debug_enabled:=true
```

OFF (current behaviour):

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true occlusion_aware_tracking:=false
```

Implementation lives entirely in `skeleton_detection/occlusion_tracking.py`;
nothing under `/usr/local/lib/python3.10/dist-packages/boxmot` is modified. See
the "Removing it" section of that module.

### TEMPORARY: new-track debug instrumentation

| Parameter | Default | Description |
|---|---:|---|
| `tracking_debug_enabled` | `false` | Write one diagnostic block per newly created BoT-SORT id |
| `tracking_debug_path` | `/ros2_ws/src/skeleton_detection/output/tracking_debug.log` | Debug file; truncated on every node launch |

With `occlusion_aware_tracking:=true` the same file also receives one line per
`NORMAL <-> OCCLUDED` transition and one block per `OCCLUDED` matched
observation (plus the frame that returns to `NORMAL`), carrying the visible
ratio, the width ratio, the reasons, whether the Kalman measurement update was
skipped, whether ReID was frozen, and the preserved velocity. No embedding
vectors are written.

Off by default, in which case a stock `BotSort` is constructed and the overhead
is zero. When on, each **newly allocated** persistent id (not an update, not a
re-activation of a lost id) appends a human-readable section containing the raw
IoU matrix, the raw pre-mask ReID distances, the masked cost matrix, the
Hungarian assignment and which stage the detection fell through. Nothing extra
goes to the ROS log.

```bash
ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
  enable_tracking:=true tracking_debug_enabled:=true
```

This is throwaway diagnostic code: see the "Removing this instrumentation"
section of `skeleton_detection/tracking_debug.py`.

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

Build (from the repository root):

```bash
docker build -t skeleton_humble_dev .
```

The Docker build:

- installs the pinned RTMO/OpenMMLab stack
- installs `boxmot==19.0.0`
- preserves `numpy==1.23.5`
- copies `models/rtmo/` (config, tracked in git) to `/opt/models/rtmo/`
- downloads and sha256-verifies the checkpoints into `/opt/models`
- verifies important imports, package versions, and asset presence

Checkpoint sources are official only:

| Asset | Source |
|---|---|
| `rtmo-m.pth` | `https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth` |
| `osnet_x0_25_msmt17.pt` | BoxMOT's own model registry URL, fetched with BoxMOT's downloader |

Relevant Docker helper files:

```text
docker/
├── fetch_models.py
├── verify_image.py
└── mmcv_ext_stub.py
```

`fetch_models.py` downloads and checksum-verifies the two checkpoints, and
asserts that the git-tracked RTMO config was copied in and loads with its
`_base_` chain resolved. It deliberately does **not** generate the config — a
missing or broken config fails `docker build`, not node start-up.

To point the node at a different config/checkpoint without rebuilding, override
the `model_config` / `checkpoint` ROS parameters (or the `RTMO_MODEL_CONFIG` /
`RTMO_CHECKPOINT` environment variables the image sets).

`verify_image.py` acts as a build-time regression gate for the Python/CUDA/OpenMMLab/BoxMOT environment.

---

# Current Limitations

- Real-person tracking and occlusion behavior still needs more live validation.
- ReID currently reduces pipeline throughput to roughly 38–40 FPS with people.
- `tracking_frame_rate` is fixed when BoT-SORT is constructed and does not dynamically adapt to measured runtime FPS.
- `cmc_method=none` assumes a static-camera baseline.
- `position` is not yet populated with depth-derived XYZ; only the scalar
  `depth` field is filled in.
- No full 3D person localization (deprojection to XYZ) is implemented yet.
- No TensorRT optimization is currently used.

---

# Planned Next Step

The next planned perception stage is:

```text
tracked person
    ↓
aligned RealSense depth          done
    ↓
robust body anchor               done (two-stage median -> PersonSkeleton.depth)
    ↓
3D deprojection                  next
    ↓
PersonSkeleton.position
```

The tracking ID will provide the persistent person identity to which future 3D location information can be attached.
