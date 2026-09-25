# Running the pipeline

Every command on this page is run **inside the container**. Get there with:

```bash
docker compose up -d
docker compose exec skeleton_humble bash
```

See [Docker & Compose](docker.md) for the container itself,
[Launch arguments](launch_arguments.md) for everything settable on the
`ros2 launch` command line, and [Parameters](parameters.md) for the full
parameter reference.

---

## 1. Build the ROS package

```bash
source /opt/ros/humble/setup.bash

cd /ros2_ws
colcon build --packages-select skeleton_detection

source /ros2_ws/install/setup.bash
```

Repeat the `colcon build` + `source` pair after changing Python source, launch
files, YAML configs or the `.msg` definitions. **No Docker image rebuild is
needed** — the repository is bind-mounted into the container.

Every later section assumes both setup files are sourced:

```bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
```

---

## 2. Executables and launch files

| | Name |
|---|---|
| Main executable | `iot_node` |
| Internal ROS node name | `rtmo_node` |
| Offline image publisher executable | `image_publisher` |
| Live bringup launch file | `skeleton_detection_bringup.launch.py` |

The executable is `iot_node`; the node it starts still calls itself
`rtmo_node`, which is the name `ros2 node list`, `ros2 node info` and the YAML
parameter files all use.

---

## 3. Input modes

`input_mode` selects where frames come from. All three converge on the same
RTMO → tracking → depth → `SkeletonFrame` path, so nothing downstream changes
with the mode.

| `input_mode` | Frames from | Depth / XYZ |
|---|---|---|
| `realsense` (default) | the D456, opened **in this process** with `pyrealsense2` | yes, aligned in-process |
| `ros_topic` | a colour-only `sensor_msgs/Image` topic — offline/dummy testing | no, `NaN` |
| `ros_camera` | one `RGBD` topic from an **externally running** `realsense2_camera` driver | yes, aligned **by the driver** |

The launch file starts **only the skeleton-detection node** in every mode. It
never launches `realsense2_camera`.

### `realsense` — direct capture (default)

The camera is opened inside the perception process, so no RGB frame crosses
DDS before inference. This is the normal live path; see section 4.

### `ros_topic` — colour-only image topic

The offline/local regression path: `image_publisher` (or any publisher) sends
`sensor_msgs/Image` on `input_topic`. No depth stream and no intrinsics, so
`position` and `depth` are `NaN` for every person. See section 6.

### `ros_camera` — external RealSense ROS driver

Consumes a single `realsense2_camera_msgs/msg/RGBD` topic from a
`realsense2_camera` driver that is **already running elsewhere**, typically in
another container.

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  input_mode:=ros_camera
```

```bash
# a different topic name
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  input_mode:=ros_camera \
  rgbd_topic:=/some/other/rgbd
```

**The external driver is responsible for** enabling colour, enabling depth,
RGB/depth synchronization, depth-to-colour alignment and publishing the RGBD
topic. Skeleton Detection only consumes the resulting message: it starts no
driver, opens no camera, calls no `rs.align` and runs no `message_filters`
synchronizer.

That means the driver must be started roughly with:

```text
enable_rgbd:=true
enable_sync:=true
align_depth.enable:=true
enable_color:=true
enable_depth:=true
```

`enable_rgbd` requires both `enable_sync` and `align_depth.enable`; without
them there is no RGBD topic to subscribe to.

One `RGBD` message carries everything this node needs, which is exactly why
this mode uses it instead of three separate subscriptions:

| RGBD field | Used for |
|---|---|
| `rgb` | converted to BGR with `CvBridge`, handed to RTMO |
| `depth` | the `(H, W)` aligned depth array, `16UC1` in millimeters |
| `rgb_camera_info` | `k` → `CameraIntrinsics(fx, fy, cx, cy, width, height)` |

Startup logs the mode and the topic, and nothing per frame:

```text
[rtmo_node]: Input mode: ros_camera
[rtmo_node]: RGBD topic: /camera/camera/rgbd
```

Then, once the first message arrives, the resolved intrinsics and depth scale:

```text
[rtmo_node]: Colour intrinsics from CameraInfo.k (848x480): fx=... fy=... cx=... cy=...
[rtmo_node]: Depth ENABLED: '16UC1' already aligned to colour by the external
             driver, depth_scale=0.001000 m/unit (no rs.align in this process)
```

`realsense_width`, `realsense_height`, `realsense_fps` and
`realsense_enable_depth` have **no effect** in this mode — the external driver
owns the stream configuration. Setting `realsense_enable_depth:=false` here
logs a warning saying so.

#### If the callback never fires

The subscription uses `best_effort` QoS to match the driver's sensor-data
publisher; a `reliable` subscription would never match it and would stay
silent. Check the topic exists and is actually publishing:

```bash
ros2 topic list | grep rgbd
ros2 topic hz   /camera/camera/rgbd
ros2 topic info /camera/camera/rgbd --verbose
```

If the topic is missing, the driver was started without `enable_rgbd:=true`.
If it exists but `hz` reports nothing, the problem is on the driver side. The
`--verbose` form prints the publisher's QoS.

A driver publishing depth that is **not** aligned to colour is rejected once,
at startup, naming the cause rather than failing once per frame:

```text
Depth image is 1280x720 but the colour stream is 848x480 ... Restart the
external driver with align_depth.enable:=true
```

---

## 4. Live RealSense pipeline

Everything in this section is `input_mode:=realsense`, the default — the
camera opened directly in this process. For the externally-driven variant see
[`ros_camera`](#ros_camera--external-realsense-ros-driver) above; the tracking
and visualization options below apply unchanged to it.

### Verify the node starts

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  run_duration_sec:=30.0
```

> **`run_duration_sec` is a DOUBLE.** Pass `30.0`, not `30` — an integer
> literal is rejected as the wrong parameter type. The same applies to every
> other float parameter, e.g. `visualization_fps:=30.0`.

A healthy start-up logs the resolved model paths:

```text
[rtmo_node]: RTMO model loaded (config=/opt/models/rtmo/rtmo-m.py, checkpoint=/opt/models/rtmo/rtmo-m.pth)
```

then periodic pipeline statistics, then a summary as it exits.

With **no camera attached** the model-loading half can still be checked on its
own — this loads the config and checkpoint and then fails at camera open, which
is enough to prove the baked assets are correct:

```bash
ros2 run skeleton_detection iot_node --ros-args -p input_mode:=realsense
```

### RTMO only (tracking off — maximum throughput)

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py
```

Equivalent, without the launch file:

```bash
ros2 run skeleton_detection iot_node --ros-args -p input_mode:=realsense
```

`person_id` is the frame-local detection index in this mode.

### RTMO + BoT-SORT, no ReID

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true \
  with_reid:=false
```

Persistent track IDs from motion/IoU alone, with very little overhead.

### RTMO + BoT-SORT + OSNet ReID

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true \
  with_reid:=true
```

With tracking on, `PersonSkeleton.person_id` is the persistent BoT-SORT track
ID.

### Occlusion-aware tracking (experimental, off by default)

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true \
  occlusion_aware_tracking:=true
```

Turn it off again explicitly with `occlusion_aware_tracking:=false` — that is
stock BoT-SORT with zero overhead. It requires `enable_tracking:=true`; on its
own it does nothing.

### Without depth

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  realsense_enable_depth:=false
```

`position` and `depth` are then `NaN` for every person.

---

## 5. Visualization

Visualization is **off by default**. It publishes
`/skeleton_detection/visualization_image` (`sensor_msgs/msg/Image`) at
424 × 240, 10 Hz, `best_effort` QoS.

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  publish_visualization_image:=true
```

Full tracking with visualization at 30 Hz:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true \
  with_reid:=true \
  publish_visualization_image:=true \
  visualization_fps:=30.0
```

The visualization rate is independent of the `SkeletonFrame` publish rate,
which always runs at the full pipeline rate. 10 / 30 / 50 Hz have all been run
successfully at 424 × 240.

A custom size, via `ros2 run`:

```bash
ros2 run skeleton_detection iot_node --ros-args \
  -p input_mode:=realsense \
  -p publish_visualization_image:=true \
  -p visualization_width:=640 \
  -p visualization_height:=360 \
  -p visualization_fps:=30.0
```

Append each person's camera-frame XYZ to the overlay label (debug):

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  publish_visualization_image:=true \
  draw_person_xyz:=true
```

### Viewing it with rqt

In a second shell into the container, with a working graphical display:

```bash
docker compose exec skeleton_humble bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash

ros2 run rqt_image_view rqt_image_view \
  /skeleton_detection/visualization_image
```

Or start `rqt_image_view` / `ros2 run rqt_gui rqt_gui` with no argument and
pick `/skeleton_detection/visualization_image` from the topic list.

The overlay contains the original RGB image, person bounding boxes, the COCO-17
joints and their 19 skeleton connections, each person's confidence, ID and
depth:

```text
ID 7  score=0.91 | Depth: 3.24 m
ID 8  score=0.84 | Depth: N/A
```

`Depth: N/A` means that person's depth is `NaN`. The visualization topic is for
inspection only — it is not a machine-readable perception contract.

---

## 6. Offline / image pipeline

The local-image path needs no camera. It is run as **two `ros2 run` commands**,
one per node — there is no launch file for it.

There is no start-up ordering requirement. `image_publisher` waits up to
`wait_for_subscriber_sec` (10.0 s by default) for a subscriber before it
publishes the first image, so the two nodes can be started in either order and
RTMO still has time to load its model.

Use the shipped YAML files, which set `input_mode: ros_topic`, enable
`save_visualization_images` and hold the image selection:

```bash
# shell 1 — the perception node
ros2 run skeleton_detection iot_node --ros-args \
  --params-file /ros2_ws/install/skeleton_detection/share/skeleton_detection/config/rtmo_node.yaml
```

```bash
# shell 2 — the image source
ros2 run skeleton_detection image_publisher --ros-args \
  --params-file /ros2_ws/install/skeleton_detection/share/skeleton_detection/config/image_publisher.yaml
```

No `-r __node:=` remap is needed: the nodes name themselves `rtmo_node` and
`image_publisher_node`, which are exactly the keys the two YAML files use.

To point at a different file, edit the YAML or override on the command line:

```bash
ros2 run skeleton_detection iot_node --ros-args \
  -p input_mode:=ros_topic \
  -p input_topic:=/dummy_camera/image_raw \
  -p save_visualization_images:=true

# in a second shell
ros2 run skeleton_detection image_publisher --ros-args \
  -p image_path:=/ros2_ws/src/skeleton_detection/data/images/000000000785.jpg
```

`image_publisher` selects its input with the **first match wins** rule:
`image_paths` (an explicit list), then `image_dir` (every supported image in a
directory, sorted by name), then `image_path` (a single file). It exits on its
own after the last image when `shutdown_after_publish` is true.

Annotated images are written to
`/ros2_ws/src/skeleton_detection/output/visualizations` (bind-mounted, so they
appear in `output/visualizations/` on the host).

There is **no depth stream and no intrinsics** on this path, so `position` and
`depth` are always `NaN`.

---

## 7. Inspecting ROS topics and messages

```bash
ros2 node list                 # the pipeline appears as /rtmo_node
ros2 node info /rtmo_node

ros2 topic list                # /skeleton_detection/frame (+ visualization_image when enabled)
ros2 topic hz   /skeleton_detection/frame
ros2 topic hz   /skeleton_detection/visualization_image
ros2 topic echo /skeleton_detection/frame --once
ros2 topic info /skeleton_detection/visualization_image --verbose
```

From a second shell in one line:

```bash
docker compose exec skeleton_humble bash -lc \
  'source /opt/ros/humble/setup.bash && source /ros2_ws/install/setup.bash && \
   ros2 topic hz /skeleton_detection/frame'
```

The visualization topic only appears when
`publish_visualization_image:=true`.

### Published messages

`/skeleton_detection/frame` carries
`patrolknight_msgs/msg/SkeletonFrame`:

```text
std_msgs/Header  header        # frame_id = camera_frame_id (camera_color_optical_frame)
int32            frame_index
float64          timestamp
PersonSkeleton[] persons
```

Each `PersonSkeleton`:

| Field | Type | Meaning |
|---|---|---|
| `person_id` | `int32` | persistent BoT-SORT track ID when tracking is on, else the frame-local detection index |
| `score` | `float32` | person/instance confidence |
| `bbox` | `float32[4]` | `[x, y, width, height]` in original camera-image pixels, unclipped |
| `joints` | `float32[]` | 17 COCO keypoints flattened to 51 floats: `[x0, y0, conf0, x1, y1, conf1, ...]` |
| `connections` | `int32[]` | the fixed 19-edge COCO topology flattened to 38 ints |
| `position` | `geometry_msgs/Point` | 3D point in **meters** in the colour camera **optical** frame: x right, y down, z forward. All-`NaN` when unavailable — never `0` |
| `depth` | `float32` | **Euclidean** camera-to-person distance `sqrt(x²+y²+z²)` in meters. **Not** the raw RealSense depth value, which is `position.z`. `NaN` when unavailable — never `0` |

Consumers must test availability with `math.isnan()`, not against `0`.

How `position` and `depth` are computed is explained in
[Architecture](architecture.md).
