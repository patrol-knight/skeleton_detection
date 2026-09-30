# Running the pipeline

Every command on this page is run **inside the container**. `docker compose
up -d` already starts the pipeline with the mounted deployment config, so for
the manual launches below use a separate, throwaway container instead (it
still needs the GPU and the `iot_ros-net` network from `compose.yaml`):

```bash
docker compose run --rm skeleton_humble bash
```

See [Docker & Compose](docker.md) for the container itself,
[Launch arguments](launch_arguments.md) for selecting the parameter file
with `config:=`, and [Parameters](parameters.md) for the full
parameter reference.

---

## 1. The ROS packages

`patrolknight_msgs` and `skeleton_detection` are built into the image at
`/opt/patrolknight_msgs` and `/opt/skeleton_detection`, and every container
shell already has them sourced. There is nothing to build inside the
container. After changing source, launch files or packaged configs, rebuild
the image (`docker compose build`); for an edit-and-run loop without a
rebuild see [Docker & Compose](docker.md#development-iterating-without-rebuilding-the-image).

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

Every node setting on this page is a value in the YAML config file passed
with `config:=`; the launch file has no other argument. For a one-off change,
copy a packaged config, edit it and select the copy:

```bash
cp /opt/skeleton_detection/share/skeleton_detection/config/rtmo_node.yaml /tmp/my.yaml
# edit /tmp/my.yaml
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py config:=/tmp/my.yaml
```

The recipes below list the YAML values to set; each is then started with that
command. See [Launch arguments](launch_arguments.md).

`input_mode` selects where frames come from. All three converge on the same
RTMO → tracking → depth → `SkeletonFrame` path, so nothing downstream changes
with the mode.

| `input_mode` | Frames from | Depth / XYZ |
|---|---|---|
| `realsense` | the D456, opened **in this process** with `pyrealsense2` | yes, aligned in-process |
| `ros_topic` | a colour-only `sensor_msgs/Image` topic — offline/dummy testing | no, `NaN` |
| `ros_camera` (default) | one `RGBD` topic from an **externally running** `realsense2_camera` driver | yes, aligned **by the driver** |

The launch file starts **only the skeleton-detection node** in every mode. It
never launches `realsense2_camera`.

### `realsense` — direct capture (optional)

The camera is opened inside the perception process, so no RGB frame crosses
DDS before inference. Selected with `config/rtmo_node_direct_realsense.yaml`;
see section 4.

### `ros_topic` — colour-only image topic

The offline/local regression path: `image_publisher` (or any publisher) sends
`sensor_msgs/Image` on `input_topic`. No depth stream and no intrinsics, so
`position` and `depth` are `NaN` for every person. See section 6.

### `ros_camera` — external RealSense ROS driver

Consumes a single `realsense2_camera_msgs/msg/RGBD` topic from a
`realsense2_camera` driver that is **already running elsewhere**, typically in
another container. It is the default: the packaged `config/rtmo_node.yaml`
selects it (with tracking, ReID, occlusion-aware tracking and visualization
on):

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py
```

For a different topic name, set `rgbd_topic: /some/other/rgbd` in your copy of
that file.

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
owns the stream configuration. Setting `realsense_enable_depth: false` here
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

Everything in this section is `input_mode: realsense`, selected with
`config/rtmo_node_direct_realsense.yaml` — the camera opened directly in this
process. Start it with:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  config:=/opt/skeleton_detection/share/skeleton_detection/config/rtmo_node_direct_realsense.yaml
```

The recipes below are values to set in your copy of that file. For the externally-driven variant see
[`ros_camera`](#ros_camera--external-realsense-ros-driver) above; the tracking
and visualization options below apply unchanged to it.

### Verify the node starts

```yaml
run_duration_sec: 30.0
```

> **`run_duration_sec` is a DOUBLE.** Write `30.0`, not `30` — an integer
> literal is rejected as the wrong parameter type. The same applies to every
> other float parameter, e.g. `visualization_fps: 30.0`.

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

`rtmo_node_direct_realsense.yaml` as shipped (tracking and visualization off):

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  config:=/opt/skeleton_detection/share/skeleton_detection/config/rtmo_node_direct_realsense.yaml
```

Equivalent, without the launch file:

```bash
ros2 run skeleton_detection iot_node --ros-args -p input_mode:=realsense
```

`person_id` is `-1` for every person in this mode (no tracker identity).

### RTMO + BoT-SORT, no ReID

```yaml
enable_tracking: true
with_reid: false
```

Persistent track IDs from motion/IoU alone, with very little overhead.

### RTMO + BoT-SORT + OSNet ReID

```yaml
enable_tracking: true
with_reid: true
```

With tracking on, `PersonSkeleton.person_id` is the persistent BoT-SORT track
ID, or `-1` for a detection that has no confirmed track yet.

### Occlusion-aware tracking (experimental, off by default)

```yaml
enable_tracking: true
occlusion_aware_tracking: true
```

Turn it off again with `occlusion_aware_tracking: false` — that is stock
BoT-SORT with zero overhead. It requires `enable_tracking: true`; on its own
it does nothing.

### Without depth

```yaml
realsense_enable_depth: false
```

`position` and `depth` are then `NaN` for every person.

---

## 5. Visualization

Visualization is **off by default**. It publishes
`/skeleton_detection/visualization_image` (`sensor_msgs/msg/Image`) at
424 × 240, 10 Hz, `best_effort` QoS.

```yaml
publish_visualization_image: true
```

Full tracking with visualization at 30 Hz:

```yaml
enable_tracking: true
with_reid: true
publish_visualization_image: true
visualization_fps: 30.0
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

```yaml
publish_visualization_image: true
draw_person_xyz: true
```

### Viewing it with rqt

The deployed container is headless, so run the viewer in a throwaway
container with X11 (see [GUI tools](docker.md#gui-tools-rqt--opt-in)):

```bash
xhost +local:docker      # on the HOST
docker compose run --rm -e DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix \
  skeleton_humble ros2 run rqt_image_view rqt_image_view \
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
  --params-file /opt/skeleton_detection/share/skeleton_detection/config/offline_rtmo_node.yaml
```

```bash
# shell 2 — the image source
ros2 run skeleton_detection image_publisher --ros-args \
  --params-file /opt/skeleton_detection/share/skeleton_detection/config/offline_image_publisher.yaml
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
  'source /opt/ros/humble/setup.bash && source /opt/skeleton_detection/setup.bash && \
   ros2 topic hz /skeleton_detection/frame'
```

The visualization topic only appears when
`publish_visualization_image: true`.

### Published messages

`/skeleton_detection/frame` carries
`patrolknight_msgs/msg/SkeletonFrame`:

```text
std_msgs/Header  header        # frame_id = camera_frame_id (camera_color_optical_frame)
int32            frame_index
PersonSkeleton[] persons
```

`header.stamp` is the only timestamp. It is copied unchanged from the input
frame's header; in `ros_camera` mode that is the RealSense driver's colour
image stamp.

Each `PersonSkeleton`:

| Field | Type | Meaning |
|---|---|---|
| `person_id` | `int32` | `>= 0`: persistent BoT-SORT track ID. `-1`: no persistent identity (tracking off, or no confirmed track for this detection yet); the person is still published |
| `score` | `float32` | person/instance confidence |
| `bbox` | `float32[4]` | `[x, y, width, height]` in original camera-image pixels, unclipped |
| `joints` | `float32[]` | 17 COCO keypoints flattened to 51 floats: `[x0, y0, conf0, x1, y1, conf1, ...]` |
| `connections` | `int32[]` | the fixed 19-edge COCO topology flattened to 38 ints |
| `position` | `geometry_msgs/Point` | 3D point in **meters** in the colour camera **optical** frame: x right, y down, z forward. All-`NaN` when unavailable — never `0` |
| `depth` | `float32` | **Euclidean** camera-to-person distance `sqrt(x²+y²+z²)` in meters. **Not** the raw RealSense depth value, which is `position.z`. `NaN` when unavailable — never `0` |

Consumers must test availability with `math.isnan()`, not against `0`.

How `position` and `depth` are computed is explained in
[Architecture](architecture.md).
