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

## 3. Live RealSense pipeline

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

## 4. Visualization

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

## 5. Offline / image pipeline

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

## 6. Inspecting ROS topics and messages

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
`skeleton_detection/msg/SkeletonFrame`:

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
