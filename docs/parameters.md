# Parameter reference

Authoritative list of every parameter the package exposes.

All names, types and defaults below are taken from the current source:

- ROS parameter defaults — `skeleton_detection/iot_node.py`
  (`RTMONode._declare_parameters`) and
  `skeleton_detection/input/image_publisher.py`
- RGBD input constants — `skeleton_detection/input/ros_camera_subscriber.py`
- Config-file values — `config/rtmo_node.yaml`,
  `config/rtmo_node_direct_realsense.yaml`, `config/offline_rtmo_node.yaml`,
  `config/offline_image_publisher.yaml`

---

## How the layers combine

1. **Node default** — what `declare_parameter` sets when the YAML does not
   mention the parameter.
2. **Config file** — the YAML selected with `config:=<path>` (default
   `config/rtmo_node.yaml`). It is the single source of truth: the
   bringup launch file passes it to the node unchanged and has no
   per-parameter arguments.

To change a value, edit the YAML (or copy a packaged one, edit it and pass it
with `config:=`) — see [Launch arguments](launch_arguments.md). Outside the
launch file, `ros2 run skeleton_detection iot_node --ros-args -p name:=value`
still sets single parameters directly, e.g. for a quick check.

### Types matter

`run_duration_sec` is declared as `0.0` and is therefore a **DOUBLE**. Write
`run_duration_sec: 30.0` in the YAML (or `-p run_duration_sec:=30.0`), never
`30`. The same applies to
every other parameter whose default below is written with a decimal point —
`visualization_fps`, `tracking_frame_rate`, `stats_log_period_sec`,
`proximity_thresh`, `publish_interval_sec` and the rest.

---

## `rtmo_node` (executable `iot_node`)

### Input selection

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `input_mode` | string | `ros_topic` | `realsense` = open the D456 in this process with `pyrealsense2`; `ros_topic` = consume a colour-only `sensor_msgs/Image`; `ros_camera` = consume one `RGBD` topic from an external `realsense2_camera` driver. Any other value fails at start-up |
| `input_topic` | string | `/dummy_camera/image_raw` | input topic in `ros_topic` mode |
| `rgbd_topic` | string | `/camera/camera/rgbd` | `ros_camera` mode only: `realsense2_camera_msgs/msg/RGBD` topic of an **externally running** driver. One topic, not three — the message already carries colour, aligned depth and `CameraInfo` |

`config/rtmo_node.yaml` (the default) sets `input_mode: ros_camera`;
`config/rtmo_node_direct_realsense.yaml` sets `input_mode: realsense`;
`config/offline_rtmo_node.yaml` sets `input_mode: ros_topic`. The mode is chosen by
selecting the config file; the launch file itself has no `input_mode`
argument.

### What each mode provides

| Mode | Colour | Depth | Intrinsics | Alignment + sync done by |
|---|---|---|---|---|
| `realsense` | `pyrealsense2`, in process | Z16, in process | colour stream profile | this package (`rs.align`) |
| `ros_topic` | `sensor_msgs/Image` | none — `NaN` | none — `NaN` | n/a |
| `ros_camera` | `RGBD.rgb` | `RGBD.depth` (`16UC1` mm) | `RGBD.rgb_camera_info.k` | the **external driver** |

In `ros_camera` mode this package performs **no** RGB/depth synchronization and
**no** depth-to-colour alignment, and never calls `rs.align`. The external
driver must be started with `enable_rgbd:=true`, `enable_sync:=true`,
`align_depth.enable:=true`, `enable_color:=true`, `enable_depth:=true`.

The depth scale on that path comes from the image **encoding**, not from a
device: `16UC1` (millimeters) → `0.001`, `32FC1` (meters) → `1.0`. Any other
encoding is refused rather than guessed. Intrinsics and the depth scale are
resolved once, from the first RGBD message.

The RGBD subscription uses `best_effort` / `keep_last` / depth `1`, matching
the driver's sensor-data QoS. A `reliable` subscription would never match it.

### Input — RealSense (`input_mode: realsense`)

These configure the in-process camera and are **unused** in `ros_topic` and
`ros_camera` mode.

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `realsense_width` | int | `848` | colour width; validated against the device's advertised profiles at start-up |
| `realsense_height` | int | `480` | colour height |
| `realsense_fps` | int | `60` | colour frame rate |
| `realsense_color_format` | string | `bgr8` | D456 advertises `bgr8` natively at this profile, so no per-frame colour conversion happens |
| `realsense_serial` | string | `""` | empty = first device found |
| `realsense_enable_depth` | bool | `true` | open the Z16 depth stream and align it to colour. `false` = colour only; `position`/`depth` are then `NaN` for every person. **`realsense` mode only** — in `ros_camera` mode the external driver decides, and the node warns if this is false |
| `camera_frame_id` | string | `camera_color_optical_frame` | `header.frame_id`; must name the colour optical frame, because `position` is always expressed in it |

An unsupported width/height/fps/format combination fails at start-up with the
device's supported list rather than being silently substituted.

### Output

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `output_topic` | string | `/skeleton_detection/frame` | `SkeletonFrame` topic |

### RTMO model

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `model_config` | string | `$RTMO_MODEL_CONFIG`, else `/opt/models/rtmo/rtmo-m.py` | MMPose RTMO-M config |
| `checkpoint` | string | `$RTMO_CHECKPOINT`, else `/opt/models/rtmo/rtmo-m.pth` | RTMO-M weights |
| `device` | string | `cuda:0` | Torch device for RTMO |
| `person_score_threshold` | double | `0.3` | detections below this person score are not published at all |

The image exports `RTMO_MODEL_CONFIG` and `RTMO_CHECKPOINT`, so a different
config/checkpoint can be used without rebuilding, either through these
parameters or through the environment variables.

### Depth / XYZ

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `depth_keypoint_score_threshold` | double | `0.3` | a keypoint votes on the person's depth only at or above this RTMO **per-joint** confidence |
| `depth_min_m` | double | `0.0` | optional lower metric sanity bound; `<= 0` disables it |
| `depth_max_m` | double | `0.0` | optional upper metric sanity bound; `<= 0` disables it |

Zero, `NaN` and infinite depth samples are always rejected regardless of these
bounds. `DepthParams.window` (the 3 × 3 sampling neighbourhood) and
`DepthParams.depth_scale` are **not** ROS parameters: the window is fixed at
`3` in `inference/depth_estimation.py`, and the scale is read from the device
in `RealSenseCapture.start()`.

### Tracking — BoT-SORT

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `enable_tracking` | bool | `false` | enable in-process BoT-SORT; makes `person_id` a persistent track ID |
| `with_reid` | bool | `true` | use OSNet appearance features inside BoT-SORT |
| `reid_checkpoint` | string | `$REID_CHECKPOINT`, else `/opt/models/reid/osnet_x0_25_msmt17.pt` | ReID weights |
| `tracking_frame_rate` | double | `0.0` | `<= 0` uses the built-in `DEFAULT_TRACKING_FRAME_RATE` of **55** (the measured pipeline rate, not the 60 Hz camera rate). Scales BoT-SORT's lost-track buffer |
| `cmc_method` | string | `none` | camera-motion compensation: `none`, `ecc`, `orb`, `sift`, `sof`. `none`/`""` is mapped to Python `None`, which is what BoxMOT wants |
| `track_high_thresh` | double | `0.5` | BoxMOT default, unchanged |
| `new_track_thresh` | double | `0.6` | BoxMOT default, unchanged |
| `track_buffer` | int | `90` | **differs from BoxMOT's 30**; frames a lost track survives *before* frame-rate scaling |
| `match_thresh` | double | `0.8` | BoxMOT default, unchanged |
| `appearance_thresh` | double | `0.25` | BoxMOT default, unchanged |
| `proximity_thresh` | double | `0.7` | **differs from BoxMOT's 0.5**; IoU-*distance* gate above which the ReID distance is discarded |

`track_buffer` and `proximity_thresh` are the **only** two
association-affecting values that differ from the BoxMOT 19.0.0 defaults.

**`track_buffer` scaling.** BoxMOT 19.0.0 computes
`max_time_lost = int(frame_rate / 30.0 * track_buffer)`, so at
`tracking_frame_rate = 55` a lost track survives **165 frames (~3.0 s)**
instead of the 55 frames (~1.0 s) that `track_buffer = 30` would give.

**`proximity_thresh` is an IoU-distance, so raising it *relaxes* the gate.**
The ReID distance is masked to `1.0` whenever
`iou_dist > proximity_thresh`, with `iou_dist = 1 - IoU`:

| `proximity_thresh` | ReID usable for |
|---:|---|
| `0.50` (BoxMOT default) | `IoU >= 0.50` |
| **`0.70` (this package)** | **`IoU >= 0.30`** |
| `0.30` | `IoU >= 0.70` (tighter — not what you want) |

### Tracking — occlusion-aware extension (experimental)

Requires `enable_tracking: true`. With `occlusion_aware_tracking: false` a
stock `BotSort` is constructed and none of this code runs.

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `occlusion_aware_tracking` | bool | `false` | master switch; `false` = stock BoT-SORT, zero overhead |
| `keypoint_visibility_threshold` | double | `0.30` | a COCO-17 joint counts as visible at or above this RTMO **per-keypoint** score |
| `visible_ratio_threshold` | double | `0.50` | `visible_ratio` below this marks the detection `OCCLUDED` — **the only classification input** |
| `normal_bbox_history_size` | int | `15` | *informational only:* length of the per-track `NORMAL` bbox-width ring buffer shown in the debug log. Must be `>= 1` |
| `min_normal_width_samples` | int | `5` | *informational only:* `NORMAL` widths needed before the debug log prints a width ratio. Must be `>= 1` |

The last two classify nothing and change no tracker behaviour; the node raises
at start-up if either is below `1`.

### Visualization

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `publish_visualization_image` | bool | `false` | publish the live annotated image topic |
| `visualization_topic` | string | `/skeleton_detection/visualization_image` | output topic |
| `visualization_width` | int | `424` | published width; must be `>= 1` |
| `visualization_height` | int | `240` | published height; must be `>= 1` |
| `visualization_fps` | double | `10.0` | publish rate; `<= 0` = every processed frame, no rate limit |
| `visualization_reliability` | string | `best_effort` | QoS reliability: `best_effort` or `reliable`; anything else fails at start-up |
| `joint_score_threshold` | double | `0.3` | minimum joint confidence to draw |
| `draw_joint_scores` | bool | `false` | draw numeric per-joint scores |
| `draw_person_xyz` | bool | `false` | DEBUG: append `XYZ: (x, y, z) m` (colour optical frame) to each label |
| `save_visualization_images` | bool | `false` | write an annotated file for **every** processed frame — not rate limited |
| `visualization_output_dir` | string | `/ros2_ws/src/skeleton_detection/output/visualizations` | saved-image directory |
| `visualization_image_format` | string | `jpg` | saved-image format |

The published QoS is fixed at `KEEP_LAST`, depth `1`, `VOLATILE`; only the
reliability is configurable.

### Runtime / statistics

| Parameter | Type | Node default | Description |
|---|---|---|---|
| `stats_log_period_sec` | double | `1.0` | periodic statistics summary; `<= 0` disables the timer |
| `log_every_frame` | bool | `false` | one log line per processed frame |
| `run_duration_sec` | **double** | `0.0` | stop automatically after N seconds; `0` = run forever. **Pass `30.0`, not `30`** |

---

## `image_publisher_node` (executable `image_publisher`)

Offline/test input. Declared in
`skeleton_detection/input/image_publisher.py`, configured by
`config/offline_image_publisher.yaml`.

| Parameter | Type | Node default | Config value | Description |
|---|---|---|---|---|
| `image_paths` | string[] (dynamic typing) | `[]` | `[]` | explicit list of files, published in the given order. **First match wins** |
| `image_dir` | string | `""` | `""` | every supported image in the directory, sorted by name. Used when `image_paths` is empty |
| `image_path` | string | `/ros2_ws/src/skeleton_detection/data/images/000000000785.jpg` | same | a single file; used when the two above are empty |
| `output_topic` | string | `/dummy_camera/image_raw` | same | topic to publish on |
| `frame_id` | string | `camera_color_optical_frame` | same | header frame id |
| `publish_interval_sec` | double | `1.0` | `1.0` | seconds between two published images |
| `wait_for_subscriber_sec` | double | `10.0` | `10.0` | wait for a matched subscriber before the first publish, then publish anyway |
| `shutdown_after_publish` | bool | `true` | `true` | exit after the last image so `ros2 run` returns on its own |

`image_paths` is declared with `ParameterDescriptor(dynamic_typing=True)`
because an empty-list default would otherwise be inferred as `BYTE_ARRAY` and
would then reject a string list from YAML or the command line.

Supported suffixes: `.jpg`, `.jpeg`, `.png`, `.bmp`.

---

## Launch argument

| Launch file | Argument | Default | Description |
|---|---|---|---|
| `skeleton_detection_bringup.launch.py` | `config` | `<share>/config/rtmo_node.yaml` | parameter file for `rtmo_node`; the only launch argument, and the single source of every node parameter |

`<share>` is `get_package_share_directory("skeleton_detection")`.

`skeleton_detection_bringup.launch.py` is the only launch file in the package.
The offline image path has no launch file: both nodes are started with
`ros2 run` and a `--params-file`, so the YAML files are passed directly rather
than through a launch argument. See
[Running the pipeline](running.md#6-offline--image-pipeline).

---

## Config-file-only values by file

**`config/rtmo_node_direct_realsense.yaml`** (direct camera path) restates most of the node
defaults explicitly, so the whole live configuration is readable in one place.
The only value in it that actually *differs* from the node default is:

- `input_mode: realsense`

In particular `track_buffer: 90` and `proximity_thresh: 0.70` are already the
node defaults — the tuning lives in `iot_node.py`, and the YAML documents it
rather than introducing it.

**`config/offline_rtmo_node.yaml`** (offline path) differs from the node defaults in:

- `publish_visualization_image: true`
- `visualization_width: 640`, `visualization_height: 425`
- `visualization_fps: 0.0` (no rate limit)
- `log_every_frame: true`
- `save_visualization_images: true`

It also sets `input_topic: /dummy_camera/image_raw`,
`visualization_output_dir` and `visualization_image_format: jpg`, all of which
match the node defaults.

Neither YAML file sets `track_high_thresh`, `new_track_thresh`,
`match_thresh`, `appearance_thresh`, `visualization_reliability` or
`draw_person_xyz`; those come from the node defaults above.

---

## Environment variables

Exported by the Docker image and used as the node's parameter defaults:

| Variable | Image value |
|---|---|
| `RTMO_MODEL_CONFIG` | `/opt/models/rtmo/rtmo-m.py` |
| `RTMO_CHECKPOINT` | `/opt/models/rtmo/rtmo-m.pth` |
| `REID_CHECKPOINT` | `/opt/models/reid/osnet_x0_25_msmt17.pt` |

A ROS parameter always wins over the environment variable, because the
variable is only read to compute the declared default.
