# Launch arguments

Quick reference for `skeleton_detection_bringup.launch.py`, the only launch
file in the package.

This page lists **only what can be set on the `ros2 launch` command line**. The
node declares 48 ROS parameters in total; the 24 below are the ones the launch
file exposes as arguments. For the other 24 — topic names, model paths,
BoxMOT association thresholds, depth sanity bounds, logging — see the full
[Parameter reference](parameters.md), which is the authoritative list.

The `DeclareLaunchArgument(...)` calls in
`launch/skeleton_detection_bringup.launch.py` are the source of truth for every
name and default on this page.

---

## The main command

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py
```

With no arguments this runs the live RealSense path with **tracking off** and
**visualization off** — RTMO detection, depth/XYZ and `SkeletonFrame`
publication at maximum throughput.

Arguments are appended as `name:=value`:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true \
  publish_visualization_image:=true
```

You can also discover the list at runtime, straight from the installed launch
file:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py --show-args
```

### Types matter

An argument whose default is written with a decimal point is a **DOUBLE**, and
an integer literal is rejected as the wrong parameter type. Pass
`run_duration_sec:=30.0`, never `run_duration_sec:=30`. The same applies to
`visualization_fps`, `proximity_thresh`, `keypoint_visibility_threshold` and
`visible_ratio_threshold`.

### How a value reaches the node

The launch file loads `config/rtmo_node_realsense.yaml` for the full parameter
set, then applies these arguments as **overrides on top of it**. So a launch
argument always wins over the YAML, which in turn wins over the node default.
`input_mode` is not an argument: the bringup file forces it to `realsense`.

---

## Camera

| Argument | Default | Description |
|---|---|---|
| `realsense_width` | `848` | Colour width. Validated against the device's advertised profiles at start-up. |
| `realsense_height` | `480` | Colour height. |
| `realsense_fps` | `60` | Colour frame rate. |
| `realsense_enable_depth` | `true` | Open the Z16 depth stream and align it to colour, so `PersonSkeleton.position` carries the person's XYZ in the colour optical frame and `depth` the Euclidean distance, in meters. `false` = colour only, `position`/`depth` published as `NaN`. |

An unsupported width/height/fps combination fails at start-up with the device's
supported list rather than being silently substituted.

## Model

| Argument | Default | Description |
|---|---|---|
| `device` | `cuda:0` | Torch device for RTMO. |

## Tracking — BoT-SORT

| Argument | Default | Description |
|---|---|---|
| `enable_tracking` | `false` | Enable in-process BoT-SORT tracking; makes `PersonSkeleton.person_id` a persistent track ID instead of the frame-local detection index. |
| `with_reid` | `true` | Use OSNet ReID appearance features inside BoT-SORT. Only has an effect when `enable_tracking:=true`. |
| `cmc_method` | `none` | Camera-motion compensation: `none`, `ecc`, `orb`, `sift`, `sof`. |
| `proximity_thresh` | `0.70` | IoU-**distance** gate above which BoT-SORT discards the ReID distance. `iou_dist = 1 - IoU`, so `0.70` keeps appearance usable down to IoU 0.30 and **raising** this value *relaxes* the gate. |
| `track_buffer` | `90` | Frames a lost track survives **before** frame-rate scaling. BoxMOT uses `int(frame_rate / 30.0 * track_buffer)`; at `tracking_frame_rate` 55 this gives `max_time_lost = 165` frames (~3.0 s). |

`proximity_thresh` and `track_buffer` are the only two association-affecting
values that differ from the BoxMOT 19.0.0 defaults.

## Tracking — occlusion-aware extension (experimental)

Requires `enable_tracking:=true`; on its own `occlusion_aware_tracking` does
nothing. With it `false`, a stock `BotSort` is constructed and none of this
code runs.

| Argument | Default | Description |
|---|---|---|
| `occlusion_aware_tracking` | `false` | EXPERIMENTAL: classify each matched detection as `NORMAL` or `OCCLUDED`; an `OCCLUDED` one skips the Kalman measurement update and freezes the ReID feature. Prediction and the track buffer are unaffected. `false` = stock BoT-SORT behaviour. |
| `keypoint_visibility_threshold` | `0.30` | A COCO-17 joint counts as visible at or above this RTMO **per-keypoint** score. |
| `visible_ratio_threshold` | `0.50` | `visible_ratio` below this marks the detection `OCCLUDED` — the only classification input. |
| `normal_bbox_history_size` | `15` | *Informational only:* per-track ring buffer of `NORMAL` bbox widths shown in the debug log. Classifies nothing. Must be `>= 1`. |
| `min_normal_width_samples` | `5` | *Informational only:* `NORMAL` widths needed before the debug log prints a bbox width ratio at all. Must be `>= 1`. |

## Visualization

| Argument | Default | Description |
|---|---|---|
| `publish_visualization_image` | `false` | Publish the live annotated visualization image topic `/skeleton_detection/visualization_image`. |
| `visualization_width` | `424` | Width of the published visualization image. Must be `>= 1`. |
| `visualization_height` | `240` | Height of the published visualization image. Must be `>= 1`. |
| `visualization_fps` | `10.0` | Visualization publish rate; `<= 0` = every processed frame, no rate limit. Independent of the `SkeletonFrame` rate, which always runs at the full pipeline rate. |
| `visualization_reliability` | `best_effort` | QoS reliability for the visualization topic: `best_effort` or `reliable`. Anything else fails at start-up. |
| `draw_person_xyz` | `false` | DEBUG: append each person's camera-frame XYZ [m] to the overlay label. |
| `save_visualization_images` | `false` | Write an annotated file for **every** processed frame — not rate limited — to `visualization_output_dir`. |

## Runtime

| Argument | Default | Description |
|---|---|---|
| `run_duration_sec` | `0.0` | Stop automatically after N seconds; `0` = run forever. **DOUBLE** — pass `30.0`, not `30`. |
| `config` | `<share>/config/rtmo_node_realsense.yaml` | Parameter file for `rtmo_node` in realsense mode. This is the only argument that is *not* forwarded as a ROS parameter — it selects the YAML the node is loaded with. |

`<share>` is `get_package_share_directory("skeleton_detection")`, i.e.
`/ros2_ws/install/skeleton_detection/share/skeleton_detection`.

---

## Example: the full pipeline

Everything on — depth, BoT-SORT with ReID, occlusion-aware tracking and
visualization at 30 Hz — stopping on its own after 60 seconds:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true \
  with_reid:=true \
  occlusion_aware_tracking:=true \
  realsense_enable_depth:=true \
  publish_visualization_image:=true \
  visualization_fps:=30.0 \
  draw_person_xyz:=true \
  run_duration_sec:=60.0
```

View the overlay from a second shell into the container:

```bash
ros2 run rqt_image_view rqt_image_view \
  /skeleton_detection/visualization_image
```

More task-oriented recipes — tracking without ReID, running without depth,
custom visualization sizes, the offline image pipeline — are in
[Running the pipeline](running.md).
