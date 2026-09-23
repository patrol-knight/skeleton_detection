# Architecture

Module responsibilities, data flow and execution order of the live pipeline.

The root [README](../README.md) has the high-level diagram; this page is the
implementation-level view. Deeper environment and model notes are in
[Implementation notes](implementation.md); every parameter mentioned here is
tabulated in [Parameters](parameters.md).

---

## The single-process design

The live perception path runs inside **one ROS node in one Python process**.
In the default `input_mode: realsense` the RealSense colour frame is handed
from the capture thread through RTMO and into BoT-SORT **by reference** — it is
never serialized, never copied between processes and never published on a ROS
image topic before inference.

That is the whole point of the design: 848 × 480 @ 60 fps `sensor_msgs/Image`
frames are ~1.2 MB each, ~70 MB/s, and pushing them between two processes over
Fast DDS is a bottleneck in this environment.

**DDS starts only at the outputs**: `/skeleton_detection/frame` and, when
enabled, `/skeleton_detection/visualization_image`.

`input_mode: ros_camera` deliberately trades that property away: frames arrive
over DDS from an external `realsense2_camera` driver, so the measured
bottleneck above applies to it and throughput is expected to be lower than the
direct path. It exists for deployments where the camera is owned by another
container, not as a replacement for the default. `input_mode: ros_topic` is
the offline/regression path and carries colour only.

---

## Input backends

Three backends, one seam. `RTMONode._start_input` is the **only** place
`input_mode` is branched on; each backend ends up calling
`RTMONode._process_frame(frame_bgr, header, depth_image)`, so nothing
downstream knows where a frame came from.

| Mode | Module | Colour | Depth | Intrinsics |
|---|---|---|---|---|
| `realsense` | `input/realsense_capture.py` | `pyrealsense2`, capture thread, latest-frame-wins | Z16, `rs.align` in the capture loop | colour stream profile, read once at `start()` |
| `ros_topic` | `iot_node._on_image` | `sensor_msgs/Image` + `CvBridge` | none → `NaN` | none → `NaN` |
| `ros_camera` | `input/ros_camera_subscriber.py` | `RGBD.rgb` + `CvBridge` | `RGBD.depth`, aligned **by the driver** | `RGBD.rgb_camera_info.k`, read once from the first message |

```text
realsense  : D456 --pyrealsense2--> capture thread --rs.align--> |
ros_topic  : /dummy_camera/image_raw --CvBridge--------------->  | _process_frame
ros_camera : external realsense2_camera --RGBD--CvBridge----->   |
```

### `ros_camera` and the alignment contract

`inference/depth_estimation.py` requires a depth image on the **colour** pixel
grid, with **colour** intrinsics. In `realsense` mode this package earns that
itself with `rs.align(rs.stream.color)`. In `ros_camera` mode the external
driver earns it with `align_depth.enable:=true`, and `enable_sync:=true` pairs
colour with depth upstream — so this package performs no alignment and no
synchronization, and never calls `rs.align`.

Because `realsense2_camera_msgs/msg/RGBD` carries colour, aligned depth and
both `CameraInfo`s in **one** message, there is no `message_filters`
synchronizer and no chance of a colour frame being matched to the wrong depth
frame. That is the reason this mode subscribes to one RGBD topic rather than
three independent topics.

Two differences from the direct path are worth knowing:

- **Depth scale comes from the encoding, not the device.** `16UC1` is
  millimetres → `0.001`; `32FC1` is metres → `1.0`; anything else is refused
  rather than guessed. The direct path instead reads `depth_scale` from the
  depth sensor.
- **QoS must be `best_effort`.** The driver publishes with sensor-data QoS; a
  `reliable` subscription would never match it. The queue depth is `1`, which
  reproduces the latest-frame-wins behaviour of the capture buffer by letting
  DDS drop stale frames instead of queueing them.

A driver started without `align_depth.enable` is caught **once**, at startup,
by comparing the depth image size against the colour intrinsics — rather than
raising once per person per frame inside the depth module.

---

## Execution order

`RTMONode._process_frame` runs these steps strictly in sequence, once per
frame:

```text
1. RTMO inference            inference.rtmo_inference.RTMOInference.infer()
2. tracking (optional)       inference.person_tracking.SkeletonTracker.update()
3. depth / XYZ               inference.depth_estimation.compute_person_position()
4. message construction      output.message_builder.build_skeleton_frame()
5. publish                   /skeleton_detection/frame
6. visualization (optional)  output.visualization.draw_skeleton_overlay()
7. statistics                utils.pipeline_stats.PipelineStats.record_frame()
```

Two ordering decisions are load-bearing:

- **Tracking runs before the message is built**, so `person_id` is already the
  persistent track ID at publication time. No second matching pass is needed.
- **Depth runs after tracking and before the message is built**, so
  `person_id` and `position`/`depth` in the published message describe the same
  detection. `_attach_person_position` never modifies boxes, IDs or keypoints.

The `SkeletonFrame` is published **before** any visualization work happens, so
nothing on the visualization path can throttle the robot-facing topic.

---

## Module responsibilities

| Module | Owns | Knows about ROS? |
|---|---|---|
| `iot_node.py` | parameters, publishers, threads, lifecycle, orchestration | yes — the only ROS surface besides the message builder |
| `input/realsense_capture.py` | `rs.pipeline`, depth→colour alignment, latest-frame-wins buffer, intrinsics, depth scale | no |
| `input/image_publisher.py` | offline image → `sensor_msgs/Image` | yes (separate node) |
| `inference/rtmo_inference.py` | RTMO-M model, checkpoint loading, result parsing | no |
| `inference/person_tracking.py` | BoT-SORT instance + OSNet ReID model | no |
| `inference/occlusion_tracking.py` | occlusion classification + `STrack` overrides | no |
| `inference/depth_estimation.py` | per-person Z and XYZ deprojection (pure NumPy) | no |
| `output/message_builder.py` | internal → ROS message conversion | yes |
| `output/visualization.py` | all drawing, plus the file-writing sink | no (duck-typed persons) |
| `utils/pipeline_stats.py` | timing and throughput accounting | no |
| `utils/coco_keypoints.py` | COCO-17 names and the 19-edge topology | no |

Everything except `iot_node.py`, `message_builder.py` and `image_publisher.py`
is ROS-free, which is why the depth, occlusion and tracking logic is testable
on plain NumPy arrays.

---

## Frame ownership and buffering

### Threading model

Two threads, in `input_mode: realsense`:

- **Capture thread** (`RealSenseCapture._capture_loop`) owns the
  `rs.pipeline` and does nothing else: pull a frameset, align it, copy the
  arrays out, drop them into a **single-slot** buffer.
- **Inference worker** (`RTMONode._inference_loop`, named `rtmo_inference`)
  takes whatever is in that slot and runs the whole pipeline on it.

The ROS executor thread runs the timers (statistics, `run_duration_sec`) and,
in `input_mode: ros_topic`, the image subscription callback — on that path
there is no capture thread and no worker.

### Latest-frame-wins

There is no queue and no backlog. If the consumer is slower than the camera,
the newest frame **overwrites** the previous unconsumed one and the overwritten
frame is counted as `dropped` in `CaptureStats`.

Latency therefore stays bounded at roughly one camera period plus one
inference, instead of growing without limit. It also means the tracker sees the
~40–55 FPS the pipeline actually *processes*, not the 60 FPS the camera
produces — which is why `tracking_frame_rate` defaults to the processed rate
(55) rather than the camera rate.

`CapturedFrame` carries the BGR image, a ROS-clock capture timestamp, a
monotonic sequence number, the RealSense hardware timestamp (diagnostics only —
it lives in the camera's clock domain and is deliberately never mixed into the
ROS header) and the aligned depth image.

---

## RealSense RGB/depth alignment

When `realsense_enable_depth` is true, a Z16 depth stream is opened alongside
the colour stream and every frameset is pushed through
`rs.align(rs.stream.color)` **in the capture loop**, before the arrays are
copied out.

**This is the only place in the package where alignment happens.**

It has to happen because the D4xx depth and colour sensors sit at different
positions on the module and have different intrinsics: the raw depth frame is
in the depth sensor's own pixel grid. RTMO runs on the *colour* image, so its
keypoints are colour pixels; indexing an unaligned depth frame with them would
read the wrong part of the scene. After `rs.align`, `depth[v, u]` is the
distance at colour pixel `(u, v)`.

Two consequences:

- Each aligned depth value is the optical-axis **Z in the colour camera
  frame**, so the matching intrinsics for deprojection are the **colour**
  stream's. They are read once in `RealSenseCapture.start()` and exposed as
  `camera_intrinsics`.
- The array is carried in **raw Z16 units**. The device depth scale (meters per
  unit, ~0.001 on a D4xx) is read once in `start()` and applied inside the
  depth module, so nothing is ever scaled twice.

The requested profile is validated against the device's advertised profiles at
start-up; an unsupported combination fails with the supported list rather than
being silently substituted.

---

## RTMO detection representation

`RTMOInference.infer()` takes a BGR frame and returns a list of
`PersonDetection`:

| Field | Shape / type | Notes |
|---|---|---|
| `bbox_xyxy` | `(4,)` float32 | `[x1, y1, x2, y2]` in original image pixels, **not clipped** |
| `score` | float | person/instance confidence in `[0, 1]` |
| `keypoints_xy` | `(17, 2)` float32 | COCO keypoints in image pixels |
| `keypoint_scores` | `(17,)` float32 | **per-joint** confidence |
| `detection_index` | int | position in this frame's list; what BoxMOT echoes back as `det_ind` |
| `track_id` | `int` or `None` | filled by the tracker; `None` = not tracked |
| `position` | `CameraPoint` | filled by the depth stage; defaults to the all-`NaN` `NO_POSITION` |

Two derived properties:

- `person_id` → `track_id` when tracked, else `detection_index`.
- `depth` → `position.distance`, i.e. `sqrt(x² + y² + z²)`. It is never stored
  separately, so it cannot drift out of sync with `position`.

**Everything stays in `xyxy`, in the original source image coordinate system.**
MMPose's bottom-up estimator already maps predictions out of the padded
640 × 640 model input back into image space, and BoT-SORT also wants `xyxy`, so
no conversion happens until `message_builder` turns the detection into a ROS
message — where the published convention is `[x, y, width, height]`.

Detections below `person_score_threshold` are dropped and never published.

---

## BoT-SORT tracking and ReID

`SkeletonTracker` owns exactly one `BotSort` instance and, when
`with_reid` is true, one OSNet ReID model. **Both are built once in
`__init__`; nothing is constructed per frame.**

### The detection contract

Verified against BoxMOT 19.0.0 `boxmot/trackers/detection_layout.py`:

```text
in : (N, 6) float32  [x1, y1, x2, y2, conf, cls]
out: (M, 8) float32  [x1, y1, x2, y2, track_id, conf, cls, det_ind]
```

`det_ind` indexes back into the array that was passed in, so a returned track
maps to the exact detection — and therefore the exact 17 keypoints — that
produced it. **No second IoU matching step is needed or performed.** Every RTMO
detection uses `cls = 0`, BoxMOT's person class.

### Why the tracker gets the frame

With `with_reid=True`, BoT-SORT crops each detection out of the frame itself
and runs the ReID backbone on those crops
(`model.get_features(boxes, img)`). That is why `update()` takes the image: it
is an **appearance-model input, not an image transport**. The crops come from
the same in-memory BGR array RTMO ran on.

### Association

Stock BoxMOT: IoU distance, the proximity gate, the ReID distance, the
appearance gate, the Hungarian assignment, the track buffer and the four
association stages are all untouched. Only two thresholds differ from the
BoxMOT defaults — `track_buffer` (90, not 30) and `proximity_thresh` (0.7, not
0.5). Both are explained in [Parameters](parameters.md).

`cmc_method: none` disables camera-motion compensation, which is the right
baseline for the current static camera. For a moving or robot-mounted camera it
should be re-evaluated. BoxMOT accepts Python `None` to disable CMC but rejects
the string `"none"`, so the node maps the friendly parameter value.

---

## Occlusion-aware extension (optional)

An **extension of the normal tracking path**, not a parallel one. It is off by
default; with `occlusion_aware_tracking: false` a stock `BotSort` is
constructed and the per-frame cost is exactly zero.

### The failure mode it addresses

A stationary person is slowly occluded by someone walking past. RTMO keeps
emitting a person box, but it shrinks onto whatever is still visible — a leg,
half a torso — so its centre and width drift sideways. BoT-SORT feeds those
partial boxes to the Kalman filter as ordinary measurements, teaching the motion
model a velocity the person never had, and crops the ReID embedding from a
half-body (or from the occluder), poisoning `smooth_feat`. By the time the
person is fully hidden the tracker is predicting away from them; when they
reappear the IoU is near zero and the appearance distance is large, so a new ID
is allocated.

### Classification

Every matched observation is classified into exactly two states, from exactly
**one** signal:

```text
is_occluded = visibility.available
              and visible_ratio < visible_ratio_threshold

visible_ratio = (# RTMO keypoints with score >= keypoint_visibility_threshold)
                / (# keypoints)
```

`visible_ratio` uses RTMO's **per-joint** `keypoint_scores`, not the aggregate
person score. Only finite scores are counted, in both the numerator and the
denominator.

### What each state does

| | `NORMAL` | `OCCLUDED` |
|---|---|---|
| Kalman prediction | runs | **runs** (unchanged) |
| Kalman measurement update | applied | **skipped** |
| Velocity `mean[4:8]` | updated | **preserved, never reset or zeroed** |
| ReID `curr_feat` / `smooth_feat` / history | updated | **frozen** |
| `NORMAL` bbox-width history | appended | **not appended** |
| Track lifetime (`frame_id`, `tracklet_len`, `state`, `conf`, `cls`, `det_ind`) | updated | updated |

The detection still receives its track ID in **both** states, so `person_id` is
unaffected. No association logic changes.

### bbox width is not a signal

An earlier version also marked a detection `OCCLUDED` when its width fell below
`0.60` of the track's recent median. That misfired whenever somebody simply
turned sideways — 16/17 keypoints visible, nothing occluding them, box down to
~0.45 of its frontal width. A person's own pose changes their box width as much
as an occluder does, so width cannot tell the two apart.

The per-track width history is still maintained and still printed in the debug
log, labelled `classification use: INFORMATIONAL ONLY`. It has zero effect on
classification or on any tracker state, and `normal_bbox_history_size` /
`min_normal_width_samples` only size that log output.

### Edge cases

- If RTMO's keypoint scores are missing or non-finite, the visible-ratio
  criterion is **skipped** (the detection stays `NORMAL`) and a throttled
  warning is logged. Missing keypoints are never read as zero confidence; the
  detection carries `note=PARTIAL_KEYPOINT_SCORES` so the caller can log it
  instead of quietly shrinking the denominator.
- A `LOST` track re-activated on an `OCCLUDED` detection gets the same
  protection: it is re-activated (ID preserved, `new_id=False`) but the partial
  box does not move the filter and does not update the appearance feature.

How it hooks into BoxMOT without patching site-packages is described in
[Implementation notes](implementation.md).

---

## Depth and XYZ computation

Implemented in `inference/depth_estimation.py`, which is pure NumPy — it knows
nothing about ROS, pyrealsense2 or `PersonDetection`, and is covered by
`test/test_person_depth.py` and `test/test_person_position.py`.

### Person Z — a two-stage median

```text
for every VISIBLE keypoint (joint score >= depth_keypoint_score_threshold):
    3x3 depth neighbourhood around (u, v)
        drop zero / NaN / inf samples
        median  ->  joint Z

drop keypoints with no valid joint Z

median of the remaining joint Z values  ->  person Z
```

**The median is used at both stages on purpose and must not be replaced by a
mean.** Depth around a person is bimodal — a pixel is either on the person or
on the background behind them — and a mean would blend the two and place the
person in the empty space between them. A median picks an actually-observed
surface.

`depth_min_m` / `depth_max_m` apply an optional metric sanity range; `<= 0`
disables that bound. Zero, `NaN` and infinite samples are rejected regardless.

### Position — bbox centre deprojection

The person is placed laterally at the centre of its bounding box (clipped to
the image, since RTMO boxes are unclipped) and deprojected with the **colour**
intrinsics:

```text
(u, v) = centre of the clipped bbox, colour-image pixels
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = person Z
depth = sqrt(X² + Y² + Z²)
```

The colour intrinsics are the right ones because the keypoints and bbox are
colour pixels and the aligned depth values are Z in the colour camera. Lens
distortion is ignored (pure pinhole); the node logs the distortion model and
coefficients at start-up so the assumption can be checked on the actual device.

### Z-depth vs Euclidean distance — not the same thing

```text
RealSense depth image value     = optical-axis Z coordinate
published PersonSkeleton.depth  = Euclidean distance = sqrt(X² + Y² + Z²)
```

They agree only near the image centre; towards the edge of the image the
Euclidean distance is noticeably larger than Z. In the module, anything named
`z` is the optical-axis coordinate and `CameraPoint.distance` is the Euclidean
distance.

### Unavailability

If no visible keypoint yields a valid depth — no depth stream, the person is
out of range, or the depth is all holes — `position` and `depth` are `NaN`,
**never `0`**. Consumers must test with `math.isnan()`; the visualization
prints `Depth: N/A`.

The position is **camera-frame only**. No transform to `base_link`, `odom` or
`map` is applied. In `input_mode: ros_topic` there is no depth stream and no
intrinsics, so `position`/`depth` are always `NaN` on that path.

---

## ROS message construction

`output/message_builder.py` is the single owner of the internal → ROS
conversion, so the wire conventions live in exactly one place:

| Field | Conversion |
|---|---|
| `bbox` | internal `xyxy` → published `[x, y, width, height]` (top-left + size), original source pixels, unclipped |
| `joints` | 17 COCO keypoints flattened to 51 floats, `[x0, y0, conf0, ...]` |
| `connections` | the fixed 19-edge COCO topology flattened to 38 ints |
| `person_id` | persistent track ID when tracked, else the frame-local index |
| `position` | `geometry_msgs/Point`, meters, colour optical frame |
| `depth` | `position.distance`, Euclidean meters |

`SkeletonFrame` adds the header (whose `frame_id` is `camera_frame_id`), the
`frame_index` counter and a `timestamp` derived from the header stamp.

`person_id_semantics()` returns the one-line description of what `person_id`
currently means; it is logged at start-up so a log reader is never in doubt.

### QoS

| Topic | Reliability | History | Depth | Durability |
|---|---|---|---|---|
| `/skeleton_detection/frame` | reliable | keep-last | 10 | volatile |
| `/skeleton_detection/visualization_image` | `visualization_reliability` (default `best_effort`) | keep-last | **1** | volatile |

Depth 1 on the visualization topic means a slow viewer can never make stale
frames pile up.

---

## Visualization path

`output/visualization.py` is the single drawing implementation in the package.
`iot_node` renders **at most once per frame, and only when a sink needs it**:

```python
due = visualization_publisher is not None and
      (now - last_visualization_time >= visualization_period)
if not due and visualization_writer is None:
    return
```

At ~55 Hz inference and 10 Hz visualization, roughly 4 of every 5 frames skip
drawing entirely.

The overlay is drawn at the **original** resolution — so RTMO coordinates are
used as-is with nothing rescaled — and any downscale to
`visualization_width` × `visualization_height` happens afterwards on the
finished overlay, with `INTER_AREA` when shrinking and `INTER_LINEAR`
otherwise.

The rate limiter advances the schedule by exactly one period rather than
resetting to "now": visualization can only be emitted on a frame boundary, so
resetting would always round the interval up and run systematically slow.

The same overlay feeds both sinks — the published topic (downscaled) and
`VisualizationWriter` (full resolution) — so the file you open and the live
topic show exactly the same annotations. The drawing functions take duck-typed
person objects (anything exposing `person_id`, `score`, `bbox`, `joints`, and
optionally `depth`/`position`), which is why no ROS message import is needed
there.

`legend_for(tracking_enabled, with_reid)` picks the footer text, so a reviewer
looking at a saved file alone knows whether the drawn ID is a persistent track
ID or a frame-local index.

`save_visualization_images` writes a file for **every** processed frame and is
**not** rate limited, which costs throughput; the node logs a warning when it is
enabled in realsense mode.

---

## Statistics flow

`utils/pipeline_stats.PipelineStats` keeps the three per-frame costs separate
on purpose:

```text
rtmo_inference_ms   RTMO-M forward + result parsing
tracking_ms         BoT-SORT association (+ ReID embedding when enabled)
total_processing_ms inference + tracking + message build + publish
```

Merging tracking into inference would hide exactly the number this work exists
to measure, so they are never summed into one field.

Frame-drop accounting lives separately in `CaptureStats`: a frame is *dropped*
when a newer camera frame replaced it in the latest-frame-wins slot before the
inference loop consumed it.

Two sinks:

- A periodic line every `stats_log_period_sec`, with `capture_fps`,
  `skeleton_fps`, `vis_fps`, `rtmo_ms`, `track_ms`, active tracks, and the
  `captured` / `processed` / `dropped` counters.
- A final summary on shutdown (`log_final_summary`) with mean / median / p95 /
  min / max for each series. The first inference is reported separately as the
  CUDA warm-up cost.

---

## Shutdown

`shutdown_pipeline()` is idempotent and safe from any thread: it sets the stop
event, joins the inference worker with a 5 s timeout, stops the capture, and
prints the final summary. It is called from `destroy_node()`, and the summary
falls back to `print()` when `rclpy` is already torn down — Ctrl+C destroys the
context before `destroy_node` runs.

`run_duration_sec > 0` installs a timer that raises `SystemExit`, which
`main()` catches so the node still shuts down cleanly and prints its summary.
