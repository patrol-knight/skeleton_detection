# Implementation notes

Lower-level detail that does not belong in [Architecture](architecture.md):
model assets and their paths, dependency workarounds, the BoxMOT hook, and the
conventions and assumptions the code relies on.

This page complements — it does not duplicate — architecture.md, which covers
data flow and execution order. The full dependency audit, with the reasoning
behind every version pin, lives in
[`../docker/PACKAGES.md`](../docker/PACKAGES.md) and is not repeated here.

---

## Model assets and container paths

All model assets are baked into the Docker image, so a fresh `git clone` +
`docker compose build` produces a fully runnable container with **no manual
file copying and no `docker cp`**:

```text
/opt/models/
├── rtmo/
│   ├── rtmo-m.py            RTMO-M config          ← tracked in git (models/rtmo/)
│   ├── default_runtime.py   its `_base_`           ← tracked in git (models/rtmo/)
│   └── rtmo-m.pth           RTMO-M Body7 weights   ← downloaded during docker build
└── reid/
    └── osnet_x0_25_msmt17.pt  OSNet ReID weights   ← downloaded during docker build
```

The **configs live in git**; the **weights do not** (`.gitignore` blocks
`*.pth` / `*.pt`). Both end up in an image layer, so deleting and recreating the
container never loses them.

Model files used to live in `data/checkpoints/`, which is both gitignored and
dockerignored — so `git clone` + `docker pull` produced an environment with
**no weights**. That is the problem baking them into the image solves.

### Sources and checksums

| Asset | Source | sha256 |
|---|---|---|
| `rtmo-m.pth` | `download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth` | `39e78cc4afed7d1dd31ea73ee04842f7d8f8e695d900d41b90746e94728a01c3` |
| `osnet_x0_25_msmt17.pt` | BoxMOT's own model registry URL, fetched with BoxMOT's downloader | `6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99` |

Checkpoint sources are **official only**. Both are sha256-verified at build
time, so a corrupted or substituted file fails `docker build` instead of
silently shipping.

**Why the ReID weights are fetched at build time.** BoxMOT would otherwise
download them on first use at runtime, into whatever directory the process
happens to use — not reproducible, and a failure on an offline robot.

### Docker helper files

```text
docker/
├── Dockerfile
├── fetch_models.py
├── verify_image.py
├── mmcv_ext_stub.py
└── PACKAGES.md
```

`fetch_models.py` downloads and checksum-verifies the two checkpoints, and
asserts that the git-tracked RTMO config was copied in and loads with its
`_base_` chain resolved. It deliberately does **not** generate the config: a
missing or broken config fails `docker build`, not node start-up.

`verify_image.py` is a build-time regression gate for the whole
Python/CUDA/OpenMMLab/BoxMOT environment. It fails the build if numpy, torch,
torchvision, the OpenMMLab versions, the RTMO import chain, pyrealsense2,
boxmot 19.0.0, the BoT-SORT/ReID imports, or any baked asset is missing or
displaced.

### Environment variables

The image exports `RTMO_MODEL_CONFIG`, `RTMO_CHECKPOINT` and
`REID_CHECKPOINT`, which are also the node's parameter defaults. To run a
different config/checkpoint without rebuilding, override the `model_config` /
`checkpoint` / `reid_checkpoint` ROS parameters, or set the environment
variables before starting the node.

---

## The RTMO config and its `_base_` chain

The config is version-controlled alongside the source:

```text
models/
└── rtmo/
    ├── rtmo-m.py           MMPose RTMO-M (Body7, 640×640) config
    └── default_runtime.py  the `_base_` rtmo-m.py inherits from
```

Both are vendored from `open-mmlab/mmpose` v1.3.2. Upstream, `rtmo-m.py`
declares `_base_ = ['../../../_base_/default_runtime.py']`, which only resolves
inside the MMPose config tree. **The sole edit made here is to point that at
the sibling `./default_runtime.py`.**

That base file is not optional — it supplies `default_scope = 'mmpose'`,
without which `init_model` cannot resolve the model registry.

The upstream config is at
`mmpose/.mim/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py`.

---

## MMPose / MMCV workarounds

### `mmcv._ext` is a stub

`mmcv-lite` ships no compiled ops, but `mmpose.models.heads.__init__`
unconditionally imports `transformer_heads` → `edpose_head.py`, which does
`from mmcv.ops import MultiScaleDeformableAttention`. That pulls in all of
`mmcv.ops`, which dies on the absent compiled `mmcv._ext`.

RTMO itself uses **no** mmcv native op — only `mmcv.cnn.ConvModule` / `Scale`
and mmpose's pure-PyTorch `nms_torch` — so `docker/mmcv_ext_stub.py` provides
an import-only `mmcv._ext` that **raises if an op is ever actually called**.

The node treats that as fatal rather than publishing bad results: both
`_inference_loop` and `_on_image` catch `NotImplementedError` and log

```text
RTMO hit a stubbed mmcv native op: ... A full mmcv build is required;
refusing to publish results.
```

Full `mmcv` 2.x does not build from source against torch 2.14 (C++ API drift).
Replace the stub with a full mmcv build if real ops are ever needed.

Version constraint worth knowing: `mmcv-lite` is pinned to **2.1.0, not
2.2.0**, because `mmdet/__init__.py` asserts `mmcv < 2.2.0` — and mmdet is
required, since RTMO's head does `from mmdet.utils import ConfigType,
reduce_mean`.

### Checkpoint loading under torch ≥ 2.6

torch 2.6 flipped the `torch.load` `weights_only` default to `True`. mmengine
0.10.7's local checkpoint loader never passes the argument, and the RTMO
checkpoint stores numpy objects the restricted unpickler cannot rebuild
(`add_safe_globals` is not sufficient).

`rtmo_inference.install_checkpoint_loader_workaround()` overrides the
registered scheme through mmengine's **public** API rather than patching
site-packages:

```python
@CheckpointLoader.register_scheme(prefixes="", force=True)
def _load_from_local_trusted(filename, map_location):
    ...
    return torch.load(filename, map_location=map_location, weights_only=False)
```

Any future code that loads this checkpoint needs the same override.

---

## ReID model details

- Backbone: **OSNet `osnet_x0_25`**, MSMT17 weights (3.06 MB).
- Built once in `SkeletonTracker.__init__` when `with_reid=True`; never per
  frame. The load time is recorded as `reid_load_seconds`.
- Runs on the same `device` as RTMO (`cuda:0` by default), in full precision
  (`half=False`).
- BoT-SORT crops each detection out of the frame itself and calls
  `model.get_features(boxes, img)`, which is why `SkeletonTracker.update()`
  takes the BGR frame.
- ReID is the largest single throughput cost in the pipeline; motion-only
  BoT-SORT adds very little.

---

## How the occlusion extension hooks into BoxMOT

**Nothing under `/usr/local/lib/python3.10/dist-packages/boxmot` is modified.**

BoxMOT builds every `STrack` through the module-level name
`boxmot.trackers.botsort.botsort.STrack` (in `_create_detections` and in
`_second_association`), and a track *is* the detection `STrack` that
`activate()` promoted. Replacing that one module global for the duration of one
`_update_impl` call is therefore enough to make every track and every detection
an `OcclusionAwareSTrack`.

Only `STrack.update`, `STrack.re_activate` and `STrack.activate` are
overridden. The association code — IoU distance, the proximity gate, the ReID
distance, the appearance gate, the Hungarian assignment, the track buffer and
the four stages — is untouched.

`make_occlusion_aware_botsort(BotSort, params, logger)` produces the subclass,
and it is only imported when `occlusion_aware_tracking` is true. With the
switch off, `SkeletonTracker` constructs a stock `BotSort` and
`occlusion_tracking.py` never runs.

Removing the feature entirely means deleting
`inference/occlusion_tracking.py` and the parameters marked
`OCCLUSION-AWARE TRACKING (delete with occlusion_tracking.py)` in
`iot_node.py`, the launch file and the YAML.

---

## Internal data structures

| Type | Module | Purpose |
|---|---|---|
| `PersonDetection` | `inference/rtmo_inference.py` | one detected person before any ROS conversion; carries `bbox_xyxy`, `score`, `keypoints_xy`, `keypoint_scores`, `detection_index`, `track_id`, `position` |
| `CapturedFrame` | `input/realsense_capture.py` | one frame handed from the capture thread: BGR image, ROS-clock ns, sequence, hardware timestamp, aligned Z16 depth |
| `CaptureStats` | `input/realsense_capture.py` | `captured` / `dropped` / `errors` and capture FPS |
| `CameraIntrinsics` | `inference/depth_estimation.py` | frozen `fx, fy, cx, cy, width, height` — plain numbers, so the module never imports pyrealsense2 |
| `CameraPoint` | `inference/depth_estimation.py` | one 3D point in meters; `distance` is the Euclidean norm |
| `DepthParams` | `inference/depth_estimation.py` | `depth_scale`, `keypoint_score_threshold`, `window` (3), `min_depth_m`, `max_depth_m` |
| `OcclusionParams` | `inference/occlusion_tracking.py` | mirrored one-for-one by ROS parameters |
| `DetectionVisibility` | `inference/occlusion_tracking.py` | frozen result of reading one detection's keypoint confidences; `available=False` means the visible-ratio criterion is skipped |
| `PipelineStats` / `_Series` | `utils/pipeline_stats.py` | per-frame millisecond samples with mean/median/p95 helpers |

`CameraIntrinsics` is built from `rs.intrinsics` with `cx = ppx` and
`cy = ppy`.

---

## Coordinate conventions

| Stage | Convention |
|---|---|
| RTMO output, tracking input | **`xyxy`** `[x1, y1, x2, y2]`, original source image pixels, unclipped |
| Published `PersonSkeleton.bbox` | **`[x, y, width, height]`** (top-left + size), same pixel space, unclipped |
| `keypoints_xy` | COCO-17 in original image pixels |
| Published `joints` | 51 floats, `[x, y, conf] × 17` |
| Published `connections` | 38 ints, the 19 COCO edges flattened |
| `position` | meters, **colour camera optical frame**: x → image right, y → image down, z → forward along the optical axis |
| `depth` | meters, **Euclidean** `sqrt(x²+y²+z²)` — *not* the RealSense Z, which is `position.z` |
| Depth image | raw Z16 units, aligned to colour; metric conversion via the device `depth_scale` |

The conversion from `xyxy` to `[x, y, w, h]` happens in exactly one place,
`output/message_builder.py`.

---

## COCO-17 topology

`utils/coco_keypoints.py` holds the shared constants:

- `NUM_COCO_KEYPOINTS = 17`
- `COCO_KEYPOINT_NAMES` — `nose`, `left_eye`, `right_eye`, `left_ear`,
  `right_ear`, `left_shoulder`, `right_shoulder`, `left_elbow`, `right_elbow`,
  `left_wrist`, `right_wrist`, `left_hip`, `right_hip`, `left_knee`,
  `right_knee`, `left_ankle`, `right_ankle`
- `COCO_CONNECTIONS` — the fixed **19** skeleton edges
- `COCO_CONNECTIONS_FLAT` — the same 19 edges flattened to 38 ints, which is
  exactly what `PersonSkeleton.connections` publishes

`RTMOInference._verify_keypoint_layout()` checks the model's keypoint layout
against these names at start-up, so a config with a different skeleton fails
loudly instead of producing silently mislabelled joints.

---

## Special assumptions

- **The depth image reaching `depth_estimation` is already aligned to colour.**
  The module never aligns anything and never indexes a raw depth frame with
  colour coordinates.
- **The intrinsics passed to `depth_estimation` are the colour stream's**, not
  the native depth sensor's.
- **Deprojection is pure pinhole.** Colour lens distortion is ignored; the
  model and coefficients are logged at start-up so the assumption is checkable
  on the real device.
- **The lateral anchor is the bbox centre**, not a body-specific anchor such as
  the torso centroid.
- **Unavailable means `NaN`, never `0`.** Consumers must use `math.isnan()`.
- **`camera_frame_id` must name the colour optical frame.** The node logs a
  warning at start-up if it does not end in `color_optical_frame`, because
  `position` is always expressed in that frame regardless of what the parameter
  says.
- **`depth_scale` is applied exactly once**, inside the depth module. The
  capture thread hands over raw Z16.
- **The hardware timestamp is diagnostics only.** It lives in the camera's
  clock domain and is deliberately never mixed into the ROS header, which uses
  the ROS clock sampled right after the frame arrives.
- **`bgr8` capture avoids a colour conversion.** The RTMO config uses
  `mean=[0,0,0]`, `std=[1,1,1]` and no `bgr_to_rgb`, and the D456 advertises
  `bgr8` natively at 848 × 480 @ 60, so the captured array goes to MMPose
  unconverted.
- **`setuptools` must stay below 80** — colcon's `setup.py` build path breaks
  on 80+. The image pins `79.0.1`.

---

## Measured performance

Typical behaviour on the DGX Spark:

| Configuration | Rate |
|---|---|
| RealSense capture (raw) | ~60 FPS (measured 59.8) |
| RTMO only | ~50–55 FPS |
| RTMO + BoT-SORT, ReID off | ~50 FPS with people |
| RTMO + BoT-SORT + ReID | ~38 FPS with 1–2 people |
| Visualization | 10 / 30 / 50 Hz all run successfully at 424 × 240 |

The largest additional cost is OSNet ReID. Motion-only BoT-SORT adds very
little. When processing falls behind the 60 FPS camera the latest-frame-wins
buffer drops stale frames rather than building latency.

---

## Current limitations

- Real-person tracking and occlusion behaviour still need more live validation.
- ReID reduces throughput to roughly 38–40 FPS with people present.
- `tracking_frame_rate` is fixed when BoT-SORT is constructed and does not
  adapt to the measured runtime FPS.
- `cmc_method: none` assumes a static-camera baseline.
- `position` is camera-optical-frame only; no TF transform to `base_link`,
  `odom` or `map` is applied yet.
- Deprojection is pinhole and the lateral anchor is the bbox centre.
- No TensorRT optimization is used.
