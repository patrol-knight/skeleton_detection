# Docker Dependency Notes

## Purpose

This file documents **what is installed in the Docker image, why, and why at
these exact versions**. It is the reference to read before changing a pin in
[`Dockerfile`](Dockerfile).

It deliberately does **not** describe how the subsystem works. For that, see:

- [`../docs/architecture.md`](../docs/architecture.md) — module
  responsibilities, data flow, execution order
- [`../docs/implementation.md`](../docs/implementation.md) — model assets,
  coordinate conventions, internal data structures

Every version below was read from the current [`Dockerfile`](Dockerfile),
[`fetch_models.py`](fetch_models.py), [`verify_image.py`](verify_image.py) or
the running container. Nothing here is aspirational.

**Host the image is built and validated on:** NVIDIA DGX Spark, GB10
(compute capability 12.1, 124610 MiB), `aarch64`, driver 580.173.02, CUDA 13.0.

---

## Base image / OS

| | |
|---|---|
| Base image | `ros:humble-ros-base` |
| OS | Ubuntu 22.04.5 LTS (jammy) |
| Python | 3.10.12 (system interpreter) |
| Architecture | `aarch64` |

Python 3.10 is not a choice — it is what jammy and ROS 2 Humble ship, and every
pin below has to have an `aarch64` + cp310 wheel or build from source in the
image.

`WORKDIR` is `/ros2_ws`, which is also the colcon workspace root that
`compose.yaml` mounts the repository into.

---

## ROS 2

ROS 2 **Humble**, from the base image. Additional apt packages installed by the
Dockerfile:

| Package | Installed version | Why it is needed |
|---|---|---|
| `ros-humble-cv-bridge` | 3.2.1-1jammy | `CvBridge` in `iot_node.py` and `input/image_publisher.py` — converts between `sensor_msgs/Image` and OpenCV arrays |
| `ros-humble-vision-opencv` | 3.2.1-1jammy | metapackage carrying `cv_bridge` |
| `ros-humble-sensor-msgs` | 4.9.2-1jammy | `sensor_msgs/Image`, on the visualization output and the offline input path |
| `ros-humble-geometry-msgs` | 4.9.2-1jammy | `geometry_msgs/Point` — the `position` field of `PersonSkeleton.msg` |
| `ros-humble-realsense2-camera` | 4.58.3-1jammy | **pulled in for its `librealsense2` dependency, not for the node.** The camera is opened in-process; no `realsense2_camera` node is ever launched |
| `ros-humble-rqt-image-view` | 1.2.0-2jammy | viewing `/skeleton_detection/visualization_image` |
| `ros-humble-rqt-gui` | 1.1.9-1jammy | rqt shell for the above |

Non-ROS apt packages: `python3-pip`, `python3-dev`, `build-essential`,
`ca-certificates`, `curl`, `gnupg` (build tooling), and `ffmpeg`, `libsm6`,
`libxext6` (OpenCV runtime shared libraries).

The package's own ROS dependencies are declared in `package.xml` (`rclpy`,
`rcl_interfaces`, `sensor_msgs`, `cv_bridge`, `ament_index_python`,
`geometry_msgs`, `std_msgs`) and are satisfied by the base image plus the above.

---

## NVIDIA / CUDA stack

Installed from
`developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/`:

| Package | Version |
|---|---|
| `cuda-nvcc-13-0` | 13.0.88-1 |
| `cuda-cudart-dev-13-0` | 13.0.96-1 |

`CUDA_HOME=/usr/local/cuda` and `/usr/local/cuda/bin` is on `PATH`.

**These are not required to run inference.** The `+cu130` torch wheels bundle
their own CUDA runtime through the `nvidia-*` pip packages. They are installed
so `nvcc` and `CUDA_HOME` exist for a future TensorRT build or a full mmcv
compilation. Removing them would shrink the image but close off that path.

GPU access at runtime comes from the NVIDIA Container Toolkit on the host via
the `deploy.resources.reservations.devices` block in `compose.yaml`.

---

## PyTorch

| Package | Version | Index |
|---|---|---|
| `torch` | 2.14.0+cu130 | `download.pytorch.org/whl/cu130` |
| `torchvision` | 0.29.0+cu130 | `download.pytorch.org/whl/cu130` |

**Why this index and version.** It is the only wheel index that publishes
`aarch64` + CUDA 13 builds, and these run on the GB10's `sm_121`. A default
`pip install torch` from PyPI does not give a CUDA-13 aarch64 build.
`torchvision` 0.29.0 is the release matched to torch 2.14.0.

Used by: RTMO-M inference (`inference/rtmo_inference.py`) and the OSNet ReID
backbone (`inference/person_tracking.py`). Both default to `cuda:0`.

### Checkpoint-loading workaround (still current)

torch 2.6 flipped the `torch.load` `weights_only` default to `True`. mmengine
0.10.7's local checkpoint loader never passes the argument, and the RTMO
checkpoint stores numpy objects that the restricted unpickler cannot rebuild
(`add_safe_globals` is not sufficient).

`inference/rtmo_inference.py::install_checkpoint_loader_workaround()`
re-registers the local checkpoint scheme through mmengine's **public**
`CheckpointLoader.register_scheme(prefixes="", force=True)` API and calls
`torch.load(..., weights_only=False)`. Nothing under `site-packages` is
patched. The runtime log line `Loads checkpoint by _local_trusted backend`
confirms the override is active.

Any torch upgrade must keep this override working, and any code that loads this
checkpoint needs the same override.

---

## MMPose / MMDetection / MMCV

| Package | Version | Why this version |
|---|---|---|
| `mmengine` | 0.10.7 | mmpose requires `>=0.6.0, <=1.0.0` |
| `mmcv-lite` | 2.1.0 | **not 2.2.0** — `mmdet/__init__.py` asserts `mmcv < 2.2.0` |
| `mmdet` | 3.3.0 | required by RTMO's head (see below) |
| `mmpose` | 1.3.2 | latest release; ships the RTMO configs and the Body7 metafile |

Used by `inference/rtmo_inference.py` via `mmpose.apis.init_model` and
`inference_bottomup`.

### Why mmdet is installed at all

`mmpose/models/heads/hybrid_heads/rtmo_head.py` does
`from mmdet.utils import ConfigType, reduce_mean`. mmdet is **not** in mmpose's
`install_requires` (it sits under the `mim` extra), so it has to be installed
explicitly — and that is what forces `mmcv-lite` down to 2.1.0.

### The `mmcv._ext` stub (still current)

`mmcv-lite` ships the pure-Python `mmcv/ops/*.py` wrappers but not the compiled
`_ext` extension they bind to. Importing `mmpose.apis` pulls in
`mmpose.models.heads.transformer_heads.edpose_head`, which unconditionally does
`from mmcv.ops import MultiScaleDeformableAttention` — so all of `mmcv.ops` is
imported even though no op is ever used.

RTMO uses **no** mmcv native op: its head needs only `mmcv.cnn.ConvModule` /
`Scale` and mmpose's pure-PyTorch `nms_torch`.

[`mmcv_ext_stub.py`](mmcv_ext_stub.py) is therefore `COPY`'d over
`/usr/local/lib/python3.10/dist-packages/mmcv/_ext.py`. It satisfies the
import-time `hasattr(ext, fn)` probe in `mmcv.utils.ext_loader.load_ext` and
returns a placeholder that **raises `NotImplementedError` if an op is actually
called**, naming the missing operator. The node treats that as fatal and
refuses to publish rather than emit a fabricated result.

Full `mmcv` 2.x does **not** build from source against torch 2.14 (C++ API
drift). This was attempted and abandoned; the stub is what avoids needing it.
If real ops are ever required (mmdet detectors, `batched_nms`, DCN,
`MultiScaleDeformableAttention`), delete the stub and install a full mmcv
build.

---

## BoxMOT / BoT-SORT

| Package | Version | Source |
|---|---|---|
| `boxmot` | **19.0.0** | PyPI (pure-Python `py3-none-any` wheel) |

Used by `inference/person_tracking.py` for person tracking, and extended by
`inference/occlusion_tracking.py`. See
[`../docs/architecture.md`](../docs/architecture.md) for what the tracking
stage does.

### The version pin is load-bearing

BoxMOT's numpy floor moved over releases:

| BoxMOT | requires-python | numpy bound | Usable here? |
|---|---|---|---|
| 20.0.0 – 23.0.0 (latest) | >=3.10,<3.14 | **>=2.2.0** | NO — displaces numpy |
| 13.0.x – **19.0.0** | >=3.9,<3.13 | unconstrained | **yes — pinned to 19.0.0** |
| <=12.0.10 | >=3.9 | ==1.26.4 | NO — displaces numpy |

19.0.0 is the **newest release that leaves numpy alone**. A plain
`pip install boxmot` was dry-run first and would have installed numpy 2.2.6,
displacing the 1.23.5 that chumpy, opencv-python 4.8.1.78 and the compiled
mmpose/xtcocotools wheels require — an ABI break, not just a version bump.

Torch is not affected: boxmot 19.0.0 asks for `torch>=2.2.1,<3` and
`torchvision>=0.17.1,<1`, which 2.14.0+cu130 / 0.29.0+cu130 already satisfy.

### Reported version mismatch — expected, not a problem

The runtime banner prints `BoxMOT v18.0.0` while the installed distribution is
19.0.0:

```text
importlib.metadata.version("boxmot") -> 19.0.0     # distribution metadata
boxmot.__version__                   -> 18.0.0     # in-code constant
```

Upstream never bumped the in-code constant for the 19.0.0 release. The
distribution metadata is authoritative, which is why
[`verify_image.py`](verify_image.py) checks
`importlib.metadata.version("boxmot")` and **not** `boxmot.__version__`. Do not
"fix" this by loosening the pin.

### Internal-API dependencies (the real upgrade risk)

The wrapper uses the public constructor, but the occlusion extension reaches
into BoxMOT internals. All of these are BoxMOT 19.0.0 paths:

| Imported symbol | Used by |
|---|---|
| `boxmot.trackers.botsort.botsort.BotSort` | `person_tracking.py` — constructed once |
| `boxmot.trackers.botsort.botsort` (module object) | `occlusion_tracking.py` — the module-global `STrack` name is swapped for the duration of one call |
| `boxmot.trackers.botsort.botsort_track.STrack` | `occlusion_tracking.py` — subclassed |
| `boxmot.trackers.botsort.basetrack.TrackState` | `occlusion_tracking.py` |
| `BotSort._update_impl(dets, img, embs)` | `occlusion_tracking.py` — overridden to swap `STrack` around a `super()` call |

It also depends on the **detection array layout** verified against
`boxmot/trackers/detection_layout.py`:

```text
in : (N, 6) float32  [x1, y1, x2, y2, conf, cls]
out: (M, 8) float32  [x1, y1, x2, y2, track_id, conf, cls, det_ind]
```

and on these constructor keyword arguments being accepted: `reid_model`,
`with_reid`, `cmc_method`, `frame_rate`, `track_high_thresh`,
`new_track_thresh`, `track_buffer`, `match_thresh`, `appearance_thresh`,
`proximity_thresh`.

Two further behavioural assumptions the tuning relies on:

- `max_time_lost = int(frame_rate / 30.0 * track_buffer)` — this is how
  `track_buffer: 90` at `frame_rate: 55` yields 165 frames (~3.0 s).
- The proximity gate masks the ReID distance when
  `iou_dist > proximity_thresh`, with `iou_dist = 1 - IoU`.

**An upgrade will very likely break the occlusion extension** even if the
public constructor still works, because a renamed module, a moved `STrack`, or
a restructured `_update_impl` silently changes which class BoxMOT instantiates.
`cmc_method` also has a quirk: BoxMOT accepts Python `None` to disable
camera-motion compensation but **rejects the string `"none"`**, so the node
maps the friendly parameter value before passing it.

---

## ReID / OSNet

ReID is provided by BoxMOT, not by a separate package. `person_tracking.py`
builds it with:

```python
from boxmot.reid import ReID
ReID(path=Path(reid_checkpoint), device=device, half=False)
```

and hands `reid.model` to `BotSort(reid_model=..., with_reid=True)`.

| | |
|---|---|
| Backbone | OSNet `osnet_x0_25`, MSMT17 weights |
| Backend | `boxmot.reid.backends.pytorch_backend.PyTorchBackend` |
| Precision | full (`half=False`) |
| Device | same as RTMO (`cuda:0` by default) |
| Weights | `/opt/models/reid/osnet_x0_25_msmt17.pt` (see Model assets) |

Only the core BoT-SORT + **PyTorch** ReID path is installed. The optional
backends `onnx` / `onnxruntime`, `openvino`, `tflite`, `yolo`
(ultralytics + yolox), `trackeval`, `evolve`, `rtdetr` and `coreml` are
deliberately **not** installed — that keeps a second inference runtime out of
the image. TensorRT is not even a BoxMOT extra, and FAISS is not a dependency.

`verify_image.py` asserts both `boxmot.reid.ReID` and `PyTorchBackend` import,
so a restructure of the ReID module fails the build rather than the robot.

---

## RealSense SDK

| Component | Version | Source |
|---|---|---|
| C++ runtime | `ros-humble-librealsense2` **2.58.3**-1jammy | apt, pulled in by `ros-humble-realsense2-camera` |
| Python bindings | `pyrealsense2` **2.58.3.10794** | PyPI |

`input/realsense_capture.py` opens the D456 in-process, so the **Python**
bindings are required; the base image only carries the C++ runtime.

**Why this version.** PyPI ships a `manylinux2014_aarch64` cp310 wheel for
2.58.3.10794, so nothing is built from source and no Intel apt repository is
needed. The version is pinned to match the apt runtime already present. The
wheel is self-contained (it bundles its own librealsense), so the two never
actually have to interoperate — keeping them on the same version avoids
surprises if both are ever loaded.

**Device access:** the container runs as root with `/dev:/dev` bind-mounted and
`privileged: true`, which is enough for libusb to claim the camera. **No udev
rules are installed inside the container.**

**Verified device** (2026-09-07): RealSense D456, serial `308222301472`,
firmware 5.17.0.10, USB 3.2. Colour profile **848×480 @ 60 fps `bgr8`** is
natively advertised, which matters for dependency reasons: the RTMO config uses
`mean=[0,0,0]`, `std=[1,1,1]` and no `bgr_to_rgb`, so the captured array reaches
MMPose with **no colour conversion**. Measured raw capture rate 59.8 FPS.

---

## Supporting Python packages

### Toolchain

| Package | Version | Why |
|---|---|---|
| `pip` | 26.2.1 | — |
| `setuptools` | 79.0.1 | **must stay `<80`** — colcon's `setup.py` build path breaks on 80+ |
| `wheel` | 0.48.0 | — |

### Numeric / imaging

| Package | Version | Why this version |
|---|---|---|
| `numpy` | **1.23.5** | the keystone pin — chumpy imports numpy aliases deleted in numpy 2.x |
| `opencv-python` | 4.8.1.78 | 5.x requires numpy>=2, which breaks chumpy |
| `scipy` | 1.15.3 | mmpose dependency |
| `matplotlib` | 3.10.9 | xtcocotools + mmpose dependency |
| `Cython` | 0.29.37 | **build-time** requirement for xtcocotools (see Known workarounds) |

### OpenMMLab transitive dependencies

| Package | Version | Why |
|---|---|---|
| `chumpy` | 0.70 | mmpose dependency; sdist, needs `--no-build-isolation` |
| `xtcocotools` | 1.14.3 | mmpose dependency; sdist, needs Cython + `--no-build-isolation` |
| `pycocotools` | 2.0.11 | mmdet dependency |
| `shapely` | 2.1.2 | mmdet dependency |
| `terminaltables` | 3.1.10 | mmdet dependency |
| `json-tricks` | 3.17.3 | mmpose dependency |
| `munkres` | 1.1.4 | mmpose dependency |

### Installed with boxmot

All `aarch64` wheels, no source builds: `filterpy` 1.4.5 (Kalman),
`lapx` 0.9.4 (Hungarian/LAP solver), `pandas` 2.3.3,
`scikit-learn` 1.7.2, `gdown` 5.2.2, `huggingface-hub` 1.30.0, plus `joblib`,
`threadpoolctl`, `yacs`, `ftfy`, `regex` and `rich`/`click` support packages.

`pandas` and `scikit-learn` ship numpy-2-built wheels but declare
`numpy>=1.22`, and both were verified to import and compute correctly under
numpy 1.23.5.

### The constraints file

The Dockerfile writes `/etc/pip-constraints.txt` and every later
`pip install` passes `-c /etc/pip-constraints.txt`. It pins: `numpy`,
`opencv-python`, `scipy`, `matplotlib`, `setuptools`, `Cython`, `torch`,
`torchvision`, `mmengine`, `mmcv-lite`, `mmdet`, `mmpose`, `chumpy`,
`xtcocotools`, `pycocotools`, `shapely`, `terminaltables`, `json-tricks`,
`munkres`.

**This file is load-bearing, not decorative.** It stops a transitive dependency
(pandas, scikit-learn, huggingface-hub) from dragging numpy 2.x in, and turns a
future incompatible resolution into a **loud build failure** instead of a
silent runtime ABI break. Always install with `-c /etc/pip-constraints.txt`.

---

## Model assets

Baked into the image at build time by [`fetch_models.py`](fetch_models.py), so
a pulled image is self-contained. Previously these lived in `data/checkpoints/`,
which is both gitignored and dockerignored — so `git clone` + `docker pull`
produced an environment with no weights.

```text
/opt/models/
├── rtmo/
│   ├── rtmo-m.py           14.4 kB   COPY'd from models/rtmo/ (tracked in git)
│   ├── default_runtime.py   1.7 kB   COPY'd from models/rtmo/ (tracked in git)
│   └── rtmo-m.pth          90.5 MB   downloaded + sha256-verified
└── reid/
    └── osnet_x0_25_msmt17.pt  3.06 MB  downloaded + sha256-verified
```

| Asset | Used by | How it gets there | sha256 |
|---|---|---|---|
| `rtmo-m.py` + `default_runtime.py` | `inference/rtmo_inference.py` | `COPY models/rtmo/ /opt/models/rtmo/` — **version-controlled, not downloaded or generated** | n/a (in git) |
| `rtmo-m.pth` | `inference/rtmo_inference.py` | downloaded from `download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth` | `39e78cc4afed7d1dd31ea73ee04842f7d8f8e695d900d41b90746e94728a01c3` |
| `osnet_x0_25_msmt17.pt` | `inference/person_tracking.py` | BoxMOT's own registry URL (`boxmot.reid.core.registry.TRAINED_URLS`), fetched with `boxmot.utils.download.download_file` | `6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99` |

Sources are **official only** — no third-party mirror. Every download is
sha256-verified and the file is deleted on mismatch, so a corrupted or
substituted file fails `docker build` rather than silently shipping.

`fetch_models.py` downloads the **weights only**. It asserts the git-tracked
RTMO config was copied in and loads with its `_base_` chain resolved, checking
that `model`, `test_dataloader` and `default_scope` are all present — a missing
COPY, a renamed file or a broken `_base_` fails the build, not node start-up.

**Why the ReID weights are fetched at build time.** BoxMOT would otherwise
download them on first use at runtime, into whatever directory the process
happens to use — not reproducible, and a failure on an offline robot.

Environment variables exported by the image, which are also the node's
parameter defaults: `RTMO_MODEL_CONFIG`, `RTMO_CHECKPOINT`, `REID_CHECKPOINT`.
A ROS parameter always wins, so a different config/checkpoint can be used
without rebuilding.

---

## Compatibility constraints

The hard edges, in priority order:

1. **`numpy == 1.23.5` is the keystone.** chumpy imports numpy aliases removed
   in 2.x; opencv-python 4.8 and the compiled mmpose/xtcocotools wheels are
   built against 1.x ABI. Anything that pulls numpy 2.x breaks the RTMO stack.
2. **`boxmot <= 19.0.0`**, because 20.0.0+ requires `numpy>=2.2`. This follows
   directly from (1).
3. **`opencv-python < 5`**, because 5.x requires `numpy>=2`.
4. **`mmcv-lite < 2.2.0`**, asserted by `mmdet/__init__.py`.
5. **`mmengine >= 0.6.0, <= 1.0.0`**, required by mmpose.
6. **`setuptools < 80`**, or colcon's `setup.py` build path breaks.
7. **torch/torchvision must stay a matched `+cu130` aarch64 pair**, or the
   CUDA 13 / GB10 build is invalidated.
8. **`pyrealsense2` needs an aarch64 cp310 wheel**, or it builds from source.

[`verify_image.py`](verify_image.py) enforces 1, 2, 4, 5, 7 and the asset
layout at **build time**, so a bad resolution fails `docker build` instead of
the robot. It checks numpy `1.23.x`, torch `2.14.0+cu130`, torchvision
`0.29.0+cu130`, opencv `4.8.x`, mmengine `0.10.7`, mmcv `2.1.0`, mmdet `3.3.0`,
mmpose `1.3.2`, pyrealsense2 `2.58.x`, boxmot `19.0.0`, the RTMO import chain,
the BoT-SORT and ReID imports, and the presence and minimum size of all four
baked assets.

---

## Known workarounds

| # | Workaround | Where | Still current? |
|---|---|---|---|
| 1 | `mmcv._ext` import-only stub | [`mmcv_ext_stub.py`](mmcv_ext_stub.py), `COPY`'d into site-packages | **yes** |
| 2 | mmengine checkpoint loader re-registered with `weights_only=False` | `inference/rtmo_inference.py` | **yes** |
| 3 | `chumpy` + `xtcocotools` installed with `--no-build-isolation` and Cython pre-installed | Dockerfile | **yes** |
| 4 | mmdet installed explicitly (not a mmpose `install_requires`) | Dockerfile | **yes** |
| 5 | `cmc_method` string `"none"` mapped to Python `None` | `iot_node.py` | **yes** |
| 6 | boxmot version read from distribution metadata, not `boxmot.__version__` | `verify_image.py` | **yes** |

**On (3):** `xtcocotools`' `setup.py` declares `xtcocotools/_mask.pyx` as an
extension source but ships no `_mask.c`. setuptools only routes `.pyx` through
Cython when Cython is importable (`setuptools.command.build_ext` imports
`Cython.Distutils.build_ext`), so without Cython the build fails on the missing
`_mask.c`. It also has no `pyproject.toml`, so PEP 517 isolation would hide
numpy from its `setup.py` — hence `--no-build-isolation` as well. `chumpy` is
an sdist with the same isolation problem.

**Not a workaround, a limitation:** full `mmcv` 2.x does not build from source
against torch 2.14 (C++ API drift). Not retried; (1) avoids needing it.

### A note on the test environment

The image carries **pytest 6.2.5** (apt) alongside **anyio 4.15.1** (pip, via
boxmot → huggingface-hub). anyio registers a `pytest11` entry point whose
plugin imports `_pytest.scope`, which exists only in pytest 7+, so bare
`pytest` dies at start-up with `ModuleNotFoundError: No module named
'_pytest.scope'`. Run the suite with `-p no:anyio` (or
`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`). This is a packaging mismatch in the image,
not a test failure — see [`../docs/debugging.md`](../docs/debugging.md).

---

## Upgrade guidance

**Do not bump a pin casually.** The versions above are a resolved set, not
independent choices: numpy 1.23.5 constrains boxmot, opencv and the OpenMMLab
wheels simultaneously, and `/etc/pip-constraints.txt` exists precisely so a
careless upgrade fails loudly.

Before changing any dependency version:

1. Read the **Compatibility constraints** section and check which of the eight
   hard edges the change touches.
2. Dry-run the resolution first (`pip install --dry-run ... -c
   /etc/pip-constraints.txt`) and look at what it wants to do to **numpy**.
3. Change one thing at a time.

After changing any dependency version, revalidate **all** of the following.
`docker compose build` alone is not sufficient — it proves imports resolve, not
that inference is still correct:

| Step | Command / check |
|---|---|
| Image builds and the gate passes | `docker compose build` (runs `verify_image.py`) |
| Package builds | `rm -rf build/skeleton_detection install/skeleton_detection && colcon build --packages-select skeleton_detection` |
| Unit tests | `python3 -m pytest test/ -q -p no:anyio` |
| RTMO inference | node starts, logs `Loaded RTMO-M` and `Keypoint layout verified: COCO-17`, and produces plausible keypoints |
| BoT-SORT tracking | `enable_tracking:=true` — IDs stay stable across frames; check the logged association gates |
| ReID | `with_reid:=true` — ReID loads, and re-identification across a brief occlusion still works |
| Occlusion-aware tracking | `occlusion_aware_tracking:=true` — **highest breakage risk on a boxmot upgrade**; confirm `OcclusionAwareSTrack` is actually instantiated and `test/test_occlusion_tracking.py` passes |
| RealSense capture | live camera opens at 848×480 @ 60 `bgr8`, capture FPS near 60 |
| Depth / XYZ | `position` and `depth` are finite and plausible for a person at a known distance — **not** `NaN` for everyone |
| ROS publication | `ros2 topic hz /skeleton_detection/frame` and `ros2 topic echo --once` show correct field values |

Dependency-specific extra care:

- **numpy** — touching this means revalidating the entire stack. Treat it as a
  rebuild of the environment, not an upgrade.
- **boxmot** — beyond the numpy floor, re-check every internal symbol in the
  *Internal-API dependencies* table. The occlusion extension swaps a module
  global; if BoxMOT moves or renames `STrack`, the swap silently stops taking
  effect and tracking quietly reverts to stock behaviour instead of erroring.
- **torch** — re-check the checkpoint-loading workaround and confirm the
  `+cu130` aarch64 wheels still exist for the target version.
- **mmcv / mmpose / mmdet** — re-check the `mmcv < 2.2.0` assertion and whether
  the `_ext` stub still satisfies the import chain.
- **pyrealsense2** — confirm an aarch64 cp310 wheel exists and that the D456
  still advertises 848×480 @ 60 `bgr8`.

Running the live pipeline for a fixed window is the quickest end-to-end check:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true run_duration_sec:=30.0
```
