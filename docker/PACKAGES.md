# RTMO-M runtime environment — required packages

Host: NVIDIA DGX Spark, GB10 (compute capability 12.1), aarch64,
driver 580.173.02, CUDA 13.0.
Container: `ros:humble-ros-base` (Ubuntu 22.04 jammy), Python 3.10.12.

## Pinned packages

| Package | Version | Source | Why this version |
|---|---|---|---|
| pip | 26.2.1 | PyPI | — |
| setuptools | 79.0.1 | PyPI | must stay `<80`; colcon's setup.py build path breaks on 80+ |
| wheel | 0.48.0 | PyPI | — |
| torch | 2.14.0+cu130 | download.pytorch.org/whl/cu130 | only index with aarch64 + CUDA 13 wheels; runs on sm_121 |
| torchvision | 0.29.0+cu130 | download.pytorch.org/whl/cu130 | matches torch 2.14.0 |
| numpy | 1.23.5 | PyPI | chumpy imports numpy aliases deleted in numpy 2.x |
| opencv-python | 4.8.1.78 | PyPI | 5.x requires numpy>=2, which breaks chumpy |
| scipy | 1.15.3 | PyPI | mmpose dependency |
| matplotlib | 3.10.9 | PyPI | xtcocotools + mmpose dependency |
| Cython | 0.29.37 | PyPI | **build-time requirement for xtcocotools** (see below) |
| chumpy | 0.70 | PyPI (sdist) | mmpose dependency; needs `--no-build-isolation` |
| xtcocotools | 1.14.3 | PyPI (sdist) | mmpose dependency; needs Cython + `--no-build-isolation` |
| pycocotools | 2.0.11 | PyPI | mmdet dependency |
| mmengine | 0.10.7 | PyPI | mmpose requires `>=0.6.0, <=1.0.0` |
| mmcv-lite | 2.1.0 | PyPI | **not 2.2.0**: mmdet 3.3.0 asserts `mmcv < 2.2.0` |
| mmdet | 3.3.0 | PyPI | required by RTMO (see below) |
| mmpose | 1.3.2 | PyPI | latest release; has RTMO configs + Body7 metafile |
| shapely | 2.1.2 | PyPI | mmdet dependency |
| terminaltables | 3.1.10 | PyPI | mmdet dependency |
| json-tricks | 3.17.3 | PyPI | mmpose dependency |
| munkres | 1.1.4 | PyPI | mmpose dependency |
| pyrealsense2 | 2.58.3.10794 | PyPI | RealSense Python bindings; version matched to the `ros-humble-librealsense2` 2.58.3 runtime (see below) |
| boxmot | 19.0.0 | PyPI | BoT-SORT + ReID; **newest release that does not force numpy>=2.2** (see below) |

## CUDA apt packages (optional for inference)

From `https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/`:

- `cuda-nvcc-13-0` 13.0.88-1
- `cuda-cudart-dev-13-0` 13.0.96-1

These are **not** needed to run RTMO: the `+cu130` torch wheels ship their own
CUDA runtime via the `nvidia-*` pip packages. They are installed so that
`nvcc` / `CUDA_HOME` exist for later TensorRT or full-mmcv compilation.

## Non-obvious findings

1. **xtcocotools needs Cython at build time.** Its `setup.py` declares
   `xtcocotools/_mask.pyx` as an extension source but ships no `_mask.c`.
   setuptools only routes `.pyx` through Cython when Cython is importable
   (`setuptools.command.build_ext` imports `Cython.Distutils.build_ext`).
   Without Cython the build fails on the missing `_mask.c`. It also has no
   `pyproject.toml`, so PEP 517 isolation hides numpy from its `setup.py` —
   hence `--no-build-isolation` as well.

2. **RTMO requires mmdet.** `mmpose/models/heads/hybrid_heads/rtmo_head.py`
   does `from mmdet.utils import ConfigType, reduce_mean`. mmdet is not in
   mmpose's install_requires (it is under the `mim` extra), so it must be
   installed explicitly. This in turn forces mmcv-lite down to 2.1.0, because
   `mmdet/__init__.py` asserts `mmcv < 2.2.0`.

3. **mmcv-lite alone cannot import mmpose.** `mmpose.models.heads.__init__`
   unconditionally imports `transformer_heads` -> `edpose_head.py`, which does
   `from mmcv.ops import MultiScaleDeformableAttention`. That pulls in all of
   `mmcv.ops`, which dies on the absent compiled `mmcv._ext`.
   RTMO itself uses **no** mmcv op (only `mmcv.cnn.ConvModule`/`Scale` and
   mmpose's pure-PyTorch `nms_torch`), so `docker/mmcv_ext_stub.py` provides an
   import-only `mmcv._ext` that raises if an op is ever actually called.
   Replace it with a full mmcv build if real ops are needed later.

4. **torch >= 2.6 breaks mmengine checkpoint loading.** torch 2.6 flipped the
   `torch.load` `weights_only` default to `True`; mmengine 0.10.7's local
   checkpoint loader never passes the argument, and the RTMO checkpoint stores
   numpy objects the restricted unpickler cannot rebuild
   (`add_safe_globals` is not sufficient). `tools/rtmo_smoke_test.py` overrides
   the loader through mmengine's public `CheckpointLoader.register_scheme(...,
   force=True)` API rather than editing site-packages. Any future ROS node that
   loads this checkpoint needs the same override.

5. Full `mmcv` 2.x does **not** build from source against torch 2.14 (C++ API
   drift). Not retried; the stub above avoids needing it for RTMO.

## RealSense (Milestone 2)

Direct, in-process capture from the D456 needs the **Python** bindings; the
base image already carries the C++ runtime.

- C++ runtime: `ros-humble-librealsense2` **2.58.3** (apt, pulled in as a
  dependency of `ros-humble-realsense2-camera` in the base layer). Provides
  `/opt/ros/humble/lib/aarch64-linux-gnu/librealsense2.so.2.58.3`.
- Python bindings: `pyrealsense2==2.58.3.10794` from PyPI. A
  `manylinux2014_aarch64` cp310 wheel exists, so **nothing is built from
  source** and no Intel apt repo is needed. The version is pinned to match the
  apt runtime above; the wheel is self-contained (it bundles its own
  librealsense) so the two never have to interoperate, but keeping them on the
  same version avoids surprises if both are ever loaded.
- Device access: the container is started with `-v /dev:/dev` and runs as root,
  which is sufficient for libusb to claim the camera. No udev rules are
  installed inside the container.

Verified device (2026-09-07): **RealSense D456**, serial `308222301472`,
firmware `5.17.0.10`, USB 3.2. Colour profile **848x480 @ 60 fps `bgr8`** is
natively advertised, so the frame handed to MMPose needs no colour conversion
(the RTMO config uses `mean=[0,0,0]`, `std=[1,1,1]`, no `bgr_to_rgb`).
Measured raw capture rate: **59.8 FPS**.

## Tracking / ReID (BoxMOT — installed, NOT yet integrated)

`boxmot` provides BoT-SORT with built-in ReID and accepts detections from an
external detector, which is what we need for RTMO. It is a pure-Python wheel
(`py3-none-any`), so aarch64 is not an issue for boxmot itself.

**Tracking is not wired into the ROS node yet.** `iot_node.py` does not import
boxmot, `person_id` is still a frame-local detection index, and no ReID runs at
runtime. This section documents the environment that the future
`skeleton_detection/tracking.py` will build on.

### Version pin — this is load-bearing

BoxMOT's numpy floor moved over time:

| BoxMOT | requires-python | numpy bound | Usable here? |
|---|---|---|---|
| 20.0.0 – 23.0.0 (latest) | >=3.10,<3.14 | **>=2.2.0** | NO — displaces numpy |
| 13.0.x – **19.0.0** | >=3.9,<3.13 | unconstrained | **yes — pinned to 19.0.0** |
| <=12.0.10 | >=3.9 | ==1.26.4 | NO — displaces numpy |

`19.0.0` is the newest release that leaves numpy alone. A plain
`pip install boxmot` was dry-run first and would have installed **numpy 2.2.6**,
displacing the 1.23.5 that chumpy, opencv-python 4.8.1.78 and the compiled
mmpose/xtcocotools wheels require — an ABI break, not just a version bump.

Install is always done with `-c /etc/pip-constraints.txt`. That constraints
file is what stops a transitive dependency (pandas, scikit-learn) from dragging
numpy 2.x in, and it turns any future incompatible resolution into a loud build
failure instead of a silent runtime ABI break.

Torch/torchvision are **not** touched: boxmot 19.0.0 asks for
`torch>=2.2.1,<3` and `torchvision>=0.17.1,<1`, which 2.14.0+cu130 /
0.29.0+cu130 already satisfy.

### Core dependencies installed with boxmot

All aarch64 wheels, no source builds: `filterpy 1.4.5`, `lapx 0.9.4`
(Hungarian/LAP solver), `pandas 2.3.3`, `scikit-learn 1.7.2`, `gdown 5.2.2`,
`huggingface-hub 1.30.0`, `joblib`, `threadpoolctl`, `yacs`, `ftfy`, `regex`,
plus `rich`/`click` support packages. `pandas` and `scikit-learn` ship
numpy-2-built wheels but declare `numpy>=1.22`, and both were verified to
import and compute correctly under numpy 1.23.5.

### Optional backends deliberately NOT installed

`onnx` / `onnxruntime`, `openvino`, `tflite`, `yolo` (ultralytics + yolox),
`trackeval`, `evolve`, `rtdetr`, `coreml`. TensorRT is not even a BoxMOT extra.
FAISS is not a dependency at all. Only the core BoT-SORT + **PyTorch** ReID
path is installed, which keeps the image free of a second inference runtime.

## Model assets baked into the image

Model files used to live in `data/checkpoints/`, which is both `.gitignore`d
and `.dockerignore`d — so `git clone` + `docker pull` produced an environment
with **no weights**. They are now baked into the image by
`docker/fetch_models.py` at build time, sha256-verified, so a pulled image is
self-contained:

```
/opt/models/
├── rtmo/
│   ├── rtmo-m.py     standalone MMPose config (65.6 kB)
│   └── rtmo-m.pth    RTMO-M Body7 checkpoint (90.5 MB)
└── reid/
    └── osnet_x0_25_msmt17.pt   ReID weights (3.06 MB)
```

| Asset | Source | sha256 |
|---|---|---|
| `rtmo-m.pth` | `download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth` | `39e78cc4afed7d1dd31ea73ee04842f7d8f8e695d900d41b90746e94728a01c3` |
| `rtmo-m.py` | the config inside the installed `mmpose` wheel, re-emitted via mmengine | n/a (generated) |
| `osnet_x0_25_msmt17.pt` | BoxMOT's own registry URL, fetched with `boxmot.utils.download.download_file` | `6f57607fed9f502b9efed546108132ee715df5a5b6e6932c6269bacb47f59f99` |

**Why the RTMO config is re-emitted rather than copied.** The MMPose config
declares a *relative* `_base_ = ['../../../_base_/default_runtime.py']`, so a
flat copy would fail to load. `docker/fetch_models.py` runs mmengine's own
`Config.fromfile(...).dump(...)`, which inlines the merged result into a
self-contained file. The dumped config was verified to produce **bit-identical**
keypoints, scores and bboxes versus the original `.mim` config.

**Why the ReID weights are fetched at build time.** BoxMOT would otherwise
download them on first use at runtime (into whatever directory the process
happens to use), which is not reproducible and fails on an offline robot. The
build uses BoxMOT's *own* registry URL and downloader — no third-party mirror —
and verifies the sha256, so a substituted or truncated file fails the build.

Environment variables exported by the image (also the node's defaults):
`RTMO_MODEL_CONFIG`, `RTMO_CHECKPOINT`, `REID_CHECKPOINT`.

`docker/verify_image.py` runs at build time and fails the build if numpy,
torch, torchvision, the OpenMMLab versions, the RTMO import chain,
pyrealsense2, boxmot 19.0.0, BoT-SORT/ReID imports, or any baked asset is
missing or displaced.

## Planned architecture (next milestone, not implemented)

```
D456 -> pyrealsense2 -> RTMO-M -> BoT-SORT + OSNet ReID -> SkeletonFrame
                                  (same process)          persistent person_id
```

`skeleton_detection/tracking.py` should load ReID from
`/opt/models/reid/osnet_x0_25_msmt17.pt`, feed BoT-SORT detections shaped
`(x1, y1, x2, y2, conf, cls)` taken from RTMO's pre-conversion xyxy boxes, and
map the returned `det_ind` column back onto the corresponding `PersonSkeleton`.

## Model (RTMO-M)

- Runtime config:     `/opt/models/rtmo/rtmo-m.py`   (baked into the image)
- Runtime checkpoint: `/opt/models/rtmo/rtmo-m.pth`  (baked into the image)
- Upstream config: the one bundled with mmpose at
  `mmpose/.mim/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py`
- Upstream weights: `rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth`
  from `https://download.openmmlab.com/mmpose/v1/projects/rtmo/`
  sha256 `39e78cc4afed7d1dd31ea73ee04842f7d8f8e695d900d41b90746e94728a01c3`

The node defaults to the `/opt/models` paths; both remain overridable with the
`model_config` and `checkpoint` ROS parameters. `data/checkpoints/` is no
longer required at runtime.
