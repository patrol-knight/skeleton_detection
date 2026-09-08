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

## Model

- Config (bundled with mmpose):
  `/usr/local/lib/python3.10/dist-packages/mmpose/.mim/configs/body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py`
- Weights: `rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth`
  from `https://download.openmmlab.com/mmpose/v1/projects/rtmo/`
  sha256 `39e78cc4afed7d1dd31ea73ee04842f7d8f8e695d900d41b90746e94728a01c3`
- Stored in `data/checkpoints/` (gitignored, inside the bind mount).
