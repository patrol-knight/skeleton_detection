# Setup

Prerequisites, host expectations and first-time repository setup.

Everything after this page assumes the container built here. For the container
workflow itself see [Docker & Compose](docker.md); for the commands that run
the pipeline see [Running the pipeline](running.md).

---

## Host expectations

The workspace is developed and validated on the following host. Nothing in the
code hard-codes it, but the pinned dependency stack (see
[`../docker/PACKAGES.md`](../docker/PACKAGES.md)) was resolved against it.

| | |
|---|---|
| Machine | NVIDIA DGX Spark |
| GPU | GB10, compute capability 12.1 |
| Architecture | `aarch64` |
| Driver / CUDA | 580.173.02 / CUDA 13.0 |
| Container base | `ros:humble-ros-base` (Ubuntu 22.04 jammy), Python 3.10.12 |

Required host software:

- Docker Engine with **Docker Compose v2 or newer** (`docker compose`, not
  `docker-compose`).
- **NVIDIA Container Toolkit.** The compose service reserves `driver: nvidia`
  with `count: all`; both RTMO-M and the OSNet ReID backbone run on `cuda:0`.
- An X server on the host if you want `rqt_image_view` inside the container.
  The compose file bind-mounts `/tmp/.X11-unix` and passes `DISPLAY` through.

The ROS 2 Humble toolchain, CUDA tools, OpenMMLab stack, RealSense bindings and
BoxMOT all live **inside the image**. Nothing has to be installed on the host
beyond Docker and the NVIDIA toolkit.

---

## Hardware

| | |
|---|---|
| Camera | Intel RealSense **D456** |
| Verified device (2026-09-07) | serial `308222301472`, firmware `5.17.0.10`, USB 3.2 |
| Colour profile used | **848 × 480 @ 60 fps `bgr8`** (natively advertised) |
| Depth profile used | Z16 at the same resolution/rate, aligned to colour |

The `bgr8` colour profile matters: the RTMO config uses `mean=[0,0,0]`,
`std=[1,1,1]` and no `bgr_to_rgb`, so the captured frame is handed to MMPose
with **no per-frame colour conversion**. Measured raw capture rate: 59.8 FPS.

In the default `input_mode: realsense` the camera is opened **in-process** by
the node with `pyrealsense2`, and no `realsense2_camera` node is involved. Raw
USB access is what the `privileged: true` + `/dev:/dev` mount in
`compose.yaml` provides.

Two other input modes need no camera on this host:

- `input_mode: ros_topic` — the offline image workflow (see
  [Running the pipeline](running.md)); `position` and `depth` are always `NaN`.
- `input_mode: ros_camera` — consumes a `realsense2_camera_msgs/msg/RGBD`
  topic from a `realsense2_camera` driver running **elsewhere** (typically
  another container), which owns the camera, the alignment and the
  synchronization. This package needs neither `pyrealsense2` nor USB access in
  that mode, only the `realsense2_camera_msgs` message package and DDS
  reachability to the driver. See
  [Input modes](running.md#3-input-modes).

---

## Repository setup

```bash
git clone <this-repo-url> skeleton_detection
cd skeleton_detection
```

Every documented command assumes you are inside that clone and uses `$(pwd)`
rather than an absolute host path, so the workflow is not tied to one machine.

### What is and is not in git

| Path | In git? | Notes |
|---|---|---|
| `models/rtmo/rtmo-m.py`, `models/rtmo/default_runtime.py` | **yes** | the RTMO config the image actually runs |
| `*.pth` / `*.pt` checkpoints | **no** | `.gitignore` blocks them; downloaded and sha256-verified during `docker build` |
| `data/` | **no** | gitignored and dockerignored |
| `output/` | **no** | generated visualization artefacts |

There is **no manual model copying and no `docker cp`**. A fresh
`git clone` + `docker compose build` produces a fully runnable container with
the weights baked into an image layer at `/opt/models`, so destroying and
recreating the container never loses them.

---

## Initial steps

1. Clone the repository (above).
2. Build the image and start the container — see
   [Docker & Compose](docker.md).
3. Build the ROS package with `colcon` inside the container and run the
   pipeline — see [Running the pipeline](running.md).

If something does not come up, [Debugging](debugging.md) covers the failures
seen on this setup.
