# Docker & Compose

How the container image is built and how its container is run.

The image is **self-contained**: `patrolknight_msgs` and this package are
built into it at `docker build` time. Container start-up only sources the
install spaces and runs the command -- it clones nothing, builds nothing and
needs no source bind mount.

Prerequisites and host expectations are in [Setup](setup.md). What to run once
you are inside the container is in [Running the pipeline](running.md).

---

## The preferred workflow

```bash
docker compose build
docker compose up -d
docker compose exec skeleton_humble bash
```

and, when you are done:

```bash
docker compose down
```

All three commands are run from the **repository root**, where `compose.yaml`
lives. `docker compose up -d` starts the pipeline with the deployment config
`config/skeleton_detection_node.yaml` (see [below](#what-composeyaml-does)).

---

## The Dockerfile

The Dockerfile lives at **`docker/Dockerfile`**. The build **context is still
the repository root**, because the build copies files from two different
top-level directories:

```dockerfile
COPY docker/mmcv_ext_stub.py /usr/local/lib/python3.10/dist-packages/mmcv/_ext.py
COPY models/rtmo/            /opt/models/rtmo/
COPY docker/fetch_models.py  /tmp/fetch_models.py
COPY docker/verify_image.py  /tmp/verify_image.py
```

`compose.yaml` therefore names the two separately:

```yaml
build:
  context: .
  dockerfile: docker/Dockerfile
```

Narrowing the context to `docker/` would put `models/` out of reach and change
which files `.dockerignore` filters, so it is deliberately left at the root.

To build the image by hand without Compose, pass the Dockerfile explicitly and
keep the context at `.`:

```bash
docker build -f docker/Dockerfile -t skeleton_humble_dev .
```

This produces the same `skeleton_humble_dev` tag that `docker compose build`
does, so both share the same layer cache.

### What the build does

- Installs the pinned ROS 2 / CUDA / OpenMMLab / RealSense / BoxMOT stack
  (versions and the reasoning behind each pin:
  [`../docker/PACKAGES.md`](../docker/PACKAGES.md)).
- Writes `/etc/pip-constraints.txt` and installs everything against it, so a
  transitive dependency cannot displace `numpy==1.23.5`.
- Installs the import-only `mmcv._ext` stub from `docker/mmcv_ext_stub.py`.
- Copies the git-tracked RTMO config `models/rtmo/` to `/opt/models/rtmo/`.
- Runs `docker/fetch_models.py`, which downloads and **sha256-verifies** the
  RTMO-M and OSNet ReID checkpoints into `/opt/models`.
- Exports `RTMO_MODEL_CONFIG`, `RTMO_CHECKPOINT` and `REID_CHECKPOINT`, which
  are also the node's parameter defaults.
- Runs `docker/verify_image.py`, a build-time gate that fails the build if any
  pinned version, import chain or baked asset is missing or displaced.
- Clones `patrolknight_msgs` (build arg `PATROLKNIGHT_MSGS_REF`, default
  `main`) and builds it into `/opt/patrolknight_msgs`.
- Copies this package (`package.xml`, `setup.py`, `setup.cfg`, `resource/`,
  `skeleton_detection/`, `config/`, `launch/`) and builds it — an
  `ament_python` package — on top into `/opt/skeleton_detection`.
- `ENTRYPOINT` sources `/opt/ros/humble`, `/opt/patrolknight_msgs` and
  `/opt/skeleton_detection`, then `exec`s the command. `CMD` is a plain
  `ros2 launch skeleton_detection skeleton_detection_bringup.launch.py`, which
  uses the packaged `config/skeleton_detection_node.yaml` (`input_mode: ros_camera`).

Expect a long first build; the checkpoint download needs network access.
Subsequent builds are almost entirely cached.

Inspect the baked assets at any time:

```bash
docker compose exec skeleton_humble ls -lh /opt/models/rtmo /opt/models/reid
```

Model-asset details, sources and checksums are in
[Implementation notes](implementation.md).

---

## What `compose.yaml` does

One service, `skeleton_humble`, which is a transcription of the working
`docker run` invocation — not a new environment.

| Setting | Value | Why |
|---|---|---|
| `image` | `skeleton_humble_dev` | same tag as the manual `docker build` |
| `container_name` | `skeleton_humble` | every documented command names it |
| `command` | `ros2 launch ... config:=/config/skeleton_detection_node.yaml` | starts the pipeline with the mounted deployment config |
| `working_dir` | `/ros2_ws` | the Dockerfile `WORKDIR` and colcon workspace root |
| `networks` | `ros-net` (external `iot_ros-net`) | DDS discovery with the IoT `realsense` / `zenoh_bridge` / `diagnose` containers |
| `ipc` | `host` | Fast DDS shared-memory transport across the boundary |
| `deploy.resources.reservations.devices` | `driver: nvidia`, `count: all` | equivalent of `--gpus all` |

**`compose up` starts the pipeline.** Every parameter comes from the mounted
config file; the command lists no parameter values.

The service is **headless and unprivileged**: no `privileged`, no device
nodes, no `DISPLAY` or X11 socket. In the default `ros_camera` deployment the
RealSense belongs to a separate `realsense2_camera` container and this one
only subscribes to its RGBD topic; the node itself opens no windows (the
visualization is a ROS image topic).

### GPU access

The `deploy` block is the Compose equivalent of `--gpus all` and requires the
NVIDIA Container Toolkit on the host. It is independent of `privileged`: the
NVIDIA runtime injects the GPU device nodes and driver libraries itself. RTMO-M and the OSNet ReID backbone both
run on `cuda:0` by default (`device` parameter).

### Direct RealSense mode (`input_mode: realsense`) — opt-in

The default compose grants no camera access. To open the D456 in this
container instead, pass its V4L2 nodes. The bundled `pyrealsense2` uses the
V4L2 backend, so `/dev/video*` and `/dev/media*` are enough — no
`privileged`, no `/dev/bus/usb`, no udev. For example, with a local override
file (on this host the D456 is `video0`–`video5` and `media0`–`media1`; check
yours with `ls /dev/video* /dev/media*`):

```yaml
# compose.realsense.yaml
services:
  skeleton_humble:
    command:
      - ros2
      - launch
      - skeleton_detection
      - skeleton_detection_bringup.launch.py
      - config:=/opt/skeleton_detection/share/skeleton_detection/config/rtmo_node_direct_realsense.yaml
    devices:
      - /dev/video0
      - /dev/video1
      - /dev/video2
      - /dev/video3
      - /dev/video4
      - /dev/video5
      - /dev/media0
      - /dev/media1
```

```bash
docker compose -f compose.yaml -f compose.realsense.yaml up -d
```

The overridden command selects the packaged `rtmo_node_direct_realsense.yaml`. Only one
process can own the camera, so stop the IoT `realsense` container first
(`compose.yaml` still joins the external `iot_ros-net` network; without the
IoT stack, create it once with `docker network create iot_ros-net`). The
node numbers can change after a replug; recreate the container if so.

### Networking and IPC

The service joins the IoT stack's bridge network `iot_ros-net` (created by the
IoT `./launch.sh`), so DDS discovery reaches the other containers on it
(`realsense`, `zenoh_bridge`, `diagnose`) with the same `ROS_DOMAIN_ID`. It does
not use host networking, so ROS tools on the host itself may not discover it;
inspect topics from inside a container on `iot_ros-net` instead.
`ipc: host` lets Fast DDS use its shared-memory transport across the container
boundary.

### GUI tools (rqt) — opt-in

The image is headless and ships no GUI tools. To look at
`/skeleton_detection/visualization_image`, use the IoT stack's `diagnose`
container, which has `rqt_image_view` / `rqt_graph`, X11 and the same
`iot_ros-net` network:

```bash
# from the iot repo root
./launch.sh up diagnose:=true
xhost +local:docker      # on the HOST
docker compose -f docker/docker-compose.yml --project-directory . \
  exec -it diagnose bash -c \
  "ros2 run rqt_image_view rqt_image_view /skeleton_detection/visualization_image"
```

### Bind mounts

```yaml
volumes:
  - ./config/skeleton_detection_node.yaml:/config/skeleton_detection_node.yaml:ro
```

The source tree is **not** mounted: the container runs the package built into
the image. Only the deployment config is mounted, read-only, so it can be
edited or swapped for another file without rebuilding the image -- restart
the container to apply it. Another deployment (e.g. the IoT compose) mounts
its own YAML the same way and passes `config:=<path>`.

---

## When a Docker image rebuild is needed

The package is built into the image, so **any change to Python source,
launch files or the packaged configs needs an image rebuild** to reach the
container. The one exception is the mounted deployment config, which only
needs a container restart.

Rebuild the image when any of these change:

- `skeleton_detection/`, `launch/`, `config/`, `package.xml`, `setup.py`, `setup.cfg`

- `docker/Dockerfile`
- a pinned dependency version
- `docker/fetch_models.py`, `docker/verify_image.py` or
  `docker/mmcv_ext_stub.py`
- `models/rtmo/` (the config baked to `/opt/models/rtmo/`)

```bash
docker compose build
docker compose up -d
```

---

## Container lifecycle

| Action | Command |
|---|---|
| Build the image | `docker compose build` |
| Start (detached) | `docker compose up -d` |
| Enter | `docker compose exec skeleton_humble bash` |
| Run one command inside | `docker compose exec skeleton_humble bash -lc '<command>'` |
| Open a second shell | run `docker compose exec skeleton_humble bash` again |
| Stop, keep the container | `docker compose stop` |
| Restart | `docker compose restart` — or `docker compose up -d` after a config change |
| Stop and remove | `docker compose down` |

`docker compose down` removes the container, not the image: the weights and
both built packages live in image layers, so recreating the container always
gets them back.

### Development: iterating without rebuilding the image

For a quick edit-and-run loop, mount the source **deliberately** into a
throwaway container and build an overlay on top of the baked-in install:

```bash
docker compose run --rm -v .:/ros2_ws/src/skeleton_detection skeleton_humble bash
# inside:
cd /ros2_ws && colcon build --packages-select skeleton_detection
source /ros2_ws/install/setup.bash    # now overrides /opt/skeleton_detection
```

This is a development convenience only; the deployed image never uses it.

Containers created by an earlier plain `docker run` use the same
`skeleton_humble` name and will make `docker compose up -d` fail with a name
conflict — see [Debugging](debugging.md).
