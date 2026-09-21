# Docker & Compose

How the container image is built and how the development container is run.

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
lives.

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
| `command` | `sleep infinity` | the container starts **idle** |
| `working_dir` | `/ros2_ws` | the Dockerfile `WORKDIR` and colcon workspace root |
| `privileged` | `true` | raw USB access for the RealSense |
| `network_mode` | `host` | DDS discovery reaches host-side tools |
| `ipc` | `host` | Fast DDS shared-memory transport across the boundary |
| `deploy.resources.reservations.devices` | `driver: nvidia`, `count: all` | equivalent of `--gpus all` |
| `environment.DISPLAY` | `${DISPLAY:-:1}` | X11 for `rqt_image_view` |

**`compose up` does not start the pipeline.** It only brings the environment
up; you enter the container and launch ROS yourself.

### GPU access

The `deploy` block is the Compose equivalent of `--gpus all` and requires the
NVIDIA Container Toolkit on the host. RTMO-M and the OSNet ReID backbone both
run on `cuda:0` by default (`device` parameter).

### RealSense device access

Granted with `privileged: true` plus a `/dev:/dev` bind mount, rather than an
explicit `devices:` mapping. A RealSense re-enumerates to a new
`/dev/bus/usb/...` path on every replug, so a fixed device mapping would break
on each reconnect. No udev rules are installed inside the container; running as
root with `/dev` mounted is sufficient for libusb to claim the camera.

### Host networking and IPC

`network_mode: host` lets DDS discovery reach `ros2 topic echo` /
`rqt_image_view` running on the host without multicast translation.
`ipc: host` lets Fast DDS use its shared-memory transport across the container
boundary.

### X11 / rqt support

`/tmp/.X11-unix` is bind-mounted and `DISPLAY` is passed through, falling back
to `:1` when the host has no `DISPLAY` set. That is enough for
`rqt_image_view` and `rqt_gui` inside the container.

### Bind mounts

```yaml
volumes:
  - .:/ros2_ws/src/skeleton_detection
  - /dev:/dev
  - /tmp/.X11-unix:/tmp/.X11-unix
```

The repository is mounted at `/ros2_ws/src/skeleton_detection` — **source
only**. Model weights come from the image at `/opt/models`, never from the
mount, so no checkpoints ever land on the host.

---

## When a Docker image rebuild is needed

Because the repository is bind-mounted, **editing Python, YAML, launch files or
message definitions on the host takes effect inside the container
immediately**. You do not rebuild the image for source edits — you re-run
`colcon build` inside the container (see
[Running the pipeline](running.md)).

Rebuild the image only when something the image itself owns changes:

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

`docker compose down` removes the container, not the image and not
`/opt/models`: the weights live in an image layer, so recreating the container
always gets them back. It **does** discard the container-local
`/ros2_ws/build`, `/ros2_ws/install` and `/ros2_ws/log`, so the next start
needs a fresh `colcon build`. Your source is on the host and is untouched.

Containers created by an earlier plain `docker run` use the same
`skeleton_humble` name and will make `docker compose up -d` fail with a name
conflict — see [Debugging](debugging.md).
