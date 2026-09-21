# Debugging

Failures actually seen on this setup, and what to do about them.

Commands assume you are inside the container with both setup files sourced:

```bash
docker compose exec skeleton_humble bash
source /opt/ros/humble/setup.bash
source /ros2_ws/install/setup.bash
```

---

## Camera

### The camera is not detected

The node validates the requested profile against the device's advertised
profiles at start-up. An unsupported width/height/fps/format combination fails
with the **supported list printed**, rather than being silently substituted —
read that list and match it.

Check the device is visible from the container:

```bash
docker compose exec skeleton_humble bash -lc \
  'python3 -c "import pyrealsense2 as rs; ctx = rs.context(); print([d.get_info(rs.camera_info.name) for d in ctx.devices])"'
```

An empty list means the container cannot see the camera at all. Check, in
order:

1. `lsusb` on the **host** — is the D456 enumerated?
2. Is the container running with `privileged: true` and `/dev:/dev`? Both come
   from `compose.yaml`; a container started by an older, hand-written
   `docker run` may be missing them.
3. Did the camera re-enumerate? A RealSense moves to a new `/dev/bus/usb/...`
   path on every replug. The `/dev` bind mount is used precisely so this keeps
   working — but a container started **before** the host saw the device may
   still need `docker compose restart`.
4. USB 3 vs USB 2. 848 × 480 @ 60 needs USB 3.x; a USB 2 port or cable will not
   advertise that profile.

With several cameras attached, pin one with `realsense_serial`.

### Verifying the model loads without a camera

This loads the config and checkpoint and then fails at camera open, which is
enough to prove the baked assets are correct:

```bash
ros2 run skeleton_detection iot_node --ros-args -p input_mode:=realsense
```

### RealSense permissions

The container runs as root with `/dev` mounted, which is enough for libusb to
claim the camera. **No udev rules are installed inside the container.** If you
run the node outside this container, you need the standard RealSense udev rules
on the host instead.

---

## Docker and Compose

### `docker compose up -d` fails with a name conflict

```text
Conflict. The container name "/skeleton_humble" is already in use by container "..."
```

A container with that name already exists — typically from an earlier
hand-written `docker run` or an older Compose version whose labels the current
Compose no longer matches. Check what is there before removing anything:

```bash
docker ps -a --filter name=skeleton_humble
docker compose ps -a
```

If `docker ps -a` shows the container but `docker compose ps -a` does not,
Compose cannot adopt it. Remove it and let Compose recreate it:

```bash
docker rm -f skeleton_humble
docker compose up -d
```

Nothing of value is lost: your source is on the host bind mount and the model
weights are in the image. You will need a fresh `colcon build`, because
`/ros2_ws/build`, `/ros2_ws/install` and `/ros2_ws/log` are container-local.

### `docker compose exec` says the service is not running

Same cause as above — Compose is not associating the running container with the
service. Verify with `docker ps`, and either recreate the container as above or
fall back to plain `docker exec -it skeleton_humble bash` for the moment.

### Checking the Compose file itself

```bash
docker compose config
```

This prints the fully resolved configuration. Confirm that `build.context`
points at the repository root and `build.dockerfile` at `docker/Dockerfile` —
the build copies from both `docker/` and `models/`, so a narrower context
breaks it.

### GPU is not available inside the container

```bash
docker compose exec skeleton_humble bash -lc \
  'python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"'
```

`False` means the NVIDIA Container Toolkit is missing or misconfigured on the
host. As a stopgap, run on CPU with `device:=cpu` — expect a large slowdown.

### USB and `/dev` problems

The container gets raw device access through `privileged: true` plus
`/dev:/dev`, not through an explicit `devices:` mapping, because a RealSense
re-enumerates to a new path on replug. If you have edited `compose.yaml` and
removed either, USB access stops working.

---

## X11 / rqt

`rqt_image_view` shows nothing, or fails with a display error.

1. Check `DISPLAY` inside the container:
   ```bash
   docker compose exec skeleton_humble bash -lc 'echo $DISPLAY'
   ```
   `compose.yaml` passes the host's `DISPLAY` through, falling back to `:1`.
2. Allow the container to talk to the host X server:
   ```bash
   xhost +local:docker      # run on the HOST
   ```
3. Confirm `/tmp/.X11-unix` is bind-mounted (`docker compose config`).
4. If the display works but the image window stays blank, the topic is probably
   not being published — visualization is **off by default**. Start with
   `publish_visualization_image:=true` and check:
   ```bash
   ros2 topic hz /skeleton_detection/visualization_image
   ```
5. A viewer subscribing with **reliable** QoS sees nothing from the default
   `best_effort` publisher. Switch the publisher with
   `visualization_reliability:=reliable`.

---

## DDS / topics

### Host-side tools do not see the topics

The container uses `network_mode: host` so DDS discovery reaches host tools
without multicast translation, and `ipc: host` so Fast DDS shared memory works
across the boundary. If either is removed from `compose.yaml`, discovery
breaks.

Check from inside the container first — if the topic is not there, it is not a
DDS problem:

```bash
ros2 node list      # expect /rtmo_node
ros2 topic list     # expect /skeleton_detection/frame
```

Then from the host with ROS 2 sourced. If it works inside but not outside,
confirm both sides use the same `ROS_DOMAIN_ID` and the same RMW
implementation.

### The topic exists but nothing arrives

- `ros2 topic hz /skeleton_detection/frame` — is the pipeline producing at all?
- With `person_score_threshold` too high, frames publish with zero persons.
- The visualization topic only exists when
  `publish_visualization_image:=true`.

---

## Performance

### Low FPS

Measure first. The node logs a statistics line every `stats_log_period_sec`
with `capture_fps`, `skeleton_fps`, `rtmo_ms`, `track_ms` and the
`captured` / `processed` / `dropped` counters, and prints a mean/median/p95
summary on shutdown. Run a fixed benchmark:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  run_duration_sec:=30.0
```

Expected on this hardware: RTMO only ~50–55 FPS, + BoT-SORT without ReID
~50 FPS, + ReID ~38 FPS with 1–2 people.

Things that cost throughput, in rough order:

1. **ReID** — the largest single cost. Try `with_reid:=false`; motion-only
   BoT-SORT adds very little.
2. **`save_visualization_images:=true`** — writes a file for **every**
   processed frame and is **not** rate limited. The node warns about this in
   realsense mode. Keep it off during live runs.
3. **A high `visualization_fps`** — drawing happens only when a sink is due, so
   10 Hz skips roughly 4 of every 5 frames; 50 Hz does not.
4. **Running on CPU** — check `device` is `cuda:0`.

A large `dropped` count is **not** a bug by itself: the capture buffer is
latest-frame-wins, so stale frames are intentionally discarded rather than
queued. It means processing is slower than the camera, and latency stays
bounded instead of growing.

The first inference is reported separately in the summary as the CUDA warm-up
cost; do not read it as steady-state latency.

---

## Tracking

### `person_id` changes every frame

Tracking is off. With `enable_tracking:=false`, `person_id` is the frame-local
detection index by design. Turn tracking on:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  enable_tracking:=true
```

The node logs which semantics are in force at start-up, and the visualization
overlay carries a matching legend, so a saved image is never ambiguous.

### ID switches

Start by reading the start-up log — `SkeletonTracker` prints every association
gate it is running with:

```text
BoT-SORT ready (with_reid=..., cmc_method=..., frame_rate=..., track_buffer=90,
max_time_lost=165 frames ~3.00s)
Association gates: proximity_thresh=0.70 (ReID eligible for IoU >= 0.30 ...),
appearance_thresh=0.25, match_thresh=0.80, track_high_thresh=0.50,
new_track_thresh=0.60
```

Then:

- **Switch after a short disappearance** → the track buffer expired.
  `max_time_lost = int(frame_rate / 30.0 * track_buffer)`, so with the defaults
  (55 / 90) a lost track survives 165 frames ≈ 3.0 s. Raise `track_buffer`.
- **Switch when two people cross** → appearance is being gated out. Remember
  `proximity_thresh` is an **IoU distance**: *raising* it relaxes the gate.
  0.70 keeps ReID usable down to `IoU >= 0.30`; lowering it to 0.30 would
  *tighten* it to `IoU >= 0.70`.
- **Switch when someone is partly hidden** → try the occlusion-aware
  extension (below).
- **Camera is moving** → `cmc_method: none` assumes a static camera.
  Re-evaluate CMC for a robot-mounted camera.

`track_buffer` and `proximity_thresh` are the only two association-affecting
values that differ from the BoxMOT 19.0.0 defaults; changing anything else
means departing from a known-good baseline.

### ReID behaviour

- ReID only runs when **both** `enable_tracking` and `with_reid` are true.
- The checkpoint must exist. A missing file fails at start-up with an explicit
  message naming `/opt/models/reid/osnet_x0_25_msmt17.pt`; rebuild the image or
  point `reid_checkpoint` at a real file.
- A weak appearance match is still rejected on its own merits by
  `appearance_thresh` (0.25), independently of the proximity gate.

### Occlusion-aware tracking diagnostics

It requires `enable_tracking:=true`; on its own it does nothing. When it is on,
the node logs a warning so it can never be on unnoticed:

```text
EXPERIMENTAL occlusion-aware tracking is ON: OCCLUDED observations no longer
update the Kalman motion state or the ReID appearance state.
```

and `SkeletonTracker` logs the thresholds actually in use.

- **Nothing is ever classified `OCCLUDED`** → `visible_ratio_threshold` (0.50)
  is too low for your scene, or `keypoint_visibility_threshold` (0.30) is so
  low that joints count as visible when they are not. `visible_ratio` is the
  fraction of RTMO's 17 **per-joint** scores at or above that threshold — the
  aggregate person score is not used.
- **Warnings about unusable keypoint scores** → the visible-ratio criterion is
  being **skipped** and the detection stays `NORMAL`. Missing keypoints are
  never read as zero confidence. The warning is throttled; a count is kept.
- **The bbox-width numbers in the debug log look wrong** → they are
  `INFORMATIONAL ONLY` and classify nothing. Width was deliberately removed as
  a signal because a person turning sideways drops to ~0.45 of their frontal
  box width with 16/17 keypoints still visible.
- To rule the feature out entirely, run with
  `occlusion_aware_tracking:=false` — that constructs a stock `BotSort` with
  zero overhead.

---

## Depth and position

### `position` / `depth` are always `NaN`

That is the documented "unavailable" value — it is never `0`. Causes, in order
of likelihood:

1. `realsense_enable_depth:=false`.
2. `input_mode: ros_topic` — the offline path has no depth stream and no
   intrinsics, so `position`/`depth` are **always** `NaN` there.
3. No visible keypoint produced a valid depth sample: the person is out of the
   depth range, or the depth is all holes. Lower
   `depth_keypoint_score_threshold` so more joints vote.
4. `depth_min_m` / `depth_max_m` are rejecting everything. Set them to `0.0` to
   disable the bounds.

Confirm depth is actually on — the node logs it at start-up:

```text
Depth ENABLED: z16 aligned to colour in RealSenseCapture, depth_scale=0.001000 m/unit; ...
```

Consumers must test with `math.isnan()`, not against `0`.

### `depth` disagrees with the RealSense viewer

Expected. `PersonSkeleton.depth` is the **Euclidean** distance
`sqrt(X² + Y² + Z²)`; the RealSense depth image value is the **optical-axis Z**,
which is `position.z`. They agree only near the image centre and diverge
towards the edges.

### The depth looks like the background, not the person

Check the start-up log for the intrinsics line and the depth-scale line. If
`depth_scale` is wrong the whole scale is wrong; if the values track the wall
behind the person, the two-stage median is voting on background pixels — raise
`depth_keypoint_score_threshold` so only confidently-visible joints vote.

---

## Build problems

### Stale ROS build/install files

Symptoms: a parameter you added is "not declared", an old module name is still
imported, a launch file that no longer exists is still found, or a renamed
module resolves to the deleted one.

`colcon` installs the Python package by **copying** it into
`install/`, so a deleted or renamed source file can survive there. Clean
rebuild:

```bash
cd /ros2_ws
rm -rf build install log
source /opt/ros/humble/setup.bash
colcon build --packages-select skeleton_detection
source /ros2_ws/install/setup.bash
```

Always re-`source` `install/setup.bash` in **every** open shell after a
rebuild; an old shell keeps pointing at the previous install tree.

### `ros2 run` cannot find the executable

The executable is `iot_node` (the node it starts is named `rtmo_node`). If
`ros2 run skeleton_detection iot_node` fails, the workspace is not sourced or
the build did not complete. Check:

```bash
ls /ros2_ws/install/skeleton_detection/lib/skeleton_detection
```

You should see `iot_node` and `image_publisher`.

### Changes to the host source do not take effect

Editing Python on the host is immediately visible inside the container (the
repository is bind-mounted), but ROS runs the **installed copy**. Re-run
`colcon build` and re-source. An image rebuild is *not* needed for source
edits — see [Docker & Compose](docker.md) for when it is.

---

## Tests

Run them inside the container:

```bash
cd /ros2_ws/src/skeleton_detection
python3 -m pytest test -q -p no:anyio
```

### The `-p no:anyio` flag is required

Without it, pytest fails to start:

```text
ModuleNotFoundError: No module named '_pytest.scope'
  ... anyio/pytest_plugin.py, line 14, in <module>
```

The container has **pytest 6.2.5** from apt and **anyio 4.15.1** from pip
(pulled in transitively by boxmot → huggingface-hub). anyio registers a
`pytest11` entry point whose plugin imports `_pytest.scope`, which only exists
in pytest 7+. Pytest auto-loads the plugin at start-up and dies before
collecting anything.

`-p no:anyio` disables just that plugin. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
works too. This is a packaging mismatch in the image, not a problem with the
tests — the suite passes once the plugin is skipped.

---

## Fatal runtime errors

### "RTMO hit a stubbed mmcv native op"

```text
RTMO hit a stubbed mmcv native op: ... A full mmcv build is required;
refusing to publish results.
```

The image ships `mmcv-lite` with an **import-only** `mmcv._ext` stub
(`docker/mmcv_ext_stub.py`) that raises if an op is actually called. RTMO uses
no mmcv native op, so this means the model path changed — a different config or
checkpoint that does use one. The node refuses to publish rather than emit a
fabricated result. See [Implementation notes](implementation.md).

### The node exits immediately at start-up

`_read_parameters` validates eagerly and raises with an explicit message. Known
checks: `input_mode` must be `realsense` or `ros_topic`;
`visualization_reliability` must be `best_effort` or `reliable`;
`visualization_width`/`height` must be `>= 1`; `normal_bbox_history_size` and
`min_normal_width_samples` must be `>= 1`.

Also check the parameter **types**: `run_duration_sec` is a DOUBLE, so
`run_duration_sec:=30` is rejected and `run_duration_sec:=30.0` is correct. The
same holds for `visualization_fps`, `tracking_frame_rate`,
`stats_log_period_sec` and `proximity_thresh`.
