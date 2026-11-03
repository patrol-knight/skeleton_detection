# Skeleton Detection

ROS 2 Humble node (`iot_node`, node name `rtmo_node`) for real-time
multi-person skeletons on an NVIDIA DGX Spark: RTMO-M 2D pose, BoT-SORT +
OSNet ReID tracking, and per-joint 3D positions from an aligned RGB-D stream.

| Topic | Type | Role |
|---|---|---|
| `/skeleton_detection/frame` | `patrolknight_msgs/SkeletonFrame3D` | official: tracked persons, 17 joints × `[X, Y, Z, conf]` in the camera optical frame |
| `/skeleton_detection/debug_frame` | `patrolknight_msgs/SkeletonFrame` | legacy V2.0 message (UV joints, bbox, person position) |
| `/skeleton_detection/visualization_image` | `sensor_msgs/Image` | 2D overlay |

## Prerequisites

- Docker with the NVIDIA container runtime (GPU).
- A running camera publishing `realsense2_camera_msgs/RGBD` on
  `/camera/camera/rgbd`, with depth aligned to colour. **The IoT stack owns
  the camera** (ZED + RGB-D adapter, or realsense2_camera); this repo never
  opens it in the default mode.
- The sibling `patrolknight_msgs` checkout with the V2.1 messages
  (`../patrolknight_msgs`) until they are on its `main`.
- The same `ROS_DOMAIN_ID` in the shells that start both stacks.

## Launch

```bash
# 1. camera only, from the iot repo (no IoT skeleton service)
cd ../iot
docker compose -f docker/docker-compose.yml --project-directory . --profile zed up -d zed

# 2. build this image once (local messages, separate tag)
cd ../skeleton_detection
docker build -f docker/Dockerfile \
  --build-context patrolknight_msgs=../patrolknight_msgs \
  -t skeleton_detection:v2.1-local .

# 3. start the standalone skeleton service
SKELETON_IMAGE=skeleton_detection:v2.1-local docker compose up -d --no-build
docker logs -f skeleton_humble     # expect "First RGBD frame: 848x480 ..."
```

Stop it with `docker compose down`. Parameters come from
`config/skeleton_detection_node.yaml` (mounted; restart to apply). Only one
skeleton service may run: stop the IoT `skeleton` container first if it is up.

## Documentation

- [Skeleton Detection V2.1](docs/Skeleton_Detection_V2.1.md) — current outputs, XYZ layout, depth gate, validation
- [Skeleton Detection V2.0](docs/Skeleton_Detection_V2.0.md) — previous pipeline and legacy interface

Reference: [setup](docs/setup.md) · [Docker & Compose](docs/docker.md) ·
[running](docs/running.md) · [parameters](docs/parameters.md) ·
[launch arguments](docs/launch_arguments.md) ·
[architecture](docs/architecture.md) · [debugging](docs/debugging.md) ·
[implementation notes](docs/implementation.md) ·
[package audit](docker/PACKAGES.md)
