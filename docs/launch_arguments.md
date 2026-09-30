# Launch arguments

`skeleton_detection_bringup.launch.py`, the only launch file in the package,
has exactly **one** argument:

| Argument | Default | Description |
|---|---|---|
| `config` | `<share>/config/rtmo_node.yaml` | Parameter file for `rtmo_node`. Any path works, e.g. a file bind-mounted into the container. |

`<share>` is `get_package_share_directory("skeleton_detection")`, i.e.
`/opt/skeleton_detection/share/skeleton_detection` in the image.

The launch file starts **only the skeleton-detection node** (`iot_node`, node
name `rtmo_node`) and passes it the selected YAML unchanged. It never launches
`realsense2_camera`; in `ros_camera` mode that driver is expected to already
be running elsewhere.

---

## The YAML is the single source of truth

Every node parameter — `input_mode` included — comes from the selected
config file. There are no per-parameter launch arguments: `enable_tracking:=true`
and the like are **not** part of the launch interface (ROS 2 accepts unknown
`name:=value` pairs on the command line without error, but the launch file
never forwards them to the node).

Packaged configs, in `<share>/config/`:

| File | `input_mode` | Use |
|---|---|---|
| `rtmo_node.yaml` | `ros_camera` | **default**: external `realsense2_camera` driver; tracking + ReID + occlusion-aware tracking + visualization on (also what `compose.yaml` mounts) |
| `rtmo_node_direct_realsense.yaml` | `realsense` | optional: D456 opened in this process; tracking and visualization off |
| `offline_rtmo_node.yaml` | `ros_topic` | offline image regression (used with `ros2 run`, see [Running the pipeline](running.md)) |
| `offline_image_publisher.yaml` | — | the `image_publisher` node that feeds `offline_rtmo_node.yaml` |

```bash
# default: packaged rtmo_node.yaml (ros_camera)
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py

# any other file, e.g. direct camera mode
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \
  config:=<share>/config/rtmo_node_direct_realsense.yaml
```

### Changing a value

Copy a packaged config, edit it, and select the copy:

```bash
cp /opt/skeleton_detection/share/skeleton_detection/config/rtmo_node.yaml /tmp/my.yaml
# edit /tmp/my.yaml, e.g.  visualization_fps: 30.0
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py config:=/tmp/my.yaml
```

For a deployment, bind-mount the file into the container instead (as
`compose.yaml` does) — editing it then needs only a container restart, not an
image rebuild.

Every parameter, its type, node default and meaning is in the
[Parameter reference](parameters.md).

### Types matter

A parameter whose default is written with a decimal point is a **DOUBLE**, and
an integer literal is rejected as the wrong type. Write
`run_duration_sec: 30.0`, never `30`. The same applies to `visualization_fps`,
`proximity_thresh`, `keypoint_visibility_threshold`,
`visible_ratio_threshold` and the other float parameters.

You can confirm the interface from the installed launch file:

```bash
ros2 launch skeleton_detection skeleton_detection_bringup.launch.py --show-args
```
