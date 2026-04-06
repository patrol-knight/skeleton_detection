# skeleton_detection

ROS 2 Python package for:

- reading a test MP4 and publishing frames as `sensor_msgs/Image`
- running OpenPifPaf-based skeleton detection on that image stream
- writing annotated video and metadata output

## Nodes

### `camera_reader_node`

Reads an MP4 with OpenCV and publishes frames to an image topic.

Default parameters are in [config/camera_reader.yaml](/home/samdong/patrolknight/anomaly_detection/skeleton_detection/config/camera_reader.yaml):

- `input_topic`: `/dummy_camera/image_raw`
- `video_path`: `/home/samdong/patrolknight/anomaly_detection/data/test_video_1.mp4`
- `publish_fps`: `30.0`
- `loop_video`: `true`

### `skeleton_detection_node`

Subscribes to an image topic, runs pose detection and simple IOU-based tracking, and writes output artifacts.

Default parameters are in [config/skeleton_detection.yaml](/home/samdong/patrolknight/anomaly_detection/skeleton_detection/config/skeleton_detection.yaml):

- `input_topic`: `/dummy_camera/image_raw`
- `checkpoint`: `shufflenetv2k16`
- `target_width`: `800`
- `target_height`: `600`
- `output_video_path`: `../data/skeleton_detection_prediction.mp4`
- `output_metadata_path`: `../data/skeleton_detection_metadata.json`

## Build

```bash
cd /home/samdong/patrolknight/anomaly_detection/skeleton_detection
source /opt/ros/humble/setup.bash
colcon build --packages-select skeleton_detection
source install/setup.bash
```

## Test Workflow

Terminal 1:

```bash
cd /home/samdong/patrolknight/anomaly_detection/skeleton_detection
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch skeleton_detection camera_reader.launch.py
```

Terminal 2:

```bash
cd /home/samdong/patrolknight/anomaly_detection/skeleton_detection
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch skeleton_detection skeleton_detection.launch.py
```

This publishes frames from the test video to `/dummy_camera/image_raw`, then runs skeleton detection on that topic.

## Notes

- `skeleton_detection_node` sets `PYTHONNOUSERSITE=1` in its launch file to avoid mixing ROS/system Python packages with user-site packages.
- `openpifpaf` must be available from the project `pifpaf_env`.
- If you want to use a different test video, change `video_path` in [config/camera_reader.yaml](/home/samdong/patrolknight/anomaly_detection/skeleton_detection/config/camera_reader.yaml).
