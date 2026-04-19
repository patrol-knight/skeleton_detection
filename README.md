# skeleton_detection

ROS 2 package for running OpenPifPaf-based skeleton detection on an image stream and producing:

- raw per-frame skeleton metadata on `/skeleton_detection/frame`
- annotated image frames on `/skeleton_output`
- an annotated MP4 on disk
- a JSON metadata log on disk

The package supports two common workflows:

1. publish frames from a local MP4 and run skeleton detection on that stream
2. subscribe to a live RealSense color topic and run skeleton detection in real time

## Outputs

By default, the detector uses the parameters in [config/skeleton_detection.yaml]:

- `/skeleton_detection/frame` publishes one `skeleton_detection/msg/SkeletonFrame` message per input frame.
- `/skeleton_output` publishes a `sensor_msgs/Image` containing the annotated frame with skeletons, bounding boxes, and tracking IDs drawn on top.
- `output_fps` controls the saved MP4 playback rate and the top-level metadata fps value. It does not throttle the detector callback.

## Build

From the workspace root:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select skeleton_detection
source install/setup.bash
```

## Common Detector Launch

The detector itself is launched the same way in both workflows:

```bash
source src/skeleton_detection_pifpaf/pifpaf_env/bin/activate
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch skeleton_detection skeleton_detection.launch.py
```

The two environment variables above keep the detector headless and avoid Qt plugin crashes while it renders annotated frames internally.

## Option 1: Local MP4 Through `camera_reader_node`

Use this when you want to run skeleton detection on a static video file on disk.

1. Open [config/camera_reader.yaml]and set the MP4 path you want to read.
2. Make sure the detector `input_topic` in [config/skeleton_detection.yaml]matches the image topic published by the camera reader. A common choice for offline playback is `/dummy_camera/image_raw`.
3. Launch the detector in a one terminal.
4. Launch the MP4 reader in another terminal.

Terminal 1, detector:

```bash
source src/skeleton_detection_pifpaf/pifpaf_env/bin/activate
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch skeleton_detection skeleton_detection.launch.py
```

Terminal 2, MP4 reader:

```bash
source src/skeleton_detection_pifpaf/pifpaf_env/bin/activate
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch skeleton_detection camera_reader.launch.py
```

## Option 2: Live RealSense Camera Input

Use this when a RealSense driver is already publishing a live color stream to `/camera/camera/color/image_raw`.

1. Start your RealSense ROS driver so the live color topic is available.
2. Keep `input_topic: /camera/camera/color/image_raw` in [config/skeleton_detection.yaml]
3. Launch the detector.

Example detector terminal:

```bash
source src/skeleton_detection_pifpaf/pifpaf_env/bin/activate
source /opt/ros/humble/setup.bash
source install/setup.bash
export QT_QPA_PLATFORM=offscreen
export MPLBACKEND=Agg
ros2 launch skeleton_detection skeleton_detection.launch.py
```

If you use Ubuntu 22.04:

```bash
ros2 run realsense2_camera realsense2_camera_node
```

If you use Intel's standard ROS RealSense driver, a common launch looks like:

```bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py
```


If your camera publishes a different topic name, update `input_topic` in [config/skeleton_detection.yaml]to match it.

## View Raw And Processed Images In One `rqt` Window

The most reliable setup is:

- run the detector in the shell you use for OpenPifPaf
- run `rqt` in a separate clean ROS shell

Terminal 3, `rqt` viewer shell:

```bash
source src/skeleton_detection_pifpaf/pifpaf_env/bin/activate 
src/skeleton_detection_pifpaf/pifpaf_env/bin/python3 /opt/ros/humble/bin/rqt

```

Inside `rqt`:

1. Open `Plugins > Visualization > Image View`.
2. Open `Plugins > Visualization > Image View` a second time.
3. Dock the two image panes side by side in the same `rqt` window.
4. Set the left pane topic to `/camera/camera/color/image_raw`.
5. Set the right pane topic to `/skeleton_output`.
6. Save the perspective if you want to reopen the same layout later.

This gives you a single `rqt` window with the raw input image on the left and the processed skeleton output on the right.


# Trouble Shoot

```bash
which python3
```
If the command returns
```bash
/home/smores/anaconda3/bin/python3
```

run 
```bash
export VIRTUAL_ENV=/home/smores/anomaly_detection_ws/src/skeleton_detection_pifpaf/pifpaf_env
export PATH="$VIRTUAL_ENV/bin:$PATH"
hash -r
which python3
```

