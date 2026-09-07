

Terminal 1: 
docker exec -it skeleton_humble bash
source /opt/ros/humble/setup.bash
ros2 launch realsense2_camera rs_launch.py

Terminal 2:
docker exec -it skeleton_humble bash
source /opt/ros/humble/setup.bash
ros2 run rqt_gui rqt_gui


