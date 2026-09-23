#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

cd /ros2_ws
colcon build --packages-select skeleton_detection

source /ros2_ws/install/setup.bash

exec "$@"