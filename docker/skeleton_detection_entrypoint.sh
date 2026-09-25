#!/bin/bash
set -e

source /opt/ros/humble/setup.bash

cd /ros2_ws

if [ ! -d "/ros2_ws/src/patrolknight_msgs" ]; then
    git clone \
        https://github.com/patrol-knight/patrolknight_msgs.git \
        /ros2_ws/src/patrolknight_msgs
fi

colcon build --packages-select patrolknight_msgs

source /ros2_ws/install/setup.bash

colcon build --packages-select skeleton_detection

source /ros2_ws/install/setup.bash

exec "$@"