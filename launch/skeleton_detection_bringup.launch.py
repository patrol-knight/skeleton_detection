"""Live bringup: starts the RTMO node (``iot_node``) with one parameter file.

Starts ONLY the RTMO node. This file NEVER launches realsense2_camera; in
``ros_camera`` mode that driver is expected to be running already, typically
in another container.

The ``config`` YAML is the single source of truth for every node parameter,
including ``input_mode``; this file does not know or care which mode the
selected YAML uses. It has no per-parameter arguments.

Default: the packaged ``config/rtmo_node.yaml`` (``input_mode: ros_camera``,
consuming the RGBD topic of an external realsense2_camera driver):

    ros2 launch skeleton_detection skeleton_detection_bringup.launch.py

Any other parameter file, e.g. the packaged
``config/rtmo_node_direct_realsense.yaml`` (the D456 opened inside this same
process) or a config mounted into the container:

    ros2 launch skeleton_detection skeleton_detection_bringup.launch.py \\
        config:=/config/rtmo_node.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share_dir = get_package_share_directory("skeleton_detection")
    default_config = os.path.join(share_dir, "config", "rtmo_node.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config",
                default_value=default_config,
                description="Parameter file for rtmo_node; the single source "
                "of truth for every node parameter, including input_mode.",
            ),
            Node(
                package="skeleton_detection",
                executable="iot_node",
                name="rtmo_node",
                output="screen",
                emulate_tty=True,
                parameters=[LaunchConfiguration("config")],
            ),
        ]
    )
