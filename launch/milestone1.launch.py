"""Milestone 1 / 1.5: local test image(s) -> RTMO-M -> SkeletonFrame + JPGs.

Starts rtmo_node first and the image_publisher a few seconds later, so the
model is loaded and subscribed before the first image is published.
Equivalent to running the two `ros2 run` commands by hand.

Both nodes are configured from the installed YAML files in
share/skeleton_detection/config, which enable annotated-image saving
(save_visualization_images: true) and hold the image selection for the
publisher.
To change the input images or the output directory, either edit those YAML
files or run the nodes manually with -p overrides.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share_dir = get_package_share_directory("skeleton_detection")

    rtmo_config_arg = DeclareLaunchArgument(
        "rtmo_config",
        default_value=os.path.join(share_dir, "config", "rtmo_node.yaml"),
        description="Parameter file for rtmo_node.",
    )
    publisher_config_arg = DeclareLaunchArgument(
        "image_publisher_config",
        default_value=os.path.join(share_dir, "config", "image_publisher.yaml"),
        description="Parameter file for image_publisher_node.",
    )

    rtmo_node = Node(
        package="skeleton_detection",
        executable="iot_node",
        name="rtmo_node",
        output="screen",
        parameters=[LaunchConfiguration("rtmo_config")],
    )

    image_publisher_node = TimerAction(
        period=5.0,  # node import + RTMO-M load takes a few seconds.
        actions=[
            Node(
                package="skeleton_detection",
                executable="image_publisher",
                name="image_publisher_node",
                output="screen",
                parameters=[LaunchConfiguration("image_publisher_config")],
            )
        ],
    )

    return LaunchDescription(
        [rtmo_config_arg, publisher_config_arg, rtmo_node, image_publisher_node]
    )
