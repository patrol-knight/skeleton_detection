"""Milestone 2: RealSense D456 -> RTMO-M -> SkeletonFrame, one process.

Starts ONLY the RTMO node. The camera is opened inside that same process with
the RealSense SDK, so the RGB frame never passes through DDS before inference.
Deliberately does NOT launch realsense2_camera.

    ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py

Optional arguments (all forwarded as ROS parameters), e.g.:

    ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
        run_duration_sec:=30.0 publish_debug_image:=true
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    share_dir = get_package_share_directory("skeleton_detection")
    default_config = os.path.join(share_dir, "config", "rtmo_node_realsense.yaml")

    arguments = [
        DeclareLaunchArgument(
            "config",
            default_value=default_config,
            description="Parameter file for rtmo_node in realsense mode.",
        ),
        DeclareLaunchArgument(
            "realsense_width", default_value="848", description="Colour width."
        ),
        DeclareLaunchArgument(
            "realsense_height", default_value="480", description="Colour height."
        ),
        DeclareLaunchArgument(
            "realsense_fps", default_value="60", description="Colour frame rate."
        ),
        DeclareLaunchArgument(
            "device", default_value="cuda:0", description="Torch device for RTMO."
        ),
        DeclareLaunchArgument(
            "publish_debug_image",
            default_value="false",
            description="Publish the annotated debug image (costs throughput).",
        ),
        DeclareLaunchArgument(
            "save_debug_images",
            default_value="false",
            description="Write annotated JPGs to disk (costs throughput).",
        ),
        DeclareLaunchArgument(
            "run_duration_sec",
            default_value="0.0",
            description="Stop automatically after N seconds (0 = run forever).",
        ),
    ]

    # The YAML file supplies the full parameter set; the launch arguments
    # override the few knobs worth changing from the command line.
    overrides = {
        "input_mode": "realsense",
        "realsense_width": LaunchConfiguration("realsense_width"),
        "realsense_height": LaunchConfiguration("realsense_height"),
        "realsense_fps": LaunchConfiguration("realsense_fps"),
        "device": LaunchConfiguration("device"),
        "publish_debug_image": LaunchConfiguration("publish_debug_image"),
        "save_debug_images": LaunchConfiguration("save_debug_images"),
        "run_duration_sec": LaunchConfiguration("run_duration_sec"),
    }

    rtmo_node = Node(
        package="skeleton_detection",
        executable="rtmo_node",
        name="rtmo_node",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("config"), overrides],
    )

    return LaunchDescription(arguments + [rtmo_node])
