import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_config = os.path.join(
        get_package_share_directory("skeleton_detection"),
        "config",
        "camera_reader.yaml",
    )

    config_arg = DeclareLaunchArgument(
        "config",
        default_value=default_config,
        description="Path to the camera reader parameter file.",
    )

    node = Node(
        package="skeleton_detection",
        executable="camera_reader_node",
        name="camera_reader_node",
        output="screen",
        parameters=[LaunchConfiguration("config")],
    )

    return LaunchDescription([config_arg, node])
