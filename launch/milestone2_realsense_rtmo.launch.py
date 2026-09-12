"""Milestone 2: RealSense D456 -> RTMO-M -> SkeletonFrame, one process.

Starts ONLY the RTMO node. The camera is opened inside that same process with
the RealSense SDK, so the RGB frame never passes through DDS before inference.
Deliberately does NOT launch realsense2_camera.

    ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py

Optional arguments (all forwarded as ROS parameters), e.g.:

    ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
        enable_tracking:=true publish_visualization_image:=true
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
            "enable_tracking",
            default_value="false",
            description="Enable in-process BoT-SORT tracking; makes "
            "PersonSkeleton.person_id a persistent track id.",
        ),
        DeclareLaunchArgument(
            "with_reid",
            default_value="true",
            description="Use OSNet ReID appearance features inside BoT-SORT.",
        ),
        DeclareLaunchArgument(
            "cmc_method",
            default_value="none",
            description="Camera-motion compensation: none, ecc, orb, sift, sof.",
        ),
        DeclareLaunchArgument(
            "track_buffer",
            default_value="90",
            description="Frames a lost track survives BEFORE frame-rate "
            "scaling. BoxMOT uses int(frame_rate / 30.0 * track_buffer); at "
            "tracking_frame_rate 55 this gives max_time_lost=165 frames.",
        ),
        # TEMPORARY TRACKING DEBUG (delete with tracking_debug.py) ---------
        DeclareLaunchArgument(
            "tracking_debug_enabled",
            default_value="false",
            description="TEMPORARY: log why BoT-SORT created each NEW track id "
            "to tracking_debug_path. Off = stock BoxMOT, zero overhead.",
        ),
        DeclareLaunchArgument(
            "tracking_debug_path",
            default_value="/ros2_ws/src/skeleton_detection/output/"
            "tracking_debug.log",
            description="TEMPORARY: new-track diagnostics file. Truncated on "
            "every node launch.",
        ),
        # ------------------------------------------------------------------
        DeclareLaunchArgument(
            "publish_visualization_image",
            default_value="false",
            description="Publish the live annotated visualization image topic.",
        ),
        DeclareLaunchArgument(
            "visualization_width",
            default_value="424",
            description="Width of the published visualization image.",
        ),
        DeclareLaunchArgument(
            "visualization_height",
            default_value="240",
            description="Height of the published visualization image.",
        ),
        DeclareLaunchArgument(
            "visualization_fps",
            default_value="10.0",
            description="Visualization publish rate (<=0 = every frame).",
        ),
        DeclareLaunchArgument(
            "visualization_reliability",
            default_value="best_effort",
            description="QoS reliability for the visualization topic: "
            "best_effort or reliable.",
        ),
        DeclareLaunchArgument(
            "save_visualization_images",
            default_value="false",
            description="Write an annotated file for every processed frame.",
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
        "enable_tracking": LaunchConfiguration("enable_tracking"),
        "with_reid": LaunchConfiguration("with_reid"),
        "cmc_method": LaunchConfiguration("cmc_method"),
        "track_buffer": LaunchConfiguration("track_buffer"),
        # TEMPORARY TRACKING DEBUG
        "tracking_debug_enabled": LaunchConfiguration("tracking_debug_enabled"),
        "tracking_debug_path": LaunchConfiguration("tracking_debug_path"),
        "publish_visualization_image": LaunchConfiguration(
            "publish_visualization_image"
        ),
        "visualization_width": LaunchConfiguration("visualization_width"),
        "visualization_height": LaunchConfiguration("visualization_height"),
        "visualization_fps": LaunchConfiguration("visualization_fps"),
        "visualization_reliability": LaunchConfiguration("visualization_reliability"),
        "save_visualization_images": LaunchConfiguration("save_visualization_images"),
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
