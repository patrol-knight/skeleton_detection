"""Milestone 2: RealSense D456 -> RTMO-M -> SkeletonFrame, one process.

Starts ONLY the RTMO node. The camera is opened inside that same process with
the RealSense SDK, so the RGB frame never passes through DDS before inference.
Deliberately does NOT launch realsense2_camera.

    ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py

Optional arguments (all forwarded as ROS parameters), e.g.:

    ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
        enable_tracking:=true publish_visualization_image:=true

Occlusion-aware tracking (experimental, off by default):

    ros2 launch skeleton_detection milestone2_realsense_rtmo.launch.py \
        enable_tracking:=true occlusion_aware_tracking:=true
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
            "realsense_enable_depth",
            default_value="true",
            description="Open the Z16 depth stream and align it to colour, so "
            "PersonSkeleton.position carries the person's XYZ in the colour "
            "optical frame and depth the Euclidean distance, in meters. "
            "false = colour only, position/depth published as NaN.",
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
            "proximity_thresh",
            default_value="0.70",
            description="IoU-DISTANCE gate above which BoT-SORT discards the "
            "ReID distance. iou_dist = 1 - IoU, so 0.70 keeps appearance "
            "usable down to IoU 0.30 and RAISING this value relaxes the gate "
            "(0.30 would tighten it to IoU >= 0.70).",
        ),
        DeclareLaunchArgument(
            "track_buffer",
            default_value="90",
            description="Frames a lost track survives BEFORE frame-rate "
            "scaling. BoxMOT uses int(frame_rate / 30.0 * track_buffer); at "
            "tracking_frame_rate 55 this gives max_time_lost=165 frames.",
        ),
        # OCCLUSION-AWARE TRACKING (delete with occlusion_tracking.py) -----
        DeclareLaunchArgument(
            "occlusion_aware_tracking",
            default_value="false",
            description="EXPERIMENTAL: classify each matched detection as "
            "NORMAL or OCCLUDED; an OCCLUDED one skips the Kalman measurement "
            "update and freezes the ReID feature. Prediction and the track "
            "buffer are unaffected. false = stock BoT-SORT behaviour.",
        ),
        DeclareLaunchArgument(
            "keypoint_visibility_threshold",
            default_value="0.30",
            description="A COCO-17 joint counts as visible at or above this "
            "RTMO per-keypoint score.",
        ),
        DeclareLaunchArgument(
            "visible_ratio_threshold",
            default_value="0.50",
            description="visible_ratio below this marks the detection "
            "OCCLUDED.",
        ),
        DeclareLaunchArgument(
            "normal_bbox_history_size",
            default_value="15",
            description="INFORMATIONAL ONLY: per-track ring buffer of bbox "
            "widths shown in the debug log. Classifies nothing.",
        ),
        DeclareLaunchArgument(
            "min_normal_width_samples",
            default_value="5",
            description="INFORMATIONAL ONLY: NORMAL widths needed before the "
            "debug log prints a bbox width ratio at all.",
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
            "draw_person_xyz",
            default_value="false",
            description="DEBUG: append each person's camera-frame XYZ [m] to "
            "the overlay label.",
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
        "realsense_enable_depth": LaunchConfiguration("realsense_enable_depth"),
        "device": LaunchConfiguration("device"),
        "enable_tracking": LaunchConfiguration("enable_tracking"),
        "with_reid": LaunchConfiguration("with_reid"),
        "cmc_method": LaunchConfiguration("cmc_method"),
        "track_buffer": LaunchConfiguration("track_buffer"),
        "proximity_thresh": LaunchConfiguration("proximity_thresh"),
        # OCCLUSION-AWARE TRACKING
        "occlusion_aware_tracking": LaunchConfiguration("occlusion_aware_tracking"),
        "keypoint_visibility_threshold": LaunchConfiguration(
            "keypoint_visibility_threshold"
        ),
        "visible_ratio_threshold": LaunchConfiguration("visible_ratio_threshold"),
        "normal_bbox_history_size": LaunchConfiguration("normal_bbox_history_size"),
        "min_normal_width_samples": LaunchConfiguration("min_normal_width_samples"),
        "publish_visualization_image": LaunchConfiguration(
            "publish_visualization_image"
        ),
        "visualization_width": LaunchConfiguration("visualization_width"),
        "visualization_height": LaunchConfiguration("visualization_height"),
        "visualization_fps": LaunchConfiguration("visualization_fps"),
        "visualization_reliability": LaunchConfiguration("visualization_reliability"),
        "draw_person_xyz": LaunchConfiguration("draw_person_xyz"),
        "save_visualization_images": LaunchConfiguration("save_visualization_images"),
        "run_duration_sec": LaunchConfiguration("run_duration_sec"),
    }

    rtmo_node = Node(
        package="skeleton_detection",
        executable="iot_node",
        name="rtmo_node",
        output="screen",
        emulate_tty=True,
        parameters=[LaunchConfiguration("config"), overrides],
    )

    return LaunchDescription(arguments + [rtmo_node])
