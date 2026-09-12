"""ROS 2 node orchestrating the in-process skeleton perception pipeline.

One node, one process. This module wires the components together and owns the
ROS surface (parameters, publishers, threads, lifecycle); the actual work lives
in focused modules::

    realsense_capture.RealSenseCapture   camera + latest-frame-wins buffer
                                         (depth aligned to colour lives here)
    rtmo_inference.RTMOInference         RTMO-M model + result parsing
    tracking.SkeletonTracker             BoT-SORT + OSNet ReID
    person_depth                         aligned depth -> per-person depth [m]
    message_builder                      internal -> ROS message conversion
    visualization                        drawing + the two visualization sinks
    pipeline_stats.PipelineStats         timing/throughput accounting

Per-frame order (tracking runs BEFORE messages are built, so person_id is
already the persistent track id at publication time)::

    frame -> RTMO -> [PersonDetection] -> tracker.update() -> track_id attached
          -> person_depth -> detection.depth [m]
          -> SkeletonFrame -> /skeleton_detection/frame -> optional visualization

Input modes (``input_mode`` parameter):

``realsense`` -- the D456 is opened by THIS process with pyrealsense2; the BGR
frame is handed to RTMO and then to BoT-SORT by reference. No RGB image ever
crosses DDS on the input path.

``ros_topic`` -- regression path: frames arrive on a sensor_msgs/Image topic.

Outputs::

    /skeleton_detection/frame               SkeletonFrame, robot-facing, ~55 Hz
    /skeleton_detection/visualization_image sensor_msgs/Image, human-facing,
                                            rate limited and downscaled
"""

import os
import threading
import time
from dataclasses import replace
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from skeleton_detection.msg import SkeletonFrame

from .message_builder import build_skeleton_frame, person_id_semantics
from .person_depth import NO_DEPTH, DepthParams, compute_person_depth
from .pipeline_stats import PipelineStats
from .realsense_capture import RealSenseCapture, RealSenseCaptureError
from .rtmo_inference import PersonDetection, RTMOInference, RTMOInferenceError
from .occlusion_tracking import OcclusionParams
from .tracking import DEFAULT_REID_CHECKPOINT, SkeletonTracker, TrackerInitError
from .visualization import (
    VisualizationWriter,
    draw_skeleton_overlay,
    joint_visibility_summary,
    legend_for,
)

DEFAULT_VISUALIZATION_DIR = "/ros2_ws/src/skeleton_detection/output/visualizations"

# Model assets are baked into the Docker image at /opt/models (see
# docker/fetch_models.py), so a pulled image is self-contained and does not
# depend on the gitignored data/ directory.
DEFAULT_MODEL_CONFIG = os.environ.get("RTMO_MODEL_CONFIG", "/opt/models/rtmo/rtmo-m.py")
DEFAULT_CHECKPOINT = os.environ.get("RTMO_CHECKPOINT", "/opt/models/rtmo/rtmo-m.pth")

INPUT_MODE_ROS_TOPIC = "ros_topic"
INPUT_MODE_REALSENSE = "realsense"

# BoT-SORT scales its track buffer by frame_rate. The tracker sees the
# PROCESSED rate (~55 Hz), not the 60 Hz camera rate, because the capture
# buffer is latest-frame-wins. Used when tracking_frame_rate <= 0.
DEFAULT_TRACKING_FRAME_RATE = 55

# TEMPORARY TRACKING DEBUG: default sink for the BoT-SORT new-track
# diagnostics. output/ is already gitignored, so nothing lands in git.
# Delete this constant together with tracking_debug.py.
DEFAULT_TRACKING_DEBUG_PATH = (
    "/ros2_ws/src/skeleton_detection/output/tracking_debug.log"
)


class RTMONode(Node):
    def __init__(self) -> None:
        super().__init__("rtmo_node")

        self._declare_parameters()
        self._read_parameters()

        self.bridge = CvBridge()
        self.frame_index = 0
        self.stats = PipelineStats()
        self.capture: Optional[RealSenseCapture] = None
        self.tracker: Optional[SkeletonTracker] = None
        self.visualization_writer: Optional[VisualizationWriter] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fatal_error: Optional[str] = None
        self._shutdown_done = False
        self._last_visualization_time = 0.0

        self.get_logger().info(
            f"rtmo_node starting in input_mode='{self.input_mode}' (PID {os.getpid()})"
        )

        self._build_components()
        self._create_publishers()
        self._start_input()
        self._log_configuration()
        self._create_timers()

    # ------------------------------------------------------------------
    # parameters
    # ------------------------------------------------------------------
    def _declare_parameters(self) -> None:
        # --- input selection ---
        self.declare_parameter("input_mode", INPUT_MODE_ROS_TOPIC)
        self.declare_parameter("input_topic", "/dummy_camera/image_raw")
        # --- RealSense (input_mode='realsense') ---
        self.declare_parameter("realsense_width", 848)
        self.declare_parameter("realsense_height", 480)
        self.declare_parameter("realsense_fps", 60)
        self.declare_parameter("realsense_color_format", "bgr8")
        self.declare_parameter("realsense_serial", "")
        # Depth stream, aligned to colour inside RealSenseCapture. Required for
        # PersonSkeleton.depth; with it off every person's depth is NaN.
        self.declare_parameter("realsense_enable_depth", True)
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame")
        # --- outputs ---
        self.declare_parameter("output_topic", "/skeleton_detection/frame")
        # --- model ---
        self.declare_parameter("model_config", DEFAULT_MODEL_CONFIG)
        self.declare_parameter("checkpoint", DEFAULT_CHECKPOINT)
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("person_score_threshold", 0.3)
        # --- per-person depth (see skeleton_detection/person_depth.py) ---
        # A keypoint votes on the person's depth only at/above this RTMO
        # per-joint confidence.
        self.declare_parameter("depth_keypoint_score_threshold", 0.3)
        # Optional metric sanity bounds; <= 0 disables the bound. Zero, NaN and
        # infinite depth samples are rejected regardless.
        self.declare_parameter("depth_min_m", 0.0)
        self.declare_parameter("depth_max_m", 0.0)
        # --- tracking (BoT-SORT + ReID, in this same process) ---
        self.declare_parameter("enable_tracking", False)
        self.declare_parameter("with_reid", True)
        self.declare_parameter("reid_checkpoint", DEFAULT_REID_CHECKPOINT)
        # <= 0 -> DEFAULT_TRACKING_FRAME_RATE (the measured pipeline rate, not
        # the camera rate); scales BoT-SORT's lost-track buffer.
        self.declare_parameter("tracking_frame_rate", 0.0)
        # 'none' (default), 'ecc', 'orb', 'sift' or 'sof'. BoxMOT wants Python
        # None to disable CMC and rejects the string, so 'none' is mapped.
        self.declare_parameter("cmc_method", "none")
        self.declare_parameter("track_high_thresh", 0.5)
        self.declare_parameter("new_track_thresh", 0.6)
        # 90 for the ID-switch investigation (was 30). BoxMOT scales it:
        # max_time_lost = int(frame_rate / 30.0 * track_buffer), so at the
        # default tracking_frame_rate of 55 this is 165 frames (~3.0 s).
        self.declare_parameter("track_buffer", 90)
        self.declare_parameter("match_thresh", 0.8)
        self.declare_parameter("appearance_thresh", 0.25)
        # 0.7, not BoxMOT's 0.5. The proximity gate masks ReID when
        # iou_dist > proximity_thresh, and iou_dist = 1 - IoU, so 0.7 keeps
        # appearance usable down to IoU 0.30 instead of IoU 0.50. RAISING this
        # value relaxes the gate; lowering it to 0.3 would tighten it to
        # IoU >= 0.70.
        self.declare_parameter("proximity_thresh", 0.7)
        # OCCLUSION-AWARE TRACKING (delete with occlusion_tracking.py) ------
        # Off by default: with it off a stock BotSort is constructed and the
        # tracker behaves exactly as before.
        self.declare_parameter("occlusion_aware_tracking", False)
        # A COCO joint counts as visible at or above this per-keypoint score.
        self.declare_parameter("keypoint_visibility_threshold", 0.30)
        # visible_ratio below this => OCCLUDED. This is the ONLY signal:
        # bbox width was removed because a person turning sideways halves
        # their box width while remaining fully visible.
        self.declare_parameter("visible_ratio_threshold", 0.50)
        # The two below size the INFORMATIONAL bbox-width history written to
        # the debug log. They do not affect classification or tracking.
        self.declare_parameter("normal_bbox_history_size", 15)
        self.declare_parameter("min_normal_width_samples", 5)
        # -------------------------------------------------------------------
        # TEMPORARY TRACKING DEBUG: off by default, so normal runs are
        # bit-identical to before. See skeleton_detection/tracking_debug.py.
        self.declare_parameter("tracking_debug_enabled", False)
        self.declare_parameter("tracking_debug_path", DEFAULT_TRACKING_DEBUG_PATH)
        # --- visualisation: human-facing only, off by default ---
        self.declare_parameter("publish_visualization_image", False)
        self.declare_parameter(
            "visualization_topic", "/skeleton_detection/visualization_image"
        )
        self.declare_parameter("visualization_width", 424)
        self.declare_parameter("visualization_height", 240)
        self.declare_parameter("visualization_fps", 10.0)
        self.declare_parameter("visualization_reliability", "best_effort")
        self.declare_parameter("joint_score_threshold", 0.3)
        self.declare_parameter("draw_joint_scores", False)
        self.declare_parameter("save_visualization_images", False)
        self.declare_parameter("visualization_output_dir", DEFAULT_VISUALIZATION_DIR)
        self.declare_parameter("visualization_image_format", "jpg")
        # --- logging / benchmarking ---
        self.declare_parameter("stats_log_period_sec", 1.0)
        self.declare_parameter("log_every_frame", False)
        self.declare_parameter("run_duration_sec", 0.0)

    def _read_parameters(self) -> None:
        def value(name):
            return self.get_parameter(name).value

        self.input_mode = str(value("input_mode")).lower()
        if self.input_mode not in (INPUT_MODE_ROS_TOPIC, INPUT_MODE_REALSENSE):
            raise RuntimeError(
                f"input_mode must be '{INPUT_MODE_ROS_TOPIC}' or "
                f"'{INPUT_MODE_REALSENSE}', got '{self.input_mode}'"
            )

        self.input_topic = str(value("input_topic"))
        self.output_topic = str(value("output_topic"))
        self.camera_frame_id = str(value("camera_frame_id"))
        self.model_config = str(value("model_config"))
        self.checkpoint = str(value("checkpoint"))
        self.device = str(value("device"))
        self.person_score_threshold = float(value("person_score_threshold"))

        self.realsense_enable_depth = bool(value("realsense_enable_depth"))
        self.depth_params = DepthParams(
            # Filled in from the device once the camera is open; 1.0 would mean
            # "the image is already in meters".
            depth_scale=1.0,
            keypoint_score_threshold=float(
                value("depth_keypoint_score_threshold")
            ),
            min_depth_m=float(value("depth_min_m")),
            max_depth_m=float(value("depth_max_m")),
        )

        self.enable_tracking = bool(value("enable_tracking"))
        self.with_reid = bool(value("with_reid"))
        self.reid_checkpoint = str(value("reid_checkpoint"))
        tracking_frame_rate = float(value("tracking_frame_rate"))
        self.tracking_frame_rate = (
            int(round(tracking_frame_rate))
            if tracking_frame_rate > 0
            else DEFAULT_TRACKING_FRAME_RATE
        )
        # OCCLUSION-AWARE TRACKING
        self.occlusion_aware_tracking = bool(value("occlusion_aware_tracking"))
        self.occlusion_params = OcclusionParams(
            keypoint_visibility_threshold=float(
                value("keypoint_visibility_threshold")
            ),
            visible_ratio_threshold=float(value("visible_ratio_threshold")),
            normal_bbox_history_size=int(value("normal_bbox_history_size")),
            min_normal_width_samples=int(value("min_normal_width_samples")),
        )
        if self.occlusion_params.normal_bbox_history_size < 1:
            raise RuntimeError(
                "normal_bbox_history_size must be >= 1, got "
                f"{self.occlusion_params.normal_bbox_history_size}"
            )
        if self.occlusion_params.min_normal_width_samples < 1:
            raise RuntimeError(
                "min_normal_width_samples must be >= 1, got "
                f"{self.occlusion_params.min_normal_width_samples}"
            )

        # TEMPORARY TRACKING DEBUG
        self.tracking_debug_enabled = bool(value("tracking_debug_enabled"))
        self.tracking_debug_path = str(value("tracking_debug_path"))

        cmc = str(value("cmc_method")).lower().strip()
        # BoxMOT's get_cmc_method() accepts None but raises on the string
        # "none"; map the friendly parameter value to the API's contract.
        self.cmc_method = None if cmc in ("", "none") else cmc

        self.publish_visualization_image = bool(value("publish_visualization_image"))
        self.visualization_topic = str(value("visualization_topic"))
        self.visualization_width = int(value("visualization_width"))
        self.visualization_height = int(value("visualization_height"))
        if self.visualization_width < 1 or self.visualization_height < 1:
            raise RuntimeError(
                "visualization_width/height must be >= 1, got "
                f"{self.visualization_width}x{self.visualization_height}"
            )
        self.visualization_fps = float(value("visualization_fps"))
        self._visualization_period = (
            1.0 / self.visualization_fps if self.visualization_fps > 0 else 0.0
        )
        self.visualization_reliability = str(value("visualization_reliability")).lower()
        if self.visualization_reliability not in ("best_effort", "reliable"):
            raise RuntimeError(
                "visualization_reliability must be 'best_effort' or 'reliable', "
                f"got '{self.visualization_reliability}'"
            )
        self.joint_score_threshold = float(value("joint_score_threshold"))
        self.draw_joint_scores = bool(value("draw_joint_scores"))
        self.save_visualization_images = bool(value("save_visualization_images"))
        self.visualization_output_dir = str(value("visualization_output_dir"))
        self.visualization_image_format = str(value("visualization_image_format"))

        self.stats_log_period_sec = float(value("stats_log_period_sec"))
        self.log_every_frame = bool(value("log_every_frame"))
        self.run_duration_sec = float(value("run_duration_sec"))

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_components(self) -> None:
        try:
            self.inference = RTMOInference(
                model_config=self.model_config,
                checkpoint=self.checkpoint,
                device=self.device,
                person_score_threshold=self.person_score_threshold,
                logger=self.get_logger(),
            )
        except RTMOInferenceError as exc:
            raise RuntimeError(str(exc)) from exc

        if self.enable_tracking:
            try:
                self.tracker = SkeletonTracker(
                    reid_checkpoint=self.reid_checkpoint,
                    device=self.device,
                    with_reid=self.with_reid,
                    frame_rate=self.tracking_frame_rate,
                    cmc_method=self.cmc_method,
                    track_high_thresh=float(
                        self.get_parameter("track_high_thresh").value
                    ),
                    new_track_thresh=float(
                        self.get_parameter("new_track_thresh").value
                    ),
                    track_buffer=int(self.get_parameter("track_buffer").value),
                    match_thresh=float(self.get_parameter("match_thresh").value),
                    appearance_thresh=float(
                        self.get_parameter("appearance_thresh").value
                    ),
                    proximity_thresh=float(
                        self.get_parameter("proximity_thresh").value
                    ),
                    logger=self.get_logger(),
                    # OCCLUSION-AWARE TRACKING
                    occlusion_aware_tracking=self.occlusion_aware_tracking,
                    occlusion_params=self.occlusion_params,
                    # TEMPORARY TRACKING DEBUG
                    debug_enabled=self.tracking_debug_enabled,
                    debug_path=self.tracking_debug_path,
                )
            except TrackerInitError as exc:
                raise RuntimeError(f"Tracking could not start: {exc}") from exc

        if self.save_visualization_images:
            self.visualization_writer = VisualizationWriter(
                output_dir=self.visualization_output_dir,
                image_format=self.visualization_image_format,
                logger=self.get_logger(),
            )

        self._legend = legend_for(self.enable_tracking, self.with_reid)

    def _create_publishers(self) -> None:
        # Robot-facing output: reliable, keep-last depth 10 (unchanged).
        self.frame_publisher = self.create_publisher(
            SkeletonFrame, self.output_topic, 10
        )

        # Human-facing output: keep-last depth 1 so a slow viewer can never
        # make stale frames pile up.
        self.visualization_publisher = None
        if self.publish_visualization_image:
            reliability = (
                QoSReliabilityPolicy.BEST_EFFORT
                if self.visualization_reliability == "best_effort"
                else QoSReliabilityPolicy.RELIABLE
            )
            self.visualization_publisher = self.create_publisher(
                Image,
                self.visualization_topic,
                QoSProfile(
                    reliability=reliability,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    depth=1,
                    durability=QoSDurabilityPolicy.VOLATILE,
                ),
            )

    def _start_input(self) -> None:
        self.subscription = None
        if self.input_mode == INPUT_MODE_ROS_TOPIC:
            self.subscription = self.create_subscription(
                Image, self.input_topic, self._on_image, 10
            )
            self.get_logger().info(f"Subscribed to {self.input_topic}")
        else:
            self._start_realsense()

    def _start_realsense(self) -> None:
        """Open the camera in THIS process and start the inference worker."""
        self.capture = RealSenseCapture(
            width=int(self.get_parameter("realsense_width").value),
            height=int(self.get_parameter("realsense_height").value),
            fps=int(self.get_parameter("realsense_fps").value),
            color_format=str(self.get_parameter("realsense_color_format").value),
            enable_depth=self.realsense_enable_depth,
            serial=str(self.get_parameter("realsense_serial").value),
            clock_ns=lambda: self.get_clock().now().nanoseconds,
            logger=self.get_logger(),
        )
        try:
            self.capture.start()
        except RealSenseCaptureError as exc:
            raise RuntimeError(f"RealSense capture could not start: {exc}") from exc

        if self.realsense_enable_depth:
            # The capture thread hands over RAW Z16; this is the only place the
            # device's meters-per-unit scale enters the depth pipeline.
            self.depth_params = replace(
                self.depth_params, depth_scale=self.capture.depth_scale
            )
            self.get_logger().info(
                "Depth ENABLED: z16 aligned to colour in RealSenseCapture, "
                f"depth_scale={self.capture.depth_scale:.6f} m/unit; "
                "PersonSkeleton.depth = median of per-joint 3x3 medians over "
                "keypoints with score >= "
                f"{self.depth_params.keypoint_score_threshold:.2f}"
                + (
                    f", clamped to [{self.depth_params.min_depth_m or 0:.2f}, "
                    f"{self.depth_params.max_depth_m:.2f}] m"
                    if self.depth_params.max_depth_m > 0
                    or self.depth_params.min_depth_m > 0
                    else ""
                )
            )
        else:
            self.get_logger().warning(
                "Depth DISABLED (realsense_enable_depth=false): "
                "PersonSkeleton.depth is NaN for every person"
            )

        intrinsics = self.capture.color_intrinsics
        if intrinsics is not None:
            self.get_logger().info(
                f"Colour intrinsics: fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
                f"ppx={intrinsics.ppx:.2f} ppy={intrinsics.ppy:.2f} "
                f"(recorded for future 3D deprojection; the scalar depth field "
                f"does not need them)"
            )

        self._worker = threading.Thread(
            target=self._inference_loop, name="rtmo_inference", daemon=True
        )
        self._worker.start()
        self.get_logger().info(
            "RealSense capture + RTMO"
            + (" + BoT-SORT" if self.tracker is not None else "")
            + f" running in this process (PID {os.getpid()}); "
            "no realsense2_camera node, no image topic on the input path"
        )

    def _log_configuration(self) -> None:
        self.get_logger().info(f"Publishing SkeletonFrame on {self.output_topic}")
        if self.tracker is not None:
            self.get_logger().info(
                f"Tracking ENABLED: BoT-SORT with_reid={self.with_reid}, "
                f"frame_rate={self.tracking_frame_rate}, cmc={self.cmc_method}, "
                f"track_buffer={self.tracker.track_buffer} -> "
                f"max_time_lost={self.tracker.max_time_lost} frames; "
                + person_id_semantics(True)
            )
            if self.occlusion_aware_tracking:
                # OCCLUSION-AWARE TRACKING: one line; the numbers already went
                # out from SkeletonTracker.
                self.get_logger().warning(
                    "EXPERIMENTAL occlusion-aware tracking is ON: OCCLUDED "
                    "observations no longer update the Kalman motion state or "
                    "the ReID appearance state. Turn it off with "
                    "occlusion_aware_tracking:=false."
                )
            if self.tracking_debug_enabled:
                # TEMPORARY TRACKING DEBUG: one line only. The detail goes to
                # the debug file, never to the ROS log.
                self.get_logger().warning(
                    "TEMPORARY new-track debug instrumentation is ON; "
                    f"writing to {self.tracking_debug_path}"
                    + (
                        " (occlusion NORMAL/OCCLUDED transitions go to the "
                        "same file)"
                        if self.occlusion_aware_tracking
                        else ""
                    )
                )
        else:
            self.get_logger().info(
                "Tracking disabled (enable_tracking=false); "
                + person_id_semantics(False)
            )
        if self.visualization_publisher is not None:
            rate = (
                f"{self.visualization_fps:.1f} Hz"
                if self.visualization_fps > 0
                else "every processed frame (no rate limit)"
            )
            self.get_logger().info(
                f"Publishing visualization image on {self.visualization_topic} "
                f"at {self.visualization_width}x{self.visualization_height}, "
                f"{rate}, bgr8, QoS={self.visualization_reliability}/keep_last/depth=1"
            )
        else:
            self.get_logger().info(
                "Visualization image publishing disabled "
                "(publish_visualization_image=false)"
            )
        if self.visualization_writer is not None:
            self.get_logger().info(
                f"Saving annotated {self.visualization_image_format.upper()} files "
                f"to {self.visualization_output_dir} (every processed frame)"
            )
            if self.input_mode == INPUT_MODE_REALSENSE:
                self.get_logger().warning(
                    "save_visualization_images encodes and writes a file for "
                    "EVERY processed frame, which costs throughput. The live "
                    "visualization topic is rate limited; file saving is not."
                )

    def _create_timers(self) -> None:
        if self.stats_log_period_sec > 0:
            self.create_timer(self.stats_log_period_sec, self._log_stats)
        if self.run_duration_sec > 0:
            self.get_logger().info(
                f"run_duration_sec={self.run_duration_sec:.1f}: the node will stop "
                "automatically and print a summary"
            )
            self.create_timer(self.run_duration_sec, self._deadline_reached)

    # ------------------------------------------------------------------
    # input handling
    # ------------------------------------------------------------------
    def _inference_loop(self) -> None:
        """Consume the newest captured frame and run the shared pipeline."""
        while not self._stop_event.is_set():
            captured = self.capture.get_latest(timeout=0.5)
            if captured is None:
                continue

            header = Header()
            header.stamp = rclpy.time.Time(
                nanoseconds=captured.capture_time_ns
            ).to_msg()
            header.frame_id = self.camera_frame_id

            try:
                self._process_frame(
                    captured.image_bgr, header, captured.depth_image
                )
            except NotImplementedError as exc:
                self._fatal_error = (
                    f"RTMO hit a stubbed mmcv native op: {exc}. A full mmcv "
                    "build is required; refusing to publish results."
                )
                self.get_logger().fatal(self._fatal_error)
                return
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self.get_logger().error(f"Inference loop error: {exc}")

    def _on_image(self, msg: Image) -> None:
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"CvBridge conversion failed: {exc}")
            return

        try:
            # ros_topic mode carries colour only, so depth stays unavailable
            # (NaN) for every person on this path.
            self._process_frame(frame_bgr, msg.header, None)
        except NotImplementedError as exc:
            self.get_logger().fatal(
                f"RTMO hit a stubbed mmcv native op: {exc}. "
                "A full mmcv build is required; refusing to publish results."
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"Frame processing failed: {exc}")

    def _deadline_reached(self) -> None:
        self.get_logger().info("run_duration_sec elapsed; shutting down")
        raise SystemExit

    # ------------------------------------------------------------------
    # the pipeline
    # ------------------------------------------------------------------
    def _process_frame(
        self,
        frame_bgr: np.ndarray,
        header: Header,
        depth_image: Optional[np.ndarray] = None,
    ) -> None:
        """RTMO -> tracking -> depth -> messages -> publish -> visualization.

        ``depth_image`` is the RAW Z16 frame ALREADY ALIGNED TO COLOUR by
        :class:`~skeleton_detection.realsense_capture.RealSenseCapture`, or
        ``None`` when no depth stream is running.
        """
        total_start = time.perf_counter()

        start = time.perf_counter()
        detections: List[PersonDetection] = self.inference.infer(frame_bgr)
        inference_ms = (time.perf_counter() - start) * 1000.0

        # Tracking runs on the SAME in-memory frame and the pre-conversion xyxy
        # boxes, before any message is built, so person_id is already the
        # persistent track id when the frame is published.
        tracking_ms = 0.0
        active_tracks = 0
        if self.tracker is not None:
            start = time.perf_counter()
            detections = self.tracker.update(
                detections,
                frame_bgr,
                # TEMPORARY TRACKING DEBUG: quoted in the debug file only.
                frame_index=self.frame_index,
                timestamp=time.time(),
            )
            tracking_ms = (time.perf_counter() - start) * 1000.0
            active_tracks = self.tracker.active_tracks

        # Depth is attached AFTER tracking and BEFORE the message is built, so
        # person_id and depth in the published message describe the same
        # detection. It never modifies boxes, ids or keypoints.
        self._attach_person_depth(detections, depth_image)

        frame_msg = build_skeleton_frame(header, self.frame_index, detections)
        self.frame_publisher.publish(frame_msg)

        self._handle_visualization(frame_bgr, frame_msg.persons, header)

        total_ms = (time.perf_counter() - total_start) * 1000.0
        self.stats.record_frame(
            inference_ms=inference_ms,
            tracking_ms=tracking_ms,
            total_ms=total_ms,
            person_count=len(frame_msg.persons),
            active_tracks=active_tracks,
        )

        if self.log_every_frame:
            height, width = frame_bgr.shape[:2]
            visible = joint_visibility_summary(
                frame_msg.persons, self.joint_score_threshold
            )
            ids = [person.person_id for person in frame_msg.persons]
            self.get_logger().info(
                f"frame {self.frame_index}: {width}x{height}, rtmo "
                f"{inference_ms:.1f} ms"
                + (f", tracking {tracking_ms:.1f} ms" if tracking_ms else "")
                + f", {len(frame_msg.persons)} person(s) ids={ids}"
                + (f", drawn joints/person: {visible}" if visible else "")
            )
        self.frame_index += 1

    def _attach_person_depth(
        self, detections: List[PersonDetection], depth_image: Optional[np.ndarray]
    ) -> None:
        """Fill ``detection.depth`` [m] for every detection in this frame.

        Thin wrapper only: the estimation itself lives in
        :mod:`skeleton_detection.person_depth`. With no depth image every
        detection keeps the NaN default, which the message and the overlay both
        render as "unavailable".
        """
        if depth_image is None:
            for detection in detections:
                detection.depth = NO_DEPTH
            return

        for detection in detections:
            detection.depth = compute_person_depth(
                depth_image,
                detection.keypoints_xy,
                detection.keypoint_scores,
                self.depth_params,
            )

    def _handle_visualization(
        self, frame_bgr: np.ndarray, persons, header: Header
    ) -> None:
        """Render at most once per frame, and only when a sink needs it.

        The SkeletonFrame is already published by this point, so nothing here
        can throttle the robot-facing topic. At ~55 Hz inference and 10 Hz
        visualization roughly 4 of every 5 frames skip drawing entirely.
        """
        now = time.perf_counter()
        due = self.visualization_publisher is not None and (
            now - self._last_visualization_time >= self._visualization_period
        )
        if not due and self.visualization_writer is None:
            return

        height, width = frame_bgr.shape[:2]
        title = (
            f"frame {self.frame_index:06d} | {width}x{height} | "
            f"persons: {len(persons)}"
        )
        # Drawn at the ORIGINAL resolution so RTMO coordinates are used as-is;
        # any downscale happens afterwards on the finished overlay.
        overlay = draw_skeleton_overlay(
            frame_bgr,
            persons,
            joint_score_threshold=self.joint_score_threshold,
            draw_joint_scores=self.draw_joint_scores,
            title=title,
            legend=self._legend,
        )

        if due:
            self._publish_visualization_image(overlay, header)
            # Advance the schedule by exactly one period rather than resetting
            # to "now": visualization can only be emitted on a frame boundary,
            # so resetting would always round the interval up and run
            # systematically slow.
            self._last_visualization_time += self._visualization_period
            if now - self._last_visualization_time > self._visualization_period:
                self._last_visualization_time = now

        if self.visualization_writer is not None:
            self.visualization_writer.write(overlay, self.frame_index)

    def _publish_visualization_image(
        self, overlay_bgr: np.ndarray, header: Header
    ) -> None:
        """Downscale the finished overlay and publish it as sensor_msgs/Image."""
        height, width = overlay_bgr.shape[:2]
        if (width, height) != (self.visualization_width, self.visualization_height):
            shrinking = (
                self.visualization_width <= width
                and self.visualization_height <= height
            )
            overlay_bgr = cv2.resize(
                overlay_bgr,
                (self.visualization_width, self.visualization_height),
                interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
            )

        image_msg = self.bridge.cv2_to_imgmsg(overlay_bgr, encoding="bgr8")
        image_msg.header = header
        self.visualization_publisher.publish(image_msg)
        self.stats.record_visualization()

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def _log_stats(self) -> None:
        if self._fatal_error is not None:
            raise SystemExit
        window = self.stats.take_window()
        if window["count"] == 0:
            return

        parts = [f"skeleton_fps={window['skeleton_fps']:.1f}"]
        if self.visualization_publisher is not None:
            parts.append(f"vis_fps={window['visualization_fps']:.1f}")
        parts.append(f"rtmo_ms={window['inference_ms']:.1f}")
        if self.tracker is not None:
            parts.append(f"track_ms={window['tracking_ms']:.1f}")
            parts.append(f"tracks={self.tracker.active_tracks}")
        if self.capture is not None:
            capture_stats = self.capture.stats
            parts.insert(0, f"capture_fps={capture_stats.capture_fps():.1f}")
            parts.append(f"captured={capture_stats.captured}")
            parts.append(f"processed={self.stats.processed}")
            parts.append(f"dropped={capture_stats.dropped}")
        else:
            parts.append(f"processed={self.stats.processed}")
        self.get_logger().info(" ".join(parts))

    def _emit(self, message: str) -> None:
        """Log via ROS when possible, else print (Ctrl+C tears the context down
        before destroy_node runs)."""
        if rclpy.ok():
            self.get_logger().info(message)
        else:
            print(message, flush=True)

    def log_final_summary(self) -> None:
        summary = self.stats.summary()
        if summary["processed"] == 0:
            self._emit("No frames were processed")
            return

        def series(name: str, data) -> List[str]:
            if not data:
                return []
            return [
                f"  {name:<17} : mean {data['mean']:.2f} ms | median "
                f"{data['median']:.2f} | p95 {data['p95']:.2f} | "
                f"min/max {data['min']:.2f}/{data['max']:.2f}"
            ]

        lines = [
            "=== pipeline summary ===",
            f"  input_mode        : {self.input_mode}",
            f"  tracking          : "
            + (
                f"ENABLED (BoT-SORT, ReID={'on' if self.with_reid else 'off'}, "
                f"frame_rate={self.tracking_frame_rate}, cmc={self.cmc_method})"
                if self.tracker is not None
                else "disabled"
            ),
            f"  processing window : {summary['elapsed_sec']:.1f} s "
            "(first to last processed frame)",
            f"  processed frames  : {summary['processed']}",
            f"  skeleton FPS      : {summary['skeleton_fps']:.2f}",
            f"  first inference   : {summary['first_inference_ms']:.1f} ms "
            "(CUDA warm-up)",
            f"  persons/frame     : {summary['mean_persons']:.2f} mean",
        ]
        lines += series("rtmo_inference_ms", summary["inference"])
        lines += series("tracking_ms", summary["tracking"])
        lines += series("total_ms", summary["total"])
        if self.tracker is not None:
            lines.append(
                f"  mean active tracks: {summary['mean_active_tracks']:.2f}"
            )
        if self.visualization_publisher is not None:
            lines += [
                f"  visualization     : {self.visualization_width}x"
                f"{self.visualization_height} @ requested "
                f"{self.visualization_fps:.1f} Hz",
                f"  vis images sent   : {self.stats.visualizations}",
                f"  vis FPS measured  : {self.stats.visualization_fps():.2f}",
            ]
        if self.capture is not None:
            capture_stats = self.capture.stats
            lines += [
                f"  captured frames   : {capture_stats.captured}",
                f"  dropped (stale)   : {capture_stats.dropped}",
                f"  capture FPS       : {capture_stats.capture_fps():.2f}",
                f"  capture errors    : {capture_stats.errors}",
            ]
        self._emit("\n".join(lines))

    # ------------------------------------------------------------------
    # shutdown
    # ------------------------------------------------------------------
    def shutdown_pipeline(self) -> None:
        """Stop worker thread and camera. Idempotent; safe from any thread."""
        if self._shutdown_done:
            return
        self._shutdown_done = True

        self._stop_event.set()

        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=5.0)
            if worker.is_alive():
                self.get_logger().warning("Inference worker did not exit in 5s")
        self._worker = None

        if self.capture is not None:
            self.capture.stop()

        self.log_final_summary()

    def destroy_node(self):
        self.shutdown_pipeline()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RTMONode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
