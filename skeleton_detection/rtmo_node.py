"""RTMO-M bottom-up pose estimation ROS 2 node.

Two input modes, selected with the ``input_mode`` parameter:

``ros_topic`` (Milestone 1 regression path)::

    /dummy_camera/image_raw (sensor_msgs/Image, bgr8)
        -> CvBridge -> numpy BGR -> RTMO -> /skeleton_detection/frame

``realsense`` (Milestone 2, same-process live capture)::

    Intel RealSense D456
        -> pyrealsense2 pipeline, IN THIS PROCESS
        -> numpy BGR (no DDS, no sensor_msgs/Image on the input path)
        -> RTMO -> /skeleton_detection/frame

Both modes share one inference/publish implementation (:meth:`_process_frame`),
so the structured output contract is identical.

Two outputs, deliberately decoupled::

    /skeleton_detection/frame               SkeletonFrame, robot-facing,
                                            every processed frame (~50 Hz)
    /skeleton_detection/visualization_image sensor_msgs/Image, human-facing,
                                            rate limited (default 10 Hz) and
                                            downscaled (default 424x240)

The visualization is produced AFTER inference and after the SkeletonFrame has
already been published, so it can never throttle the robot-facing topic, and it
is never part of the input path.
"""

import os
import os.path as osp
import statistics
import threading
import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from skeleton_detection.msg import PersonSkeleton, SkeletonFrame

from .coco_keypoints import (
    COCO_CONNECTIONS_FLAT,
    COCO_KEYPOINT_NAMES,
    NUM_COCO_KEYPOINTS,
)
from .realsense_capture import RealSenseCapture, RealSenseCaptureError
from .visualization import draw_skeleton_overlay, joint_visibility_summary


DEFAULT_VISUALIZATION_DIR = "/ros2_ws/src/skeleton_detection/output/visualizations"

DEFAULT_MODEL_CONFIG = (
    "/usr/local/lib/python3.10/dist-packages/mmpose/.mim/configs/"
    "body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py"
)
DEFAULT_CHECKPOINT = (
    "/ros2_ws/src/skeleton_detection/data/checkpoints/"
    "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth"
)

INPUT_MODE_ROS_TOPIC = "ros_topic"
INPUT_MODE_REALSENSE = "realsense"


def install_checkpoint_loader_workaround() -> None:
    """Allow mmengine to load the RTMO checkpoint under torch >= 2.6.

    torch 2.6 flipped the ``torch.load`` ``weights_only`` default to ``True``.
    mmengine 0.10.7's local checkpoint loader never passes the argument and the
    RTMO checkpoint stores numpy objects the restricted unpickler cannot
    rebuild.  Override the registered scheme through mmengine's public API
    instead of patching site-packages (see docker/PACKAGES.md, finding 4).
    """
    import torch
    from mmengine.runner import CheckpointLoader

    @CheckpointLoader.register_scheme(prefixes="", force=True)
    def _load_from_local_trusted(filename, map_location):
        filename = osp.expanduser(filename)
        if not osp.isfile(filename):
            raise FileNotFoundError(f"{filename} can not be found.")
        return torch.load(filename, map_location=map_location, weights_only=False)


class PipelineStats:
    """Throughput/latency counters for the live pipeline.

    ``dropped`` is owned by the capture buffer: a frame is dropped when a newer
    frame replaces it before the inference loop consumed it (latest-frame-wins).
    """

    def __init__(self) -> None:
        self.processed = 0
        self.inference_times_ms: List[float] = []
        self._window_processed = 0
        self._window_start = time.perf_counter()
        self._window_inference_ms: List[float] = []
        self.first_inference_ms: Optional[float] = None
        # Measured from the FIRST processed frame, so model loading and camera
        # start-up do not depress the reported throughput.
        self.first_process_time: Optional[float] = None
        self.last_process_time: Optional[float] = None
        # Visualization images actually rendered+published (rate limited).
        self.visualizations = 0
        self._window_visualizations = 0
        self.first_visualization_time: Optional[float] = None
        self.last_visualization_time: Optional[float] = None

    def record_visualization(self) -> None:
        now = time.perf_counter()
        if self.first_visualization_time is None:
            self.first_visualization_time = now
        self.last_visualization_time = now
        self.visualizations += 1
        self._window_visualizations += 1

    def visualization_fps(self) -> float:
        if (
            self.first_visualization_time is None
            or self.last_visualization_time is None
            or self.visualizations < 2
        ):
            return 0.0
        span = self.last_visualization_time - self.first_visualization_time
        return (self.visualizations - 1) / span if span > 0 else 0.0

    def record(self, inference_ms: float) -> None:
        now = time.perf_counter()
        if self.first_process_time is None:
            self.first_process_time = now
        self.last_process_time = now
        self.processed += 1
        self.inference_times_ms.append(inference_ms)
        self._window_inference_ms.append(inference_ms)
        self._window_processed += 1
        if self.first_inference_ms is None:
            self.first_inference_ms = inference_ms

    def take_window(self):
        """Return (fps, mean_ms, count) since the previous call and reset it."""
        now = time.perf_counter()
        span = now - self._window_start
        count = self._window_processed
        mean_ms = (
            sum(self._window_inference_ms) / len(self._window_inference_ms)
            if self._window_inference_ms
            else 0.0
        )
        fps = count / span if span > 0 else 0.0
        visualization_fps = (
            self._window_visualizations / span if span > 0 else 0.0
        )
        self._window_start = now
        self._window_processed = 0
        self._window_visualizations = 0
        self._window_inference_ms = []
        return fps, mean_ms, count, visualization_fps

    def summary(self, warmup_frames: int = 1) -> dict:
        """Aggregate stats; the first ``warmup_frames`` are excluded from the
        steady-state numbers because the first inference pays CUDA warm-up."""
        steady = self.inference_times_ms[warmup_frames:]
        if (
            self.first_process_time is not None
            and self.last_process_time is not None
            and self.processed > 1
        ):
            elapsed = self.last_process_time - self.first_process_time
            skeleton_fps = (self.processed - 1) / elapsed if elapsed > 0 else 0.0
        else:
            elapsed = 0.0
            skeleton_fps = 0.0
        result = {
            "elapsed_sec": elapsed,
            "processed": self.processed,
            "skeleton_fps": skeleton_fps,
            "first_inference_ms": self.first_inference_ms or 0.0,
        }
        if steady:
            ordered = sorted(steady)
            result.update(
                {
                    "steady_frames": len(steady),
                    "inference_mean_ms": statistics.fmean(steady),
                    "inference_median_ms": statistics.median(steady),
                    "inference_p95_ms": ordered[max(0, int(len(ordered) * 0.95) - 1)],
                    "inference_min_ms": ordered[0],
                    "inference_max_ms": ordered[-1],
                }
            )
        return result


class RTMONode(Node):
    def __init__(self) -> None:
        super().__init__("rtmo_node")

        # --- input selection ---
        self.declare_parameter("input_mode", INPUT_MODE_ROS_TOPIC)
        self.declare_parameter("input_topic", "/dummy_camera/image_raw")
        # --- RealSense (input_mode='realsense') ---
        self.declare_parameter("realsense_width", 848)
        self.declare_parameter("realsense_height", 480)
        self.declare_parameter("realsense_fps", 60)
        self.declare_parameter("realsense_color_format", "bgr8")
        self.declare_parameter("realsense_serial", "")
        self.declare_parameter("camera_frame_id", "camera_color_optical_frame")
        # --- outputs ---
        self.declare_parameter("output_topic", "/skeleton_detection/frame")
        # --- model ---
        self.declare_parameter("model_config", DEFAULT_MODEL_CONFIG)
        self.declare_parameter("checkpoint", DEFAULT_CHECKPOINT)
        self.declare_parameter("device", "cuda:0")
        # RTMO's own test_cfg keeps everything above 0.1; this is an extra
        # node-level filter so obvious false positives are not published.
        self.declare_parameter("person_score_threshold", 0.3)
        # --- visualisation: human-facing only, off by default ---
        # Live annotated image on a ROS topic, for rqt_image_view. Rate and
        # resolution are limited independently of the SkeletonFrame output, so
        # enabling this never throttles the robot-facing topic.
        self.declare_parameter("publish_visualization_image", False)
        self.declare_parameter(
            "visualization_topic", "/skeleton_detection/visualization_image"
        )
        self.declare_parameter("visualization_width", 424)
        self.declare_parameter("visualization_height", 240)
        # Target publish rate for the visualization topic. <= 0 means "every
        # processed frame" (no rate limit).
        self.declare_parameter("visualization_fps", 10.0)
        # 'best_effort' (live stream, stale frames dropped) or 'reliable'
        # (needed by viewers that subscribe with reliable QoS).
        self.declare_parameter("visualization_reliability", "best_effort")
        # Drawing options shared by the live topic and the saved files.
        self.declare_parameter("joint_score_threshold", 0.3)
        self.declare_parameter("draw_joint_scores", False)
        # Offline artefact: write the SAME overlay to disk as image files.
        self.declare_parameter("save_visualization_images", False)
        self.declare_parameter("visualization_output_dir", DEFAULT_VISUALIZATION_DIR)
        self.declare_parameter("visualization_image_format", "jpg")
        # --- logging / benchmarking ---
        self.declare_parameter("stats_log_period_sec", 1.0)
        # Log one line per processed frame. Sensible for the single-image
        # regression path, far too noisy at 55 FPS.
        self.declare_parameter("log_every_frame", False)
        # >0: shut the node down after N seconds (scripted benchmark runs).
        self.declare_parameter("run_duration_sec", 0.0)

        self.input_mode = str(self.get_parameter("input_mode").value).lower()
        if self.input_mode not in (INPUT_MODE_ROS_TOPIC, INPUT_MODE_REALSENSE):
            raise RuntimeError(
                f"input_mode must be '{INPUT_MODE_ROS_TOPIC}' or "
                f"'{INPUT_MODE_REALSENSE}', got '{self.input_mode}'"
            )

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.camera_frame_id = str(self.get_parameter("camera_frame_id").value)
        model_config = str(self.get_parameter("model_config").value)
        checkpoint = str(self.get_parameter("checkpoint").value)
        self.device = str(self.get_parameter("device").value)
        self.person_score_threshold = float(
            self.get_parameter("person_score_threshold").value
        )
        self.publish_visualization_image = bool(
            self.get_parameter("publish_visualization_image").value
        )
        self.visualization_topic = str(self.get_parameter("visualization_topic").value)
        self.visualization_width = int(self.get_parameter("visualization_width").value)
        self.visualization_height = int(
            self.get_parameter("visualization_height").value
        )
        if self.visualization_width < 1 or self.visualization_height < 1:
            raise RuntimeError(
                "visualization_width/height must be >= 1, got "
                f"{self.visualization_width}x{self.visualization_height}"
            )
        self.visualization_fps = float(self.get_parameter("visualization_fps").value)
        # Rate limiting uses a monotonic clock; <= 0 disables the limit.
        self._visualization_period = (
            1.0 / self.visualization_fps if self.visualization_fps > 0 else 0.0
        )
        self._last_visualization_time = 0.0
        self.visualization_reliability = str(
            self.get_parameter("visualization_reliability").value
        ).lower()
        if self.visualization_reliability not in ("best_effort", "reliable"):
            raise RuntimeError(
                "visualization_reliability must be 'best_effort' or 'reliable', "
                f"got '{self.visualization_reliability}'"
            )
        self.joint_score_threshold = float(
            self.get_parameter("joint_score_threshold").value
        )
        self.draw_joint_scores = bool(self.get_parameter("draw_joint_scores").value)
        self.save_visualization_images = bool(
            self.get_parameter("save_visualization_images").value
        )
        self.visualization_output_dir = str(
            self.get_parameter("visualization_output_dir").value
        )
        self.visualization_image_format = (
            str(self.get_parameter("visualization_image_format").value)
            .lstrip(".")
            .lower()
        )
        if self.visualization_image_format not in ("jpg", "jpeg", "png"):
            raise RuntimeError(
                "visualization_image_format must be one of 'jpg', 'jpeg', 'png'; "
                f"got '{self.visualization_image_format}'"
            )
        self.stats_log_period_sec = float(
            self.get_parameter("stats_log_period_sec").value
        )
        self.log_every_frame = bool(self.get_parameter("log_every_frame").value)
        self.run_duration_sec = float(self.get_parameter("run_duration_sec").value)

        self.bridge = CvBridge()
        self.frame_index = 0
        self.stats = PipelineStats()
        self.capture: Optional[RealSenseCapture] = None
        self._worker: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._fatal_error: Optional[str] = None
        self._shutdown_done = False

        self.get_logger().info(
            f"rtmo_node starting in input_mode='{self.input_mode}' (PID {os.getpid()})"
        )

        self._load_model(model_config, checkpoint)
        self._prepare_visualization_output_dir()

        # Robot-facing output. QoS unchanged from the previous milestones:
        # reliable, keep-last depth 10.
        self.frame_publisher = self.create_publisher(SkeletonFrame, self.output_topic, 10)

        # Human-facing output. Keep-last depth 1 so a slow viewer can never
        # make stale frames pile up -- for live inspection the newest frame is
        # the only one worth having.
        self.visualization_publisher = None
        if self.publish_visualization_image:
            reliability = (
                QoSReliabilityPolicy.BEST_EFFORT
                if self.visualization_reliability == "best_effort"
                else QoSReliabilityPolicy.RELIABLE
            )
            visualization_qos = QoSProfile(
                reliability=reliability,
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                durability=QoSDurabilityPolicy.VOLATILE,
            )
            self.visualization_publisher = self.create_publisher(
                Image, self.visualization_topic, visualization_qos
            )

        self.subscription = None
        if self.input_mode == INPUT_MODE_ROS_TOPIC:
            self.subscription = self.create_subscription(
                Image, self.input_topic, self._on_image, 10
            )
            self.get_logger().info(f"Subscribed to {self.input_topic}")
        else:
            self._start_realsense()

        self.get_logger().info(f"Publishing SkeletonFrame on {self.output_topic}")
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
        if self.save_visualization_images:
            self.get_logger().info(
                f"Saving annotated {self.visualization_image_format.upper()} files "
                f"to {self.visualization_output_dir} (every processed frame)"
            )

        if self.input_mode == INPUT_MODE_REALSENSE and self.save_visualization_images:
            self.get_logger().warning(
                "save_visualization_images is enabled in realsense mode: it "
                "encodes and writes a file for EVERY processed frame, which "
                "costs throughput. The live visualization topic is rate limited; "
                "file saving is not."
            )

        if self.stats_log_period_sec > 0:
            self.create_timer(self.stats_log_period_sec, self._log_stats)
        if self.run_duration_sec > 0:
            self.get_logger().info(
                f"run_duration_sec={self.run_duration_sec:.1f}: the node will stop "
                "automatically and print a summary"
            )
            self.create_timer(self.run_duration_sec, self._deadline_reached)

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def _load_model(self, model_config: str, checkpoint: str) -> None:
        """Build RTMO-M once, at node start-up (never per frame)."""
        import torch

        if not osp.isfile(model_config):
            raise RuntimeError(f"RTMO config not found: {model_config}")
        if not osp.isfile(checkpoint):
            raise RuntimeError(f"RTMO checkpoint not found: {checkpoint}")

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available inside this container, but device="
                    f"'{self.device}' was requested. Start the container with "
                    "GPU access or set the 'device' parameter to 'cpu' explicitly."
                )
            gpu_index = torch.cuda.current_device()
            self.get_logger().info(
                f"CUDA available: using {torch.cuda.get_device_name(gpu_index)} "
                f"(device='{self.device}', torch {torch.__version__})"
            )
        else:
            self.get_logger().warning(f"Running RTMO on '{self.device}' (no CUDA)")

        install_checkpoint_loader_workaround()

        # Imported here (not at module import time) so the checkpoint-loader
        # override is registered first and import failures are reported by the
        # node logger with full context.
        from mmpose.apis import inference_bottomup, init_model

        self._inference_bottomup = inference_bottomup

        start = time.time()
        self.model = init_model(model_config, checkpoint, device=self.device)
        self.get_logger().info(
            f"Loaded RTMO-M in {time.time() - start:.2f}s "
            f"(config={model_config}, checkpoint={checkpoint})"
        )

        self._verify_keypoint_layout()

    def _verify_keypoint_layout(self) -> None:
        """Fail loudly if the checkpoint is not COCO-17 in the expected order."""
        dataset_meta = getattr(self.model, "dataset_meta", None) or {}
        id2name = dataset_meta.get("keypoint_id2name")
        if not id2name:
            self.get_logger().warning(
                "Model dataset_meta has no keypoint_id2name; assuming COCO-17 order"
            )
            return

        names = [id2name[index] for index in sorted(id2name)]
        if names != COCO_KEYPOINT_NAMES:
            raise RuntimeError(
                "Model keypoint layout does not match the COCO-17 order this "
                f"package publishes.\n  model: {names}\n  expected: "
                f"{COCO_KEYPOINT_NAMES}"
            )
        self.get_logger().info(
            f"Keypoint layout verified: COCO-17 ({NUM_COCO_KEYPOINTS} joints)"
        )

    # ------------------------------------------------------------------
    # realsense mode
    # ------------------------------------------------------------------
    def _start_realsense(self) -> None:
        """Open the camera in THIS process and start the inference worker."""
        self.capture = RealSenseCapture(
            width=int(self.get_parameter("realsense_width").value),
            height=int(self.get_parameter("realsense_height").value),
            fps=int(self.get_parameter("realsense_fps").value),
            color_format=str(self.get_parameter("realsense_color_format").value),
            serial=str(self.get_parameter("realsense_serial").value),
            # Capture timestamps come from the ROS clock, the same clock used
            # for the published header (see _process_frame).
            clock_ns=lambda: self.get_clock().now().nanoseconds,
            logger=self.get_logger(),
        )
        try:
            self.capture.start()
        except RealSenseCaptureError as exc:
            raise RuntimeError(f"RealSense capture could not start: {exc}") from exc

        intrinsics = self.capture.color_intrinsics
        if intrinsics is not None:
            self.get_logger().info(
                f"Colour intrinsics: fx={intrinsics.fx:.2f} fy={intrinsics.fy:.2f} "
                f"ppx={intrinsics.ppx:.2f} ppy={intrinsics.ppy:.2f} "
                f"(recorded for future depth/3D work; unused in this milestone)"
            )

        self._worker = threading.Thread(
            target=self._inference_loop, name="rtmo_inference", daemon=True
        )
        self._worker.start()
        self.get_logger().info(
            "RealSense capture + RTMO inference running in this process "
            f"(PID {os.getpid()}); no realsense2_camera node, no image topic "
            "on the input path"
        )

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
                self._process_frame(captured.image_bgr, header)
            except NotImplementedError as exc:
                self._fatal_error = (
                    f"RTMO hit a stubbed mmcv native op: {exc}. A full mmcv "
                    "build is required; refusing to publish results."
                )
                self.get_logger().fatal(self._fatal_error)
                return
            except Exception as exc:  # noqa: BLE001 - keep the loop alive
                self.get_logger().error(f"Inference loop error: {exc}")

    def _deadline_reached(self) -> None:
        self.get_logger().info("run_duration_sec elapsed; shutting down")
        raise SystemExit

    # ------------------------------------------------------------------
    # ros_topic mode
    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - report and keep the node alive
            self.get_logger().error(f"CvBridge conversion failed: {exc}")
            return

        try:
            self._process_frame(frame_bgr, msg.header)
        except NotImplementedError as exc:
            self.get_logger().fatal(
                f"RTMO hit a stubbed mmcv native op: {exc}. "
                "A full mmcv build is required; refusing to publish results."
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"RTMO inference failed: {exc}")

    # ------------------------------------------------------------------
    # shared inference + publish path (both input modes)
    # ------------------------------------------------------------------
    def _process_frame(self, frame_bgr: np.ndarray, header: Header) -> None:
        start = time.perf_counter()
        # Official MMPose bottom-up API: it owns resize/pad/normalisation and
        # maps results back to original image coordinates.
        results = self._inference_bottomup(self.model, frame_bgr)
        inference_ms = (time.perf_counter() - start) * 1000.0

        persons = self._build_person_messages(results)

        frame_msg = SkeletonFrame()
        frame_msg.header = header
        frame_msg.frame_index = self.frame_index
        frame_msg.timestamp = header.stamp.sec + header.stamp.nanosec / 1e9
        frame_msg.persons = persons
        self.frame_publisher.publish(frame_msg)

        height, width = frame_bgr.shape[:2]

        # --- visualisation: human-facing, rate limited, never blocks the above.
        # The SkeletonFrame is already published by this point, so nothing here
        # can throttle the robot-facing topic. Drawing happens ONLY when a
        # visualization frame is actually due: at ~50 Hz inference and 10 Hz
        # visualization, roughly 4 of every 5 frames skip drawing entirely.
        now = time.perf_counter()
        visualization_due = self.visualization_publisher is not None and (
            now - self._last_visualization_time >= self._visualization_period
        )

        overlay = None
        if visualization_due or self.save_visualization_images:
            title = (
                f"frame {self.frame_index:06d} | {width}x{height} | "
                f"persons: {len(persons)}"
            )
            # Drawn at the ORIGINAL resolution so RTMO coordinates are used
            # as-is; the downscale happens afterwards on the finished overlay.
            overlay = draw_skeleton_overlay(
                frame_bgr,
                persons,
                joint_score_threshold=self.joint_score_threshold,
                draw_joint_scores=self.draw_joint_scores,
                title=title,
            )

        if visualization_due and overlay is not None:
            self._publish_visualization_image(overlay, header)
            # Advance the schedule by exactly one period rather than resetting
            # to "now". Visualization can only be emitted on a frame boundary
            # (~18 ms apart), so resetting to now would always round the
            # interval UP to the next frame and run systematically slow
            # (~9.2 Hz when 10 Hz was asked for). Accumulating the target time
            # makes the long-run average match the requested rate, with jitter
            # of at most one frame period.
            self._last_visualization_time += self._visualization_period
            if now - self._last_visualization_time > self._visualization_period:
                # First frame, or the loop fell behind (stall/pause): resync
                # instead of trying to catch up with a burst.
                self._last_visualization_time = now

        saved_path = None
        if self.save_visualization_images and overlay is not None:
            saved_path = self._save_visualization_image(overlay)

        self.stats.record(inference_ms)

        if self.log_every_frame:
            visible = joint_visibility_summary(persons, self.joint_score_threshold)
            self.get_logger().info(
                f"frame {self.frame_index}: {width}x{height}, inference "
                f"{inference_ms:.1f} ms, {len(persons)} person(s) published on "
                f"{self.output_topic}"
                + (f", drawn joints/person: {visible}" if visible else "")
                + (f", saved {saved_path}" if saved_path else "")
            )
        self.frame_index += 1

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def _log_stats(self) -> None:
        if self._fatal_error is not None:
            raise SystemExit
        fps, mean_ms, count, visualization_fps = self.stats.take_window()
        if count == 0:
            return

        visualization = (
            f" vis_fps={visualization_fps:.1f}"
            if self.visualization_publisher is not None
            else ""
        )
        if self.capture is not None:
            capture_stats = self.capture.stats
            self.get_logger().info(
                f"capture_fps={capture_stats.capture_fps():.1f} "
                f"skeleton_fps={fps:.1f}{visualization} "
                f"inference_ms={mean_ms:.1f} "
                f"captured={capture_stats.captured} "
                f"processed={self.stats.processed} "
                f"dropped={capture_stats.dropped}"
            )
        else:
            self.get_logger().info(
                f"skeleton_fps={fps:.1f}{visualization} inference_ms={mean_ms:.1f} "
                f"processed={self.stats.processed}"
            )

    def _emit(self, message: str) -> None:
        """Log via ROS when possible, else print.

        On Ctrl+C rclpy tears the context down before ``destroy_node`` runs, so
        a plain ``get_logger().info`` at shutdown fails with "publisher's
        context is invalid". Falling back to stdout keeps the final summary
        readable without that noise.
        """
        if rclpy.ok():
            self.get_logger().info(message)
        else:
            print(message, flush=True)

    def log_final_summary(self) -> None:
        summary = self.stats.summary()
        if summary["processed"] == 0:
            self._emit("No frames were processed")
            return

        lines = [
            "=== RTMO pipeline summary ===",
            f"  input_mode        : {self.input_mode}",
            f"  processing window : {summary['elapsed_sec']:.1f} s "
            "(first to last processed frame)",
            f"  processed frames  : {summary['processed']}",
            f"  skeleton FPS      : {summary['skeleton_fps']:.2f}",
            f"  first inference   : {summary['first_inference_ms']:.1f} ms (CUDA warm-up)",
        ]
        if "inference_mean_ms" in summary:
            lines += [
                f"  steady-state over : {summary['steady_frames']} frames",
                f"  inference mean    : {summary['inference_mean_ms']:.2f} ms",
                f"  inference median  : {summary['inference_median_ms']:.2f} ms",
                f"  inference p95     : {summary['inference_p95_ms']:.2f} ms",
                f"  inference min/max : {summary['inference_min_ms']:.2f} / "
                f"{summary['inference_max_ms']:.2f} ms",
            ]
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
    # result parsing
    # ------------------------------------------------------------------
    def _build_person_messages(self, results) -> List[PersonSkeleton]:
        """Convert RTMO ``PoseDataSample`` predictions into PersonSkeleton msgs.

        ``pred_instances`` (already in original-image coordinates) provides:
          - ``bboxes``          (N, 4) float, xyxy
          - ``scores``          (N,)   float, person/instance score
          - ``keypoints``       (N, 17, 2) float, (x, y) in image pixels
          - ``keypoint_scores`` (N, 17)    float, per-joint confidence
        """
        persons: List[PersonSkeleton] = []
        if not results:
            return persons

        pred_instances = results[0].pred_instances
        keypoints = np.asarray(pred_instances.keypoints, dtype=np.float32)
        keypoint_scores = np.asarray(pred_instances.keypoint_scores, dtype=np.float32)
        scores = np.asarray(pred_instances.scores, dtype=np.float32)
        bboxes = np.asarray(pred_instances.bboxes, dtype=np.float32)

        for index in range(keypoints.shape[0]):
            score = float(scores[index])
            if score < self.person_score_threshold:
                continue

            person_keypoints = keypoints[index]
            if person_keypoints.shape[0] != NUM_COCO_KEYPOINTS:
                self.get_logger().error(
                    f"Expected {NUM_COCO_KEYPOINTS} keypoints, got "
                    f"{person_keypoints.shape[0]}; skipping detection {index}"
                )
                continue

            # xyxy -> [x, y, width, height] (top-left + size), the convention
            # used everywhere in this package.
            x_min, y_min, x_max, y_max = (float(value) for value in bboxes[index])

            # [x0, y0, conf0, x1, y1, conf1, ...] -> 17 * 3 = 51 floats.
            joints: List[float] = []
            for joint_index in range(NUM_COCO_KEYPOINTS):
                joints.append(float(person_keypoints[joint_index, 0]))
                joints.append(float(person_keypoints[joint_index, 1]))
                joints.append(float(keypoint_scores[index, joint_index]))

            person_msg = PersonSkeleton()
            # Frame-local detection index, NOT a tracking id.
            person_msg.person_id = index
            person_msg.score = score
            person_msg.bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            person_msg.joints = joints
            person_msg.connections = list(COCO_CONNECTIONS_FLAT)
            # No depth in this milestone: position is always zero.
            person_msg.position = Point(x=0.0, y=0.0, z=0.0)
            persons.append(person_msg)

        return persons

    # ------------------------------------------------------------------
    # visualisation helpers (live topic + saved files)
    # ------------------------------------------------------------------
    def _publish_visualization_image(
        self, overlay_bgr: np.ndarray, header: Header
    ) -> None:
        """Downscale the finished overlay and publish it as sensor_msgs/Image.

        Resizing the COMPLETED overlay (rather than drawing on a pre-shrunk
        frame) keeps every bbox/joint/text coordinate in the original RTMO
        image space, so nothing has to be rescaled by hand.
        """
        height, width = overlay_bgr.shape[:2]
        if (width, height) != (self.visualization_width, self.visualization_height):
            # INTER_AREA is the right filter for shrinking but a poor one for
            # enlarging, so pick per direction.
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

    def _prepare_visualization_output_dir(self) -> None:
        """Create the annotated-image output directory if saving is enabled.

        The default lives inside the bind-mounted source tree, so files written
        here appear directly on the host and can be opened from the VS Code SSH
        file explorer -- no rqt, X11 or display forwarding involved.
        """
        if not self.save_visualization_images:
            return

        # Remember which directories we are about to create so their
        # permissions can be widened afterwards (see _relax_permissions).
        created: List[str] = []
        candidate = osp.abspath(self.visualization_output_dir)
        while candidate and not osp.isdir(candidate):
            created.append(candidate)
            parent = osp.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent

        try:
            os.makedirs(self.visualization_output_dir, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create visualization_output_dir '{self.visualization_output_dir}': {exc}"
            ) from exc

        for directory in created:
            self._relax_permissions(directory, 0o777)

        if not os.access(self.visualization_output_dir, os.W_OK):
            raise RuntimeError(
                f"visualization_output_dir '{self.visualization_output_dir}' is not writable"
            )

        self.get_logger().info(
            f"Annotated images will be written to {self.visualization_output_dir} "
            f"as frame_XXXXXX_rtmo.{self.visualization_image_format}"
        )

    def _save_visualization_image(self, overlay_bgr: np.ndarray) -> Optional[str]:
        """Write one annotated frame; returns the path, or None on failure.

        Naming convention: ``frame_<frame_index padded to 6>_rtmo.<ext>``.
        ``frame_index`` increases for every frame the node processes and resets
        when the node restarts, so files never collide within a run, but a NEW
        run overwrites the files of the previous one -- that overwrite is
        logged as a warning rather than done silently.
        """
        filename = f"frame_{self.frame_index:06d}_rtmo.{self.visualization_image_format}"
        path = os.path.join(self.visualization_output_dir, filename)

        if os.path.exists(path):
            self.get_logger().warning(
                f"Overwriting existing {path} (left over from an earlier run)"
            )

        try:
            written = cv2.imwrite(path, overlay_bgr)
        except cv2.error as exc:
            self.get_logger().error(f"Failed to write {path}: {exc}")
            return None

        if not written:
            self.get_logger().error(f"cv2.imwrite returned False for {path}")
            return None

        self._relax_permissions(path, 0o666)
        return path

    @staticmethod
    def _relax_permissions(path: str, mode: int) -> None:
        """Make container-written files manageable from the host.

        This container runs as root, so anything it writes into the bind mount
        is root-owned on the host.  Widening the mode keeps the offline
        workflow friction-free.  Best effort: silently skipped when not running
        as root or on filesystems that refuse chmod.
        """
        if os.geteuid() != 0:
            return
        try:
            os.chmod(path, mode)
        except OSError:
            pass

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
