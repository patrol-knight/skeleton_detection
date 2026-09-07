"""RTMO-M bottom-up pose estimation ROS 2 node.

Data flow::

    /dummy_camera/image_raw (sensor_msgs/Image, bgr8)
        -> CvBridge -> numpy BGR (original resolution)
        -> mmpose.apis.inference_bottomup (official RTMO test pipeline)
        -> /skeleton_detection/frame (skeleton_detection/SkeletonFrame)   [always]
        -> /skeleton_detection/debug_image (sensor_msgs/Image)            [optional]
        -> <debug_output_dir>/frame_XXXXXX_rtmo.jpg                       [optional]

The structured SkeletonFrame output is the contract; the debug topic and the
saved annotated files are additional artifacts for human inspection.  Both use
the same drawing code (``skeleton_detection.visualization``), and the overlay
is rendered at most once per frame.
"""

import os
import os.path as osp
import time
from typing import List, Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from sensor_msgs.msg import Image

from skeleton_detection.msg import PersonSkeleton, SkeletonFrame

from .coco_keypoints import (
    COCO_CONNECTIONS_FLAT,
    COCO_KEYPOINT_NAMES,
    NUM_COCO_KEYPOINTS,
)
from .visualization import draw_skeleton_overlay, joint_visibility_summary


DEFAULT_DEBUG_OUTPUT_DIR = "/ros2_ws/src/skeleton_detection/output/visualizations"


DEFAULT_MODEL_CONFIG = (
    "/usr/local/lib/python3.10/dist-packages/mmpose/.mim/configs/"
    "body_2d_keypoint/rtmo/body7/rtmo-m_16xb16-600e_body7-640x640.py"
)
DEFAULT_CHECKPOINT = (
    "/ros2_ws/src/skeleton_detection/data/checkpoints/"
    "rtmo-m_16xb16-600e_body7-640x640-39e78cc4_20231211.pth"
)


def install_checkpoint_loader_workaround() -> None:
    """Allow mmengine to load the RTMO checkpoint under torch >= 2.6.

    torch 2.6 flipped the ``torch.load`` ``weights_only`` default to ``True``.
    mmengine 0.10.7's local checkpoint loader never passes the argument and the
    RTMO checkpoint stores numpy objects the restricted unpickler cannot
    rebuild.  Override the registered scheme through mmengine's public API
    instead of patching site-packages (same approach as the earlier RTMO smoke
    test; see docker/PACKAGES.md, finding 4).
    """
    import torch
    from mmengine.runner import CheckpointLoader

    @CheckpointLoader.register_scheme(prefixes="", force=True)
    def _load_from_local_trusted(filename, map_location):
        filename = osp.expanduser(filename)
        if not osp.isfile(filename):
            raise FileNotFoundError(f"{filename} can not be found.")
        return torch.load(filename, map_location=map_location, weights_only=False)


class RTMONode(Node):
    def __init__(self) -> None:
        super().__init__("rtmo_node")

        self.declare_parameter("input_topic", "/dummy_camera/image_raw")
        self.declare_parameter("output_topic", "/skeleton_detection/frame")
        self.declare_parameter("debug_image_topic", "/skeleton_detection/debug_image")
        self.declare_parameter("model_config", DEFAULT_MODEL_CONFIG)
        self.declare_parameter("checkpoint", DEFAULT_CHECKPOINT)
        self.declare_parameter("device", "cuda:0")
        # RTMO's own test_cfg keeps everything above 0.1; this is an extra
        # node-level filter so obvious false positives are not published.
        self.declare_parameter("person_score_threshold", 0.3)
        self.declare_parameter("publish_debug_image", True)
        # --- offline visualisation (Milestone 1.5) ---
        # Joints below this confidence are not drawn, and a skeleton line is
        # drawn only when BOTH of its endpoints are above it. Drawing only;
        # the published SkeletonFrame always carries all 17 joints.
        self.declare_parameter("joint_score_threshold", 0.3)
        self.declare_parameter("draw_joint_scores", False)
        self.declare_parameter("save_debug_images", False)
        self.declare_parameter("debug_output_dir", DEFAULT_DEBUG_OUTPUT_DIR)
        self.declare_parameter("debug_image_format", "jpg")

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.output_topic = str(self.get_parameter("output_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        model_config = str(self.get_parameter("model_config").value)
        checkpoint = str(self.get_parameter("checkpoint").value)
        self.device = str(self.get_parameter("device").value)
        self.person_score_threshold = float(
            self.get_parameter("person_score_threshold").value
        )
        self.publish_debug_image = bool(self.get_parameter("publish_debug_image").value)
        self.joint_score_threshold = float(
            self.get_parameter("joint_score_threshold").value
        )
        self.draw_joint_scores = bool(self.get_parameter("draw_joint_scores").value)
        self.save_debug_images = bool(self.get_parameter("save_debug_images").value)
        self.debug_output_dir = str(self.get_parameter("debug_output_dir").value)
        self.debug_image_format = (
            str(self.get_parameter("debug_image_format").value).lstrip(".").lower()
        )
        if self.debug_image_format not in ("jpg", "jpeg", "png"):
            raise RuntimeError(
                "debug_image_format must be one of 'jpg', 'jpeg', 'png'; got "
                f"'{self.debug_image_format}'"
            )

        self.bridge = CvBridge()
        self.frame_index = 0

        self._load_model(model_config, checkpoint)

        self._prepare_debug_output_dir()

        self.frame_publisher = self.create_publisher(SkeletonFrame, self.output_topic, 10)
        self.debug_image_publisher = (
            self.create_publisher(Image, self.debug_image_topic, 10)
            if self.publish_debug_image
            else None
        )
        self.subscription = self.create_subscription(
            Image, self.input_topic, self._on_image, 10
        )

        self.get_logger().info(
            f"Subscribed to {self.input_topic}; publishing SkeletonFrame on "
            f"{self.output_topic}"
            + (
                f"; debug image on {self.debug_image_topic}"
                if self.publish_debug_image
                else ""
            )
            + (
                f"; saving annotated {self.debug_image_format.upper()} files to "
                f"{self.debug_output_dir}"
                if self.save_debug_images
                else "; annotated file saving disabled (save_debug_images=false)"
            )
        )

    # ------------------------------------------------------------------
    # model
    # ------------------------------------------------------------------
    def _load_model(self, model_config: str, checkpoint: str) -> None:
        """Build RTMO-M once, at node start-up (never per callback)."""
        import torch

        if not osp.isfile(model_config):
            raise RuntimeError(f"RTMO config not found: {model_config}")
        if not osp.isfile(checkpoint):
            raise RuntimeError(f"RTMO checkpoint not found: {checkpoint}")

        if self.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA is not available inside this container, but device="
                    f"'{self.device}' was requested. Milestone 1 requires GPU "
                    "inference; start the container with GPU access or set the "
                    "'device' parameter to 'cpu' explicitly."
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
    # callback
    # ------------------------------------------------------------------
    def _on_image(self, msg: Image) -> None:
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:  # noqa: BLE001 - report and keep the node alive
            self.get_logger().error(f"CvBridge conversion failed: {exc}")
            return

        try:
            start = time.time()
            # Official MMPose bottom-up API: it owns resize/pad/normalisation
            # and maps results back to original image coordinates.
            results = self._inference_bottomup(self.model, frame_bgr)
            elapsed = time.time() - start
        except NotImplementedError as exc:
            # The container ships mmcv-lite + a stubbed mmcv._ext; a real op
            # call means RTMO needs a full mmcv build. Do not fake a result.
            self.get_logger().fatal(
                f"RTMO hit a stubbed mmcv native op: {exc}. "
                "A full mmcv build is required; refusing to publish results."
            )
            raise
        except Exception as exc:  # noqa: BLE001
            self.get_logger().error(f"RTMO inference failed: {exc}")
            return

        persons = self._build_person_messages(results)

        frame_msg = SkeletonFrame()
        frame_msg.header = msg.header
        frame_msg.frame_index = self.frame_index
        frame_msg.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        frame_msg.persons = persons
        self.frame_publisher.publish(frame_msg)

        height, width = frame_bgr.shape[:2]

        # Render the overlay at most once, then reuse it for the ROS debug
        # topic and/or the saved file.
        overlay = None
        if self.debug_image_publisher is not None or self.save_debug_images:
            title = (
                f"frame {self.frame_index:06d} | {width}x{height} | "
                f"persons: {len(persons)}"
            )
            overlay = draw_skeleton_overlay(
                frame_bgr,
                persons,
                joint_score_threshold=self.joint_score_threshold,
                draw_joint_scores=self.draw_joint_scores,
                title=title,
            )

        if self.debug_image_publisher is not None and overlay is not None:
            debug_msg = self.bridge.cv2_to_imgmsg(overlay, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_image_publisher.publish(debug_msg)

        saved_path = None
        if self.save_debug_images and overlay is not None:
            saved_path = self._save_debug_image(overlay)

        visible = joint_visibility_summary(persons, self.joint_score_threshold)
        self.get_logger().info(
            f"frame {self.frame_index}: {width}x{height}, inference "
            f"{elapsed * 1000.0:.1f} ms, {len(persons)} person(s) published on "
            f"{self.output_topic}"
            + (f", drawn joints/person: {visible}" if visible else "")
            + (f", saved {saved_path}" if saved_path else "")
        )
        self.frame_index += 1

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
            # Milestone 1: frame-local detection index, NOT a tracking id.
            person_msg.person_id = index
            person_msg.score = score
            person_msg.bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            person_msg.joints = joints
            person_msg.connections = list(COCO_CONNECTIONS_FLAT)
            # No depth in Milestone 1: position is always zero.
            person_msg.position = Point(x=0.0, y=0.0, z=0.0)
            persons.append(person_msg)

        return persons

    # ------------------------------------------------------------------
    # optional debug visualisation (topic + saved files)
    # ------------------------------------------------------------------
    def _prepare_debug_output_dir(self) -> None:
        """Create the annotated-image output directory if saving is enabled.

        The default lives inside the bind-mounted source tree
        (``/ros2_ws/src/skeleton_detection/output/visualizations``), so files
        written here appear directly on the host at
        ``<repo>/output/visualizations`` and can be opened from the VS Code SSH
        file explorer -- no rqt, X11 or display forwarding involved.
        """
        if not self.save_debug_images:
            return

        # Remember which directories we are about to create so their
        # permissions can be widened afterwards (see _relax_permissions).
        created: List[str] = []
        candidate = osp.abspath(self.debug_output_dir)
        while candidate and not osp.isdir(candidate):
            created.append(candidate)
            parent = osp.dirname(candidate)
            if parent == candidate:
                break
            candidate = parent

        try:
            os.makedirs(self.debug_output_dir, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Cannot create debug_output_dir '{self.debug_output_dir}': {exc}"
            ) from exc

        for directory in created:
            self._relax_permissions(directory, 0o777)

        if not os.access(self.debug_output_dir, os.W_OK):
            raise RuntimeError(
                f"debug_output_dir '{self.debug_output_dir}' is not writable"
            )

        self.get_logger().info(
            f"Annotated images will be written to {self.debug_output_dir} "
            f"as frame_XXXXXX_rtmo.{self.debug_image_format}"
        )

    def _save_debug_image(self, overlay_bgr: np.ndarray) -> Optional[str]:
        """Write one annotated frame; returns the path, or None on failure.

        Naming convention: ``frame_<frame_index padded to 6>_rtmo.<ext>``,
        e.g. ``frame_000000_rtmo.jpg``.  ``frame_index`` increases for every
        frame the node processes and resets when the node restarts, so files
        never collide within a run, but a NEW run overwrites the files of the
        previous one -- that overwrite is logged as a warning rather than done
        silently.
        """
        filename = f"frame_{self.frame_index:06d}_rtmo.{self.debug_image_format}"
        path = os.path.join(self.debug_output_dir, filename)

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
        is root-owned on the host.  The host user can always READ it (which is
        all that is needed to view the image in VS Code), but could not delete
        or overwrite it.  Widening the mode keeps the offline workflow
        friction-free.  Best effort: silently skipped when not running as root
        or on filesystems that refuse chmod.
        """
        if os.geteuid() != 0:
            return
        try:
            os.chmod(path, mode)
        except OSError:
            pass


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RTMONode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
