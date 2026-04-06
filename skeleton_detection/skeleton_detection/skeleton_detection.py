import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image as PILImage
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image

from .iou_tracking import IOUTracker, extract_bbox
from .postprocessing import FrameArtifactsWriter
from .preprocessing import preprocess_image


COCO_CONNECTIONS = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6],
    [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6],
]


def configure_openpifpaf_path() -> None:
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    candidate_roots = []

    env_root = os.environ.get("OPENPIFPAF_VENV_ROOT")
    if env_root:
        candidate_roots.append(Path(env_root))

    module_path = Path(__file__).resolve()
    candidate_roots.extend(parent / "pifpaf_env" for parent in module_path.parents)

    seen = set()
    for root in candidate_roots:
        if root in seen:
            continue
        seen.add(root)

        site_packages = root / "lib" / python_version / "site-packages"
        if site_packages.is_dir():
            site_packages_str = str(site_packages)
            if site_packages_str not in sys.path:
                # Append to preserve stdlib precedence and avoid shadowing by venv backports.
                sys.path.append(site_packages_str)
            return


def draw_tracking_overlays(frame_rgb: np.ndarray, metadata: Sequence[Dict]) -> np.ndarray:
    for person in metadata:
        bbox = person.get("bbox")
        if bbox is None:
            continue

        x, y, width, height = [int(round(value)) for value in bbox]
        if width <= 0 or height <= 0:
            continue

        track_id = person["person_id"]
        color = (0, 255, 0)

        cv2.rectangle(frame_rgb, (x, y), (x + width, y + height), color, 2)

        text_origin_y = y - 10 if y > 20 else y + 20
        cv2.putText(
            frame_rgb,
            f"ID {track_id}",
            (x, text_origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return frame_rgb


class SkeletonDetectionNode(Node):
    def __init__(self) -> None:
        super().__init__("skeleton_detection_node")

        self.declare_parameter("input_topic", "/dummy_camera/image_raw")
        self.declare_parameter("checkpoint", "shufflenetv2k16")
        self.declare_parameter("iou_threshold", 0.3)
        self.declare_parameter("max_missed_frames", 50)
        self.declare_parameter("target_width", 800)
        self.declare_parameter("target_height", 600)
        self.declare_parameter("output_video_path", "/data/skeleton_detection_prediction.mp4")
        self.declare_parameter("output_metadata_path", "/data/skeleton_detection_metadata.json")
        self.declare_parameter("output_fps", 30.0)
        self.declare_parameter("write_video", True)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.target_size: Tuple[int, int] = (
            int(self.get_parameter("target_width").value),
            int(self.get_parameter("target_height").value),
        )
        self.bridge = CvBridge()
        self.frame_index = 0
        self._closed = False

        checkpoint = str(self.get_parameter("checkpoint").value)
        iou_threshold = float(self.get_parameter("iou_threshold").value)
        max_missed_frames = int(self.get_parameter("max_missed_frames").value)
        output_video_path = str(self.get_parameter("output_video_path").value)
        output_metadata_path = str(self.get_parameter("output_metadata_path").value)
        output_fps = float(self.get_parameter("output_fps").value)
        write_video = bool(self.get_parameter("write_video").value)

        configure_openpifpaf_path()

        try:
            import openpifpaf
        except ImportError as exc:
            raise RuntimeError(
                "openpifpaf is required to run skeleton_detection_node"
            ) from exc

        self.predictor = openpifpaf.Predictor(checkpoint=checkpoint)
        self.annotation_painter = openpifpaf.show.AnnotationPainter()
        self.image_canvas = openpifpaf.show.image_canvas
        self.tracker = IOUTracker(
            iou_threshold=iou_threshold,
            max_missed_frames=max_missed_frames,
        )
        self.artifact_writer = FrameArtifactsWriter(
            output_video_path=output_video_path,
            output_metadata_path=output_metadata_path,
            frame_size=self.target_size,
            fps=output_fps,
            write_video=write_video,
        )

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self._on_image,
            10,
        )

        self.get_logger().info(
            f"Listening on {self.input_topic}, target_size={self.target_size}, "
            f"metadata={output_metadata_path}, video={output_video_path}"
        )

    def _process_frame(self, frame_rgb: np.ndarray):
        start_total = time.time()

        t0 = time.time()
        pil_image = preprocess_image(PILImage.fromarray(frame_rgb), target_size=self.target_size)
        input_np = np.asarray(pil_image)
        t_pre = time.time() - t0

        t1 = time.time()
        predictions, _, _ = self.predictor.pil_image(pil_image)
        t_inference = time.time() - t1

        t_nn = t_inference * 0.7
        t_dec = t_inference * 0.3

        bboxes = [extract_bbox(annotation) for annotation in predictions]
        track_ids = self.tracker.update(bboxes)

        metadata: List[Dict] = []
        for index, annotation in enumerate(predictions):
            bbox = bboxes[index]
            metadata.append(
                {
                    "person_id": track_ids[index],
                    "score": round(float(annotation.score), 4),
                    "bbox": None if bbox is None else [round(float(value), 2) for value in bbox],
                    "joints": annotation.data.tolist(),
                    "connections": COCO_CONNECTIONS,
                }
            )

        t2 = time.time()
        with self.image_canvas(input_np) as axis:
            self.annotation_painter.annotations(axis, predictions)
            axis.figure.canvas.draw()
            width, height = axis.figure.canvas.get_width_height()
            annotated_frame = np.frombuffer(axis.figure.canvas.buffer_rgba(), dtype=np.uint8)
            annotated_frame = annotated_frame.reshape((height, width, 4))[:, :, :3].copy()

        if annotated_frame.shape[1] != self.target_size[0] or annotated_frame.shape[0] != self.target_size[1]:
            annotated_frame = cv2.resize(annotated_frame, self.target_size, interpolation=cv2.INTER_LINEAR)

        annotated_frame = draw_tracking_overlays(annotated_frame, metadata)
        t_post = time.time() - t2

        total_time = time.time() - start_total
        timings = {
            "preprocessing": t_pre,
            "neural_network": t_nn,
            "decoder": t_dec,
            "postprocessing": t_post,
            "total": total_time,
        }

        return annotated_frame, metadata, timings

    def _on_image(self, msg: Image) -> None:
        try:
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            annotated_frame_rgb, metadata, timings = self._process_frame(frame_rgb)
            annotated_frame_bgr = cv2.cvtColor(annotated_frame_rgb, cv2.COLOR_RGB2BGR)

            stamp = msg.header.stamp.sec + (msg.header.stamp.nanosec / 1e9)
            self.artifact_writer.append(
                frame_index=self.frame_index,
                timestamp_sec=stamp,
                annotated_frame_bgr=annotated_frame_bgr,
                persons=metadata,
                timings=timings,
            )

            self.get_logger().info(
                f"Processed frame {self.frame_index} with {len(metadata)} detections in {timings['total']:.3f} seconds"
            )
            self.frame_index += 1
        except Exception as exc:
            self.get_logger().error(f"Failed to process image frame: {exc}")

    def destroy_node(self):
        if not self._closed:
            try:
                self.artifact_writer.close()
            finally:
                self._closed = True
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SkeletonDetectionNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
