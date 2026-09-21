"""Publish one or more local test images as sensor_msgs/msg/Image.

Milestone 1 stand-in for a real camera: it lets the RTMO node be exercised
end-to-end without RealSense streaming.  Each image is published on
``/dummy_camera/image_raw`` with a valid header (node clock stamp + optical
frame id), which is exactly what the RealSense driver will publish later, so
``iot_node`` does not need to change when this node is replaced.

Image selection (first match wins):
  1. ``image_paths``  - list of files, published in the given order
  2. ``image_dir``    - every supported image in the directory, sorted by name
  3. ``image_path``   - a single file (the original one-image behaviour)

The node waits until a subscriber is matched before publishing the first
image, so nothing is lost to DDS discovery timing, then publishes one image
every ``publish_interval_sec`` seconds and shuts itself down.  Exactly one
message is published per image, so each image causes exactly one RTMO
inference.
"""

import os
from typing import List

import cv2
import rclpy
from cv_bridge import CvBridge
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import Image


DEFAULT_IMAGE_PATH = "/ros2_ws/src/skeleton_detection/data/images/000000000785.jpg"
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")


class ImagePublisherNode(Node):
    def __init__(self) -> None:
        super().__init__("image_publisher_node")

        # An empty list default would be inferred as BYTE_ARRAY by rclpy and
        # would then reject a string list from YAML/CLI, so this parameter is
        # declared with dynamic typing.
        self.declare_parameter(
            "image_paths", [], ParameterDescriptor(dynamic_typing=True)
        )
        self.declare_parameter("image_dir", "")
        self.declare_parameter("image_path", DEFAULT_IMAGE_PATH)
        self.declare_parameter("output_topic", "/dummy_camera/image_raw")
        self.declare_parameter("frame_id", "camera_color_optical_frame")
        # Seconds between two published images. This is an offline inspection
        # aid, not an FPS benchmark: 1 s is plenty.
        self.declare_parameter("publish_interval_sec", 1.0)
        # Give the subscriber time to appear; publish anyway once it expires.
        self.declare_parameter("wait_for_subscriber_sec", 10.0)
        # Exit after the last publish so `ros2 run` returns on its own.
        self.declare_parameter("shutdown_after_publish", True)

        self.output_topic = str(self.get_parameter("output_topic").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.publish_interval_sec = float(
            self.get_parameter("publish_interval_sec").value
        )
        self.wait_for_subscriber_sec = float(
            self.get_parameter("wait_for_subscriber_sec").value
        )
        self.shutdown_after_publish = bool(
            self.get_parameter("shutdown_after_publish").value
        )

        self.image_paths = self._resolve_image_paths()
        self.frames = self._load_images(self.image_paths)

        self.bridge = CvBridge()
        self.next_index = 0
        self._waited_sec = 0.0

        self.publisher = self.create_publisher(Image, self.output_topic, 10)
        self.timer = self.create_timer(self.publish_interval_sec, self._on_timer)

        listing = "\n".join(
            f"  [{index}] {path} ({frame.shape[1]}x{frame.shape[0]})"
            for index, (path, frame) in enumerate(zip(self.image_paths, self.frames))
        )
        self.get_logger().info(
            f"Publishing {len(self.frames)} image(s) on {self.output_topic} "
            f"every {self.publish_interval_sec:.1f}s with "
            f"frame_id='{self.frame_id}':\n{listing}"
        )

    # ------------------------------------------------------------------
    # input resolution
    # ------------------------------------------------------------------
    def _resolve_image_paths(self) -> List[str]:
        raw_paths = self.get_parameter("image_paths").value or []
        if raw_paths:
            paths = [str(path) for path in raw_paths]
            self.get_logger().info(f"Using image_paths ({len(paths)} entries)")
            return paths

        image_dir = str(self.get_parameter("image_dir").value)
        if image_dir:
            if not os.path.isdir(image_dir):
                raise RuntimeError(f"image_dir is not a directory: {image_dir}")
            paths = sorted(
                os.path.join(image_dir, name)
                for name in os.listdir(image_dir)
                if name.lower().endswith(SUPPORTED_SUFFIXES)
            )
            if not paths:
                raise RuntimeError(
                    f"No {'/'.join(SUPPORTED_SUFFIXES)} files found in {image_dir}"
                )
            self.get_logger().info(
                f"Using image_dir '{image_dir}' ({len(paths)} images, sorted by name)"
            )
            return paths

        return [str(self.get_parameter("image_path").value)]

    def _load_images(self, paths: List[str]):
        """Load every image up front so a bad path fails before publishing."""
        missing = [path for path in paths if not os.path.isfile(path)]
        if missing:
            raise RuntimeError("Test image(s) not found: " + ", ".join(missing))

        frames = []
        for path in paths:
            frame_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if frame_bgr is None:
                raise RuntimeError(f"OpenCV failed to decode image: {path}")
            frames.append(frame_bgr)
        return frames

    # ------------------------------------------------------------------
    # publishing
    # ------------------------------------------------------------------
    def _on_timer(self) -> None:
        subscribers = self.publisher.get_subscription_count()
        if (
            self.next_index == 0
            and subscribers == 0
            and self._waited_sec < self.wait_for_subscriber_sec
        ):
            self._waited_sec += self.publish_interval_sec
            self.get_logger().info(
                f"Waiting for a subscriber on {self.output_topic} "
                f"({self._waited_sec:.1f}/{self.wait_for_subscriber_sec:.1f}s)"
            )
            return

        if self.next_index == 0 and subscribers == 0:
            self.get_logger().warning(
                f"No subscriber on {self.output_topic}; publishing anyway"
            )

        path = self.image_paths[self.next_index]
        frame_bgr = self.frames[self.next_index]

        image_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        image_msg.header.stamp = self.get_clock().now().to_msg()
        image_msg.header.frame_id = self.frame_id
        self.publisher.publish(image_msg)
        self.next_index += 1

        self.get_logger().info(
            f"Published image {self.next_index}/{len(self.frames)} "
            f"({os.path.basename(path)}) on {self.output_topic} "
            f"(stamp={image_msg.header.stamp.sec}."
            f"{image_msg.header.stamp.nanosec:09d}, "
            f"{image_msg.width}x{image_msg.height}, encoding={image_msg.encoding})"
        )

        if self.next_index >= len(self.frames):
            self.timer.cancel()
            self.get_logger().info(
                f"Published all {len(self.frames)} image(s); "
                f"expect {len(self.frames)} SkeletonFrame message(s)"
            )
            if self.shutdown_after_publish:
                # Let the last message leave the middleware before tearing down.
                self.create_timer(1.0, self._shutdown)

    def _shutdown(self) -> None:
        self.get_logger().info("Done publishing test image(s), shutting down")
        raise SystemExit


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImagePublisherNode()
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
