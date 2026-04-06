from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraReaderNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_reader_node")

        default_video_path = Path(__file__).resolve().parents[2] / "data" / "test_video_1.mp4"

        self.declare_parameter("input_topic", "/dummy_camera/image_raw")
        self.declare_parameter("video_path", str(default_video_path))
        self.declare_parameter("publish_fps", 30.0)

        self.input_topic = str(self.get_parameter("input_topic").value)
        self.video_path = str(self.get_parameter("video_path").value)
        self.publish_fps = float(self.get_parameter("publish_fps").value)

        self.publisher = self.create_publisher(Image, self.input_topic, 10)
        self.bridge = CvBridge()
        self.capture = cv2.VideoCapture(self.video_path)
        self.frame_index = 0

        if not self.capture.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.video_path}")

        timer_period = 1.0 / max(self.publish_fps, 0.1)
        self.timer = self.create_timer(timer_period, self._publish_frame)

        self.get_logger().info(
            f"Publishing {self.video_path} to {self.input_topic} at {self.publish_fps:.2f} FPS"
        )

    def _publish_frame(self) -> None:
        if self.frame_index > 300:
            print("Stopping after 10 frames for testing purposes.")
            self.timer.cancel()
            return

        ok, frame_bgr = self.capture.read()

        if not ok:
            self.get_logger().info("Reached end of video. Stopping camera_reader_node.")
            self.timer.cancel()
            return

        msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        self.publisher.publish(msg)

        self.frame_index += 1

    def destroy_node(self):
        if hasattr(self, "capture") and self.capture is not None:
            self.capture.release()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraReaderNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
