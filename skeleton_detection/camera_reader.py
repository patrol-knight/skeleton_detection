import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraSubscriberNode(Node):
    def __init__(self) -> None:
        super().__init__("camera_subscriber_node")

        # 1. Declare the topic parameters
        # Change default to your actual camera topic
        self.declare_parameter("input_topic", "/camera/camera/color/image_raw")
        self.declare_parameter("output_topic", "/dummy_camera/image_raw")

        self.input_topic = self.get_parameter("input_topic").get_parameter_value().string_value
        self.output_topic = self.get_parameter("output_topic").get_parameter_value().string_value

        self.bridge = CvBridge()

        # 2. Create a Subscriber instead of a VideoCapture
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self._image_callback,
            10  # QoS profile depth
        )

        # 3. (Optional) Create a Publisher if you want to relay the frames
        self.publisher = self.create_publisher(Image, self.output_topic, 10)

        self.get_logger().info(f"Subscribed to: {self.input_topic}")
        self.get_logger().info(f"Relaying to: {self.output_topic}")

    def _image_callback(self, msg: Image) -> None:
        try:
            # Convert ROS Image message to OpenCV BGR format
            frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

            # --- INSERT ANY PROCESSING HERE (Optional) ---
            # e.g., frame_bgr = cv2.flip(frame_bgr, 1) 

            # Convert back to ROS message and publish
            output_msg = self.bridge.cv2_to_imgmsg(frame_bgr, encoding="bgr8")
            output_msg.header = msg.header # Preserve original timestamp and frame_id
            
            self.get_logger().info("Received and processed a frame, publishing...")
            self.publisher.publish(output_msg)

        except Exception as e:
            self.get_logger().error(f"Failed to process incoming image: {e}")

def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraSubscriberNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()