#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2

class CompressedImageSubscriber(Node):
    def __init__(self):
        super().__init__('compressed_image_subscriber')
        # Subscribe to the compressed image topic
        self.subscription = self.create_subscription(
            CompressedImage,
            '/zed/zed_node/right/image_rect_color/compressed',  # Change to your topic name
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning

        # CvBridge for converting ROS images to OpenCV images
        self.bridge = CvBridge()

        # Create an OpenCV window
        cv2.namedWindow('Compressed Image', cv2.WINDOW_AUTOSIZE)
        self.get_logger().info('CompressedImageSubscriber initialized and subscribed')

    def listener_callback(self, msg: CompressedImage):
        try:
            # Convert CompressedImage ROS message to OpenCV image
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Display the image
            cv2.imshow('Compressed Image', cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error converting or displaying image: {e}')


def main(args=None):
    # Initialize ROS 2 Python client library
    rclpy.init(args=args)

    # Create the node
    node = CompressedImageSubscriber()

    try:
        # Keep the node alive to receive messages
        rclpy.spin(node)

    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down...')

    finally:
        # Cleanup
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
