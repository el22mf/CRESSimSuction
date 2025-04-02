import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
import numpy as np
from cv_bridge import CvBridge

class BloodDetectionNode(Node):
    def __init__(self):
        super().__init__('blood_detection')

        # Subscribe to the raw camera image
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for the processed image
        self.publisher = self.create_publisher(
            Image,
            '/camera/blood_mask',
            10
        )

        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.get_logger().info('Received an image!')

        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define lower and upper HSV threshold for detecting blood
        lower_red = np.array([0, 80, 30])
        upper_red = np.array([10, 255, 150])
        
        mask = cv2.inRange(hsv, lower_red, upper_red)  # Create the binary mask
        
        # Convert mask to a ROS Image message and publish it
        blood_mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        self.publisher.publish(blood_mask_msg)

        self.get_logger().info("Published blood detection mask.")

def main(args=None):
    rclpy.init(args=args)
    node = BloodDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
