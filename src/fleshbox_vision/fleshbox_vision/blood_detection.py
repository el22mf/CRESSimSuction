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

        # Define the pixel-to-meter conversion
        self.pixel_to_meter = 4.4 / 100  # Each pixel represents 0.044 meters (4.4 cm)

        # Define transformation matrix for spatial adjustment (example)
        self.src_pts = np.float32([[0, 0], [99, 0], [0, 99]])  # Image corners
        self.dst_pts = np.float32([[0, 0], [99, 5], [5, 99]])  # Adjusted based on real-world scaling
        self.transformation_matrix = cv2.getAffineTransform(self.src_pts, self.dst_pts)

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

        # Apply the spatial transformation
        transformed_mask = cv2.warpAffine(mask, self.transformation_matrix, (mask.shape[1], mask.shape[0]))

        # Convert transformed mask to a ROS Image message and publish it
        blood_mask_msg = self.bridge.cv2_to_imgmsg(transformed_mask, encoding='mono8')
        self.publisher.publish(blood_mask_msg)

        self.get_logger().info("Published transformed blood detection mask.")

def main(args=None):
    rclpy.init(args=args)
    node = BloodDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
