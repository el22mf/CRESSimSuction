import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped
from std_msgs.msg import Float32
import cv2
import numpy as np
from cv_bridge import CvBridge

class BloodDetectionNode(Node):
    def __init__(self):
        super().__init__('blood_detection')

        # Subscribe to raw camera image
        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )
        
        # Publisher for the blood mask image
        self.publisher = self.create_publisher(
            Image,
            '/camera/blood_mask',
            10
        )
        
        # Publisher for the detected blood blob center
        self.center_publisher = self.create_publisher(
            PointStamped,
            '/camera/blood_blob_center',
            10
        )

        self.volume_publisher = self.create_publisher(
            Float32, 
            '/blood_volume', 
            10
        )

        self.bridge = CvBridge()

    def image_callback(self, msg):
        self.get_logger().info('Received an image!')

        # Convert ROS Image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        cv_image = cv2.flip(cv_image, 1)  # Horizontal flip

        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Define lower and upper HSV threshold for detecting blood
        lower_red = np.array([0, 80, 30])
        upper_red = np.array([10, 255, 150])
        
        # Create the binary mask for detecting blood
        mask = cv2.inRange(hsv, lower_red, upper_red)

        blood_pixels = cv2.countNonZero(mask)

        image_width = mask.shape[1]
        image_height = mask.shape[0]
        blood_thickness_m = 0.002

        ortho_half_height = 2.2  # Unity's orthographicSize
        ortho_height = ortho_half_height * 2
        ortho_width = ortho_height * (image_width / image_height)
        total_area = ortho_width * ortho_height

        area_per_pixel = total_area / (image_width * image_height)
        blood_area = blood_pixels * area_per_pixel
        blood_volume = blood_area * blood_thickness_m 

        volume_msg = Float32()
        volume_msg.data = blood_volume
        self.volume_publisher.publish(volume_msg)

        self.get_logger().info(f"Estimated blood volume: {blood_volume * 1e6:.2f} mL")


        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour (assuming it represents the blood)
            largest_contour = max(contours, key=cv2.contourArea)

            # Get the center of the blood region (centroid of the largest contour)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                # Invert Y-coordinate to make the origin bottom-left
                height = cv_image.shape[0]
                cY_inverted = height - cY

                # Publish the center of the blood blob
                blood_center_msg = PointStamped()
                blood_center_msg.header.stamp = self.get_clock().now().to_msg()
                blood_center_msg.header.frame_id = 'camera_frame'  # Use an appropriate frame_id
                blood_center_msg.point.x = float(cX)
                blood_center_msg.point.y = float(cY_inverted)
                blood_center_msg.point.z = 0.0  # Assuming we're in 2D (image plane)
                self.center_publisher.publish(blood_center_msg)

                self.get_logger().info(f"Blood blob center: ({cX}, {cY_inverted})")
            else:
                self.get_logger().warn("No valid contours found!")

        
        # Publish the raw blood mask
        blood_mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        self.publisher.publish(blood_mask_msg)

        self.get_logger().info("Published blood detection mask, center, and volume.")

def main(args=None):
    rclpy.init(args=args)
    node = BloodDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
