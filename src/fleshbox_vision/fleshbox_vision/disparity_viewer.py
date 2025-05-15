import rclpy
from rclpy.node import Node
from stereo_msgs.msg import DisparityImage
import numpy as np
import cv2
from cv_bridge import CvBridge

class DisparityViewer(Node):
    def __init__(self):
        super().__init__('disparity_viewer')
        self.bridge = CvBridge()
        self.create_subscription(DisparityImage, '/stereo/disparity', self.callback, 10)

    def callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding='32FC1')
            img = np.nan_to_num(img, nan=0.0)
            disp_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
            disp_uint8 = disp_normalized.astype(np.uint8)
            cv2.imshow("Disparity", disp_uint8)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = DisparityViewer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
