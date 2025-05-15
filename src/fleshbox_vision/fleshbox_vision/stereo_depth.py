import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField
from cv_bridge import CvBridge
import numpy as np
import cv2
import struct
import time

class StereoDepthViewer(Node):
    def __init__(self):
        super().__init__('stereo_depth_real_viewer')

        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None

        self.focal_length_px = 222.2222 
        self.cx = 80.0   
        self.cy = 120.0
        self.baseline_m = 0.6      

        self.pointcloud_pub = self.create_publisher(PointCloud2, '/fleshbox/pointcloud', 10)

        self.stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=16 * 6,
            blockSize=7,
            P1=8 * 3 * 7 ** 2,
            P2=32 * 3 * 7 ** 2,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        self.create_subscription(Image, '/left_camera/image_raw', self.left_callback, 10)
        self.create_subscription(Image, '/right_camera/image_raw', self.right_callback, 10)

        cv2.namedWindow("Disparity", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    def left_callback(self, msg):
        self.left_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_compute_depth()

    def right_callback(self, msg):
        self.right_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.try_compute_depth()

    def try_compute_depth(self):
        if self.left_image is None or self.right_image is None:
            return

        if self.left_image is None or self.right_image is None:
            return

        left_gray = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_image, cv2.COLOR_BGR2GRAY)

        if left_gray.shape != right_gray.shape:
            self.get_logger().warn(f"Mismatched image sizes: left={left_gray.shape}, right={right_gray.shape}")
            return

        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        disparity = np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0)

        valid_disp = disparity > 0
        depth_m = np.zeros_like(disparity)
        depth_m[valid_disp] = (self.focal_length_px * self.baseline_m) / disparity[valid_disp]

        depth_m = depth_m[:, 40:]
        self.left_image = self.left_image[:, 40:]

        disp_normalized = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
        disp_color = cv2.applyColorMap(np.uint8(disp_normalized), cv2.COLORMAP_TURBO)

        depth_clip = np.clip(depth_m, np.percentile(depth_m, 2), np.percentile(depth_m, 98))
        depth_normalized = cv2.normalize(depth_clip, None, 0, 255, cv2.NORM_MINMAX)
        depth_color = cv2.applyColorMap(np.uint8(depth_normalized), cv2.COLORMAP_VIRIDIS)
        depth_color = cv2.GaussianBlur(depth_color, (3, 3), 0)

        cv2.imshow("Disparity", disp_color)
        cv2.imshow("Depth", depth_color)
        cv2.waitKey(1)

        self.generate_downsampled_point_cloud(depth_m, self.left_image)

    def generate_downsampled_point_cloud(self, depth_map, rgb_image, target_size=(100, 100), depth_trunc=2.0):
        depth_resized = cv2.resize(depth_map, target_size, interpolation=cv2.INTER_NEAREST)
        rgb_resized = cv2.resize(rgb_image, target_size, interpolation=cv2.INTER_LINEAR)

        h, w = target_size
        i, j = np.meshgrid(np.arange(w), np.arange(h))

        scale = target_size[0] / self.left_image.shape[1]
        fx = self.focal_length_px * scale
        fy = self.focal_length_px * scale
        cx = self.cx * scale
        cy = self.cy * scale

        z = depth_resized
        x = (i - cx) * z / fx
        y = (j - cy) * z / fy

        xyz = np.stack((x, y, z), axis=-1).reshape(-1, 3)
        rgb = rgb_resized.reshape(-1, 3)

        valid = (z.reshape(-1) > 0.2) & (z.reshape(-1) < depth_trunc) & np.all(np.isfinite(xyz), axis=1)

        points = xyz[valid]
        colors = rgb[valid]

        msg = self.create_pointcloud2(points, colors.astype(np.uint8))
        self.pointcloud_pub.publish(msg)

    def create_pointcloud2(self, points, colors):
        header = self.get_clock().now().to_msg()

        fields = [
            PointField(name='x', offset=0,  datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4,  datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8,  datatype=PointField.FLOAT32, count=1),
            PointField(name='r', offset=12, datatype=PointField.UINT8, count=1),
            PointField(name='g', offset=13, datatype=PointField.UINT8, count=1),
            PointField(name='b', offset=14, datatype=PointField.UINT8, count=1),
        ]

        data = []
        for pt, color in zip(points, colors):
            data.append(struct.pack('fffBBB', pt[0], pt[1], pt[2], color[0], color[1], color[2]))

        pc2 = PointCloud2()
        pc2.header.stamp = header
        pc2.header.frame_id = 'camera_link'
        pc2.height = 1
        pc2.width = len(data)
        pc2.fields = fields
        pc2.is_bigendian = False
        pc2.point_step = 15
        pc2.row_step = pc2.point_step * pc2.width
        pc2.data = b''.join(data)
        pc2.is_dense = True
        return pc2


def main(args=None):
    rclpy.init(args=args)
    node = StereoDepthViewer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()
