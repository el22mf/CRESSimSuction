import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from PyKDL import Rotation
import numpy as np
import cv2
import threading
import time
import matplotlib.pyplot as plt
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

SimToSI = 0.05  # Simulation to SI conversion factor

class RobotData:
    def __init__(self):
        self.measured_cp = PoseStamped()

class SurChalCrtkTestNode(Node):
    def __init__(self):
        super().__init__('sur_chal_crtk_test')
        self.robData = RobotData()
        self.bridge = CvBridge()
        self.latest_mask = None
        self.latest_depth = None
        self.track_blob = False

        arm_name = "PSM1"
        measured_cp_topic = arm_name + "/measured_cp"
        self.servo_cp_topic = arm_name + "/servo_cp"

        self.create_subscription(PoseStamped, measured_cp_topic, self.measured_cp_cb, 10)
        self.create_subscription(Image, '/camera/blood_mask', self.mask_callback, 10)
        self.create_subscription(Image, '/stereo/depth_map', self.depth_callback, 10)

        self.volume_data = []  # holds blood volume over time
        self.create_subscription(Float32, '/blood_volume', self.volume_callback, 10)

        self.latest_pointcloud = None
        self.create_subscription(PointCloud2, '/fleshbox/pointcloud', self.pointcloud_callback, 10)


        self.servo_cp_pub = self.create_publisher(PoseStamped, self.servo_cp_topic, 10)
        self.completion_pub = self.create_publisher(String, 'completion_status', 10)



        self.servo_cp_msg = PoseStamped()
        self.servo_cp_msg.pose.position.x = 0.0
        self.servo_cp_msg.pose.position.y = 0.0
        self.servo_cp_msg.pose.position.z = -0.1 * SimToSI
        R_7_0 = Rotation.RPY(-1.57079, 0.0, 1.57079)
        quat = R_7_0.GetQuaternion()
        self.servo_cp_msg.pose.orientation.x = quat[0]
        self.servo_cp_msg.pose.orientation.y = quat[1]
        self.servo_cp_msg.pose.orientation.z = quat[2]
        self.servo_cp_msg.pose.orientation.w = quat[3]

        self.timer = self.create_timer(0.02, self.timer_callback)
        self.user_input_thread = threading.Thread(target=self.user_input_loop, daemon=True)
        self.user_input_thread.start()

        self.pixel_to_meter = 4.4 / 100
        self.origin_offset_x = 0
        self.origin_offset_y = -1.52

        self.blob_pixel_counts = []

    def measured_cp_cb(self, msg):
        self.robData.measured_cp = msg

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')
        except Exception as e:
            self.get_logger().error(f"Depth image conversion failed: {e}")

    def get_depth_at_pixel(self, x, y):
        if self.latest_pointcloud is None:
            print("No point cloud available.")
            return None

        width = 100
        index = y * width + x

        try:
            for i, point in enumerate(pc2.read_points(self.latest_pointcloud, field_names=("x", "y", "z", "r", "g", "b"), skip_nans=False)):
                if i == index:
                    x_val, y_val, z_val, *_ = point
                    if np.isfinite(z_val) and z_val > 0.001:
                        return z_val  # already in meters
                    else:
                        return None
        except Exception as e:
            print(f"Error reading point cloud: {e}")
            return None




    def mask_callback(self, msg):
        self.latest_mask = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

    def timer_callback(self):
        self.servo_cp_pub.publish(self.servo_cp_msg)

    def volume_callback(self, msg):
        self.volume_data.append(msg.data)

    def pointcloud_callback(self, msg):
        self.latest_pointcloud = msg



    def check_no_blood_left(self):
        if self.latest_mask is None:
            return False
        white_pixel_count = np.count_nonzero(self.latest_mask == 255)
        return white_pixel_count == 0

    def get_blob_center_and_size(self):
        if self.latest_mask is None:
            print("No mask available.")
            return None

        params = cv2.SimpleBlobDetector_Params()
        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 20000
        params.filterByColor = True
        params.blobColor = 255
        params.filterByCircularity = False
        params.filterByConvexity = False
        params.filterByInertia = False
        detector = cv2.SimpleBlobDetector_create(params)

        keypoints = detector.detect(self.latest_mask)
        if not keypoints:
            print("No blobs detected.")
            return None

        largest_blob = max(keypoints, key=lambda k: k.size)
        cx, cy = largest_blob.pt

        x_si = (cx / 99) * 4.4 - 2.2 + self.origin_offset_x
        flipped_cy = 99 - cy
        y_si = (flipped_cy / 99) * 4.4 - 2.2 + self.origin_offset_y

        contours, _ = cv2.findContours(self.latest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("No contours found.")
            return (x_si, y_si), None

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        bounding_box_px = {
            'min_x_px': x,
            'max_x_px': x + w,
            'min_y_px': y,
            'max_y_px': y + h
        }


        return (x_si, y_si), bounding_box_px

    # def count_pixels(self, mask):
    #     if mask is None:
    #         self.blob_pixel_counts.append(0)
    #         return
    #     num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    #     if num_labels > 1:
    #         largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    #         pixel_count = stats[largest_label, cv2.CC_STAT_AREA]
    #     else:
    #         pixel_count = 0
    #     self.blob_pixel_counts.append(pixel_count)


    # def plot_blob_pixel_counts(self):
    #     plt.figure()
    #     plt.plot(self.blob_pixel_counts, marker='o')
    #     plt.title("Blob Pixel Count Over Time")
    #     plt.xlabel("Time (s)")
    #     plt.ylabel("Pixels in Blob")
    #     plt.grid(True)
    #     plt.show()

    def plot_blood_volume(self):
        plt.figure()
        plt.plot(np.array(self.volume_data) * 1e6, marker='o')  # convert to mL
        plt.title("Estimated Blood Volume Over Time")
        plt.xlabel("Time (s)")
        plt.ylabel("Volume (mL)")
        plt.grid(True)
        plt.show()

    def start_volume_tracking(self):
        self.volume_data = []
        self._track_volume = True

        def track():
            while self._track_volume:
                time.sleep(1)
        threading.Thread(target=track, daemon=True).start()

    def stop_volume_tracking_and_plot(self):
        self._track_volume = False
        self.plot_blood_volume()





    def user_input_loop(self):
        while rclpy.ok():
            print("\nCurrent Position:")
            print(f"  X: {self.servo_cp_msg.pose.position.x / SimToSI}")
            print(f"  Y: {self.servo_cp_msg.pose.position.y / SimToSI}")
            print(f"  Z: {self.servo_cp_msg.pose.position.z / SimToSI}")

            try:
                key = int(input(
                    "\nOptions:\n"
                    "1 - Move manually (X/Y/Z input)\n"
                    "2 - Enter raw pixel coordinate (x, y)\n"
                    "3 - Follow blood blob center\n"
                    "4 - Global zigzag path\n"
                    "5 - Adaptive zigzag path\n"
                    "6 - Spiral inward path\n"
                    "7 - Spiral outward path\n"
                    "0 - Exit\n"
                ))

                if key == 0:
                    break

                elif key in [1, 2]:
                    if key == 1:    # 1 - Move manually (X/Y/Z input)
                        axis = int(input("Axis: 1-X, 2-Y, 3-Z: "))
                        target_value = float(input("Enter target value: "))
                        if axis == 1:
                            self.servo_cp_msg.pose.position.x = target_value * SimToSI
                        elif axis == 2:
                            self.servo_cp_msg.pose.position.y = target_value * SimToSI
                        elif axis == 3:
                            self.servo_cp_msg.pose.position.z = target_value * SimToSI
                        else:
                            print("Invalid axis.")

                    elif key == 2:  # 2 - Enter raw pixel coordinate (x, y)
                        y_px = int(input("Enter pixel X (0-99): "))
                        x_px = int(input("Enter pixel Y (0-99): "))
                        x_si = ((x_px / 99) * 4.4) - 2.2 + self.origin_offset_x
                        y_si = ((y_px / 99) * 4.4) - 2.2 + self.origin_offset_y
                        self.servo_cp_msg.pose.position.x = x_si
                        self.servo_cp_msg.pose.position.y = y_si

                elif key == 3:  # 3 - Follow blood blob center
                    if self.track_blob:
                        print("Already tracking blob.")
                        continue

                    self.track_blob = True
                    print("Following blob centre")

                    self.start_volume_tracking()

                    def track_loop():
                        start_time = time.time()
                        while time.time() - start_time < 20:
                            blob, _ = self.get_blob_center_and_size()
                            if blob is None:
                                continue
                            x_si, y_si = blob
                            x_px = int(((x_si - self.origin_offset_x + 2.2) / 4.4) * 99)
                            y_px = int(((y_si - self.origin_offset_y + 2.2) / 4.4) * 99)
                            #self.servo_cp_msg.pose.position.z = -27 * SimToSI

                            depth_si = self.get_depth_at_pixel(x_px, y_px)  # Get depth in meters
                            hover_offset = 0.005  # hover 0.5 cm above the tissue
                            if depth_si is not None:
                                z_px = - (depth_si - hover_offset)
                                self.servo_cp_msg.pose.position.z = z_px
                            else:
                                z_px = -27 * SimToSI  # fallback if depth missing
                                self.servo_cp_msg.pose.position.z = z_px

                            self.servo_cp_msg.pose.position.x = x_si
                            self.servo_cp_msg.pose.position.y = y_si
                            
                            print(f"End position (in px): X={x_px}, Y={y_px}, Z={z_px}")

                            time.sleep(0.2)

                        self.track_blob = False
                        self.stop_volume_tracking_and_plot()
                        print("Stopped tracking blob after 20 seconds.")

                    threading.Thread(target=track_loop, daemon=True).start()



                elif key == 4:  # 4 - Global zigzag path
                    self.start_volume_tracking()
                    print("Executing Global Zigzag Path")
    
                    self.servo_cp_msg.pose.position.z = -26 * SimToSI
                    z = self.servo_cp_msg.pose.position.z / SimToSI
                    time.sleep(1)

                    step_size_px = 5 #  # Move by 5 pixels at a time
                    step_size_si = (step_size_px / 99) * 4.4

                    width_px = 100
                    height_px = 100

                    num_steps_x = max(1, width_px // step_size_px + 1)
                    num_steps_y = max(1, height_px // step_size_px + 1)

                    start_x_px = 0
                    start_y_px = 99

                    # Adjust starting point (top-left corner of the grid)
                    start_x_si = (start_x_px / 99) * 4.4 - 2.2 + self.origin_offset_x
                    start_y_si = (start_y_px / 99) * 4.4 - 2.2 + self.origin_offset_y

                    for row in range(num_steps_y):
                        y_si = start_y_si - (row * step_size_si)
        
                        x_range = range(num_steps_x) if row % 2 == 0 else reversed(range(num_steps_x))

                        for col in x_range:
                            x_si = start_x_si + (col * step_size_si)
                           
                            x_px = int(((x_si - self.origin_offset_x + 2.2) / 4.4) * 99)
                            y_px = int(((y_si - self.origin_offset_y + 2.2) / 4.4) * 99)

                            self.servo_cp_msg.pose.position.x = x_si
                            self.servo_cp_msg.pose.position.y = y_si

                            depth_si = self.get_depth_at_pixel(x_px, y_px)
                            hover_offset = 0.005
                            if depth_si is not None:
                                self.servo_cp_msg.pose.position.z = - (depth_si - hover_offset)
                            else:
                                self.servo_cp_msg.pose.position.z = -27 * SimToSI


                            print(f"End position (in px): X={x_px}, Y={y_px}, Z={self.servo_cp_msg.pose.position.z / SimToSI}")

                            time.sleep(0.5)
            
                            self.completion_pub.publish(String(data="Zigzag path in progress"))

                            ## Check if blood has been removed
                            #if self.check_no_blood_left():
                            #    self.completion_pub.publish(String(data="No blood left after Zigzag"))
                            #    return  # Exit early if blood is gone

                    self.completion_pub.publish(String(data="Zigzag path completed"))
                    self.stop_volume_tracking_and_plot()


                elif key == 5:  # 5 - Adaptive zigzag path
                    self.start_volume_tracking()
                    print("Executing Adaptive Zigzag Path")
                    blob, bounding_box = self.get_blob_center_and_size()
                    if blob is None or bounding_box is None:
                        continue
    
                    self.servo_cp_msg.pose.position.z = -27 * SimToSI
                    z = self.servo_cp_msg.pose.position.z / SimToSI
                    time.sleep(1)

                    step_size_px = 5 #  # Move by 5 pixels at a time
                    step_size_si = (step_size_px / 99) * 4.4

                    min_x_px = bounding_box['min_x_px']
                    max_x_px = bounding_box['max_x_px']
                    min_y_px = bounding_box['min_y_px']
                    max_y_px = bounding_box['max_y_px']

                    min_y_px_flipped = 99 - max_y_px
                    max_y_px_flipped = 99 - min_y_px

                    # Calculate scan range in pixels
                    width_px = max_x_px - min_x_px
                    height_px = max_y_px_flipped - min_y_px_flipped

                    num_steps_x = max(1, width_px // step_size_px + 1)
                    num_steps_y = max(1, height_px // step_size_px + 1)

                    start_x_px = min_x_px
                    start_y_px = max_y_px_flipped  # top of flipped image


                    # Adjust starting point (top-left corner of the grid)
                    self.servo_cp_msg.pose.position.x = x_si
                    self.servo_cp_msg.pose.position.y = y_si

                    depth_si = self.get_depth_at_pixel(x_px, y_px)
                    hover_offset = 0.005
                    if depth_si is not None:
                        self.servo_cp_msg.pose.position.z = - (depth_si - hover_offset)
                    else:
                        self.servo_cp_msg.pose.position.z = -27 * SimToSI


                    for row in range(num_steps_y):
                        y_si = start_y_si - (row * step_size_si)  # Y-coordinate for the row
        
                        # Zigzag: even rows go left-to-right, odd rows go right-to-left
                        x_range = range(num_steps_x) if row % 2 == 0 else reversed(range(num_steps_x))

                        for col in x_range:
                            x_si = start_x_si + (col * step_size_si)  # X-coordinate for each step
            
                            # Update servo position with pixel coordinates
               
                            x_px = int(((x_si - self.origin_offset_x + 2.2) / 4.4) * 99)
                            y_px = int(((y_si - self.origin_offset_y + 2.2) / 4.4) * 99)

                            self.servo_cp_msg.pose.position.x = x_si
                            self.servo_cp_msg.pose.position.y = y_si

                            #if col % 5 == 0:  # Print position every 5th column
                            print(f"End position (in px): X={x_px}, Y={y_px}, Z={self.servo_cp_msg.pose.position.z / SimToSI}")

                            # Wait before moving to the next step (allow suctioning)
                            time.sleep(0.5)
            
                            self.completion_pub.publish(String(data="Zigzag path in progress"))

                            ## Check if blood has been removed
                            #if self.check_no_blood_left():
                            #    self.completion_pub.publish(String(data="No blood left after Zigzag"))
                            #    return  # Exit early if blood is gone

                    self.completion_pub.publish(String(data="Zigzag path completed"))
                    self.stop_volume_tracking_and_plot()


                elif key == 6:  # Inward spiral path
                    self.start_volume_tracking()
                    print("Starting inward spiral path.")
                    theta_step_deg = 15  # degrees
                    max_r = 40  # max pixel length to search from center
                    suction_pause = 0.5  # seconds between moves

                    theta_deg = 0
                    start_time = time.time()
                    while time.time() - start_time < 60:
                        result = self.get_blob_center_and_size()
                        if result is None:
                            continue
                        (x_si, y_si), _ = result

                        # Convert SI to pixel for raycasting
                        cx_px = int(((x_si - self.origin_offset_x + 2.2) / 4.4) * 99)
                        cy_px = 99 - int(((y_si - self.origin_offset_y + 2.2) / 4.4) * 99)

                        theta_rad = np.deg2rad(theta_deg)
                        found_edge = False

                        for r in range(1, max_r):
                            x_px = int(cx_px + r * np.cos(theta_rad))
                            y_px = int(cy_px - r * np.sin(theta_rad))  # flip y-axis
                            if (0 <= x_px < 100) and (0 <= y_px < 100):
                                if self.latest_mask is None or self.latest_mask[y_px, x_px] == 0:
                                    found_edge = True
                                    break
                            else:
                                break

                        if not found_edge:
                            theta_deg = (theta_deg + theta_step_deg) % 360
                            continue

                        # r-1 is the last blood pixel, convert to SI
                        edge_x_px = int(cx_px + (r-3) * np.cos(theta_rad))    # r-n increases the distance from last blood pixel
                        edge_y_px = int(cy_px - (r-3) * np.sin(theta_rad))

                        x_si = ((edge_x_px / 99) * 4.4) - 2.2 + self.origin_offset_x
                        y_si = (((99 - edge_y_px) / 99) * 4.4) - 2.2 + self.origin_offset_y

                        self.servo_cp_msg.pose.position.x = x_si
                        self.servo_cp_msg.pose.position.y = y_si

                        x_px = int(((x_si - self.origin_offset_x + 2.2) / 4.4) * 99)
                        y_px = int(((y_si - self.origin_offset_y + 2.2) / 4.4) * 99)

                        depth_si = self.get_depth_at_pixel(x_px, y_px)
                        hover_offset = 0.005
                        if depth_si is not None:
                            self.servo_cp_msg.pose.position.z = - (depth_si - hover_offset)
                        else:
                            self.servo_cp_msg.pose.position.z = -27 * SimToSI

                        theta_deg = (theta_deg + theta_step_deg) % 360
                        time.sleep(suction_pause)

                    self.completion_pub.publish(String(data="Spiral suction complete"))
                    print("Finished spiral suction.")
                    self.stop_volume_tracking_and_plot()



                elif key == 7:  # 7 - Spiral outward path
                    self.start_volume_tracking()
                    print("Executing Spiral Outward Path")
    
                    # Get blob center and size
                    blob, bounding_box = self.get_blob_center_and_size()
                    if blob is None or bounding_box is None:
                        continue
    
                    start_x_si, start_y_si = blob
    
                    # Adjust Z-coordinate for servo position
                    self.servo_cp_msg.pose.position.z = -25 * SimToSI
                    z = self.servo_cp_msg.pose.position.z / SimToSI
                    time.sleep(1)
    
                    min_x_px = bounding_box['min_x_px']
                    max_x_px = bounding_box['max_x_px']
                    min_y_px = bounding_box['min_y_px']
                    max_y_px = bounding_box['max_y_px']

                    min_y_px_flipped = 99 - max_y_px
                    max_y_px_flipped = 99 - min_y_px

                    width_px = max_x_px - min_x_px
                    height_px = max_y_px_flipped - min_y_px_flipped

                    if width_px > height_px:
                        maxdim = width_px
                    else:
                        maxdim = height_px

                    steps = 75
                    a = 0.01
                    b = 0.05 + (maxdim / 2500)  # Increase 'b' based on the max dimension of the blob (width or height)


                    for i in range(steps):
                        theta = i * 0.3
                        r = a + b * theta

                        x_si = start_x_si + r * np.cos(theta)
                        y_si = start_y_si + r * np.sin(theta)

                        self.servo_cp_msg.pose.position.x = x_si
                        self.servo_cp_msg.pose.position.y = y_si

                        x_px = int(((x_si - self.origin_offset_x + 2.2) / 4.4) * 99)
                        y_px = int(((y_si - self.origin_offset_y + 2.2) / 4.4) * 99)

                        depth_si = self.get_depth_at_pixel(x_px, y_px)
                        hover_offset = 0.005
                        if depth_si is not None:
                            self.servo_cp_msg.pose.position.z = - (depth_si - hover_offset)
                        else:
                            self.servo_cp_msg.pose.position.z = -27 * SimToSI


                        print(f"End position (in px): X={x_px}, Y={y_px}, Z={self.servo_cp_msg.pose.position.z / SimToSI}")

                        self.servo_cp_pub.publish(self.servo_cp_msg)
                        time.sleep(0.5)

                        self.completion_pub.publish(String(data="Spiral path in progress"))

                        # Check if blood has been removed, and if so, stop the spiral
                        if self.check_no_blood_left():
                            self.completion_pub.publish(String(data="No blood left after Spiral"))
                            break  # Exit the loop if blood is gone

                    # Final completion notification
                    self.completion_pub.publish(String(data="Spiral path completed"))
                    self.stop_volume_tracking_and_plot()


                else:
                    print("Invalid option.")

            except ValueError:
                print("Invalid input. Please enter numeric values.")

def main(args=None):
    rclpy.init(args=args)
    node = SurChalCrtkTestNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
