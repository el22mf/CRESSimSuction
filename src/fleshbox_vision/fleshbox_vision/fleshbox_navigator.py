import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from PyKDL import Rotation
import threading

SimToSI = 0.05  # Simulation to SI conversion factor

class RobotData:
    def __init__(self):
        self.measured_cp = PoseStamped()

class SurChalCrtkTestNode(Node):
    def __init__(self):
        super().__init__('sur_chal_crtk_test')
        self.robData = RobotData()

        arm_name = "PSM1"
        measured_cp_topic = arm_name + "/measured_cp"
        self.servo_cp_topic = arm_name + "/servo_cp"

        self.create_subscription(PoseStamped, measured_cp_topic, self.measured_cp_cb, 10)
        self.servo_cp_pub = self.create_publisher(PoseStamped, self.servo_cp_topic, 10)

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

    def measured_cp_cb(self, msg):
        self.robData.measured_cp = msg

    def timer_callback(self):
        self.servo_cp_pub.publish(self.servo_cp_msg)

    def user_input_loop(self):
        while rclpy.ok():
            print("Current Position:")
            print(f"  X: {self.servo_cp_msg.pose.position.x / SimToSI}")
            print(f"  Y: {self.servo_cp_msg.pose.position.y / SimToSI}")
            print(f"  Z: {self.servo_cp_msg.pose.position.z / SimToSI}")

            try:
                key = int(input("Press: \n"
                                "1 - Move along X axis, \n"
                                "2 - Move along Y axis, \n"
                                "3 - Move along Z axis, \n"
                                "0 - Exit\n"))
                if key == 0:
                    break
                if key in [1, 2, 3]:
                    target_value = float(input("Enter target position value: "))
                    if key == 1:
                        self.servo_cp_msg.pose.position.x = target_value * SimToSI
                    elif key == 2:
                        self.servo_cp_msg.pose.position.y = target_value * SimToSI
                    elif key == 3:
                        self.servo_cp_msg.pose.position.z = target_value * SimToSI
                else:
                    print("Invalid Entry")
            except ValueError:
                print("Invalid input. Please enter a number.")


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