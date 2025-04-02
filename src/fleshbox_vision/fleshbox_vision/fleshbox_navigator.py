import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JointState
from PyKDL import Rotation

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

        valid_key = False
        while not valid_key:
            try:
                key = int(input("Press: \n"
                                "1 - Move along X axis, \n"
                                "2 - Move along Y axis, \n"
                                "3 - Move along Z axis \n"))
            except ValueError:
                key = None
            if key in [1, 2, 3]:
                valid_key = True
                self.mode = key
                self.target_value = float(input("Enter target position value: "))
            else:
                print("Invalid Entry")

        self.timer = self.create_timer(0.02, self.timer_callback)

    def measured_cp_cb(self, msg):
        self.robData.measured_cp = msg

    def timer_callback(self):
        if self.mode == 1:
            self.servo_cp_msg.pose.position.x = self.target_value * SimToSI
        elif self.mode == 2:
            self.servo_cp_msg.pose.position.y = self.target_value * SimToSI
        elif self.mode == 3:
            self.servo_cp_msg.pose.position.z = self.target_value * SimToSI

        self.servo_cp_pub.publish(self.servo_cp_msg)


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
