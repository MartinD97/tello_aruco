#!/usr/bin/env python3

import pprint
import math
import rclpy
import threading
import numpy as np
import time
import av
import tf2_ros
import cv2
import yaml

from djitellopy import Tello
from scipy.spatial.transform import Rotation as R

from rclpy.node import Node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from tello_msg.msg import TelloStatus, TelloID, TelloWifiConfig
from std_msgs.msg import Empty, UInt8, Bool, String
from sensor_msgs.msg import Image, Imu, BatteryState, Temperature, CameraInfo
from geometry_msgs.msg import Twist, TransformStamped, PoseArray, Pose, PoseStamped, Point
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import ament_index_python
from visualization_msgs.msg import Marker

# Tello ROS node class, inherits from the Tello controller object.
#
# Can be configured to be used by multiple drones, publishes, all data collected from the drone and provides control using ROS messages.
class TelloNode():
    def __init__(self, node):
        # ROS node
        self.node = node

        self.node.declare_parameter(
            name="calibration_file",
            value="/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/config/calibration_tello.yaml",
            descriptor=ParameterDescriptor(
                type=ParameterType.PARAMETER_STRING,
                description="Path to camera calibration file (yaml format).",
            ),
        )

        self.calibration_file = (
            self.node.get_parameter("calibration_file").get_parameter_value().string_value
        )

        # Declare parameters
        self.node.declare_parameter('connect_timeout', 10.0)
        self.node.declare_parameter('tello_ip', '192.168.10.1')
        self.node.declare_parameter('tf_base', 'map')
        self.node.declare_parameter('tf_drone', 'tello_frame')
        self.node.declare_parameter('tf_pub', True)
        self.node.declare_parameter('camera_info_file', '')

        # Get parameters
        self.connect_timeout = float(self.node.get_parameter('connect_timeout').value)
        self.tello_ip = str(self.node.get_parameter('tello_ip').value)
        self.tf_base = str(self.node.get_parameter('tf_base').value)
        self.tf_drone = str(self.node.get_parameter('tf_drone').value)
        self.tf_pub = bool(self.node.get_parameter('tf_pub').value)
        self.camera_info_file = str(self.node.get_parameter('camera_info_file').value)

        # Camera information loaded from calibration yaml
        self.camera_info = None

        # Check if camera info file was received as argument
        if len(self.camera_info_file) == 0:
            self.camera_info_file = '/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/config/calibration_tello.yaml'

        with open(self.camera_info_file, 'r') as file:
           self.camera_info = yaml.load(file, Loader=yaml.FullLoader)
           self.node.get_logger().info(f'Tello: Camera information YAML loaded {self.camera_info}')

        # Configure drone connection
        Tello.TELLO_IP = self.tello_ip
        Tello.RESPONSE_TIMEOUT = int(self.connect_timeout)

        # Connect to drone
        self.node.get_logger().info('Tello: Connecting to drone')

        self.tello = Tello()
        self.tello.connect()

        self.node.get_logger().info('Tello: Connected to drone')

        # Publishers and subscribers
        self.setup_publishers()
        self.setup_subscribers()

        self.position = [0.0, 0.0, 0.0]
        # Processing threads
        self.start_video_capture()
        self.start_tello_status()
        self.start_tello_odom()

        self.node.get_logger().info('Tello: Driver node ready')

    # Setup ROS publishers of the node.
    def setup_publishers(self):
        self.pub_image_raw = self.node.create_publisher(Image, 'tello/image_raw/Image', 1)
        self.pub_camera_info = self.node.create_publisher(CameraInfo, 'tello/camera_info', 1)
        self.pub_status = self.node.create_publisher(TelloStatus, 'tello/status', 1)
        self.pub_id = self.node.create_publisher(TelloID, 'tello/id', 1)
        self.pub_imu = self.node.create_publisher(Imu, 'tello/imu', 1)
        self.pub_battery = self.node.create_publisher(BatteryState, 'tello/battery', 1)
        self.pub_temperature = self.node.create_publisher(Temperature, 'tello/temperature', 1)
        self.pub_odom = self.node.create_publisher(Odometry, 'tello/odom', 1)
        self.publisher_pose = self.node.create_publisher(PoseStamped, 'tello/pose', 10)
        self.publisher_yaw = self.node.create_publisher(PoseStamped, 'tello/yaw', 10)

        # TF broadcaster
        if self.tf_pub:
            self.tf_broadcaster = tf2_ros.TransformBroadcaster(self.node)

    # Setup the topic subscribers of the node.
    def setup_subscribers(self):
        self.sub_emergency = self.node.create_subscription(Empty, 'emergency', self.cb_emergency, 1)
        self.sub_takeoff = self.node.create_subscription(Empty, 'takeoff', self.cb_takeoff, 1)
        self.sub_land = self.node.create_subscription(Empty, 'land', self.cb_land, 1)
        self.sub_control = self.node.create_subscription(Twist, 'control', self.cb_control, 1)
        self.sub_flip = self.node.create_subscription(String, 'flip', self.cb_flip, 1)
        self.sub_wifi_config = self.node.create_subscription(TelloWifiConfig, 'wifi_config', self.cb_wifi_config, 1)
        self.xyz_orbslam = self.node.create_subscription(PoseStamped, '/vicon/Tello_42/Tello_42', self.pointmsg, 1)


    def pointmsg(self, msg):
        # set for us configuration
        self.position = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        #self.node.get_logger().info(f"Received: x={self.position[0]:.3f}, y={self.position[0]:.3f}, z={self.position[2]:.3f}")

    # Get the orientation of the drone as a quaternion
    def get_orientation_quaternion(self):
        # in order, rotation of axes x, y, z
        deg_to_rad = math.pi / 180.0
        yaw = PoseStamped()
        yaw.pose.orientation.x = -(self.tello.get_yaw() * deg_to_rad)
        yaw.pose.orientation.y = -(self.tello.get_pitch() * deg_to_rad)
        yaw.pose.orientation.z = self.tello.get_roll() * deg_to_rad
        self.publisher_yaw.publish(yaw)
        return euler_to_quaternion([
            -(self.tello.get_yaw() * deg_to_rad),
            -(self.tello.get_pitch() * deg_to_rad),
            self.tello.get_roll() * deg_to_rad
        ])

    # Start drone info thread
    def start_tello_odom(self, rate=1.0/10.0):
        def status_odom():
            while True:
                # matrix_rot = np.array([
                #     [0, 0, 1],
                #     [0, -1, 0],
                #     [1, 0, 0]
                # ])
                q = self.get_orientation_quaternion()
                # q_matrix = R.from_quat(q).as_matrix()
                # prod_matrix = np.dot(matrix_rot, q_matrix)
                # q = R.from_matrix(prod_matrix).as_quat()

                # TF
                if self.tf_pub:
                    t = TransformStamped()
                    t.header.stamp = self.node.get_clock().now().to_msg()
                    t.header.frame_id = self.tf_base
                    t.child_frame_id = self.tf_drone
                    t.transform.translation.x = 0.0
                    t.transform.translation.y = 0.0
                    t.transform.translation.z = self.tello.get_barometer() / 100

                # IMU
                if self.pub_imu.get_subscription_count() > 0:
                    msg = Imu()
                    msg.header.stamp = self.node.get_clock().now().to_msg()
                    msg.header.frame_id = self.tf_drone
                    msg.linear_acceleration.x = self.tello.get_acceleration_x() / 100.0
                    msg.linear_acceleration.y = self.tello.get_acceleration_y() / 100.0
                    msg.linear_acceleration.z = self.tello.get_acceleration_z() / 100.0
                    msg.orientation.x = q[0]
                    msg.orientation.y = q[1]
                    msg.orientation.z = q[2]
                    msg.orientation.w = q[3]
                    self.pub_imu.publish(msg)

                # Odometry
                if self.pub_odom:
                    odom_msg = Odometry()
                    odom_msg.header.stamp = self.node.get_clock().now().to_msg()
                    odom_msg.header.frame_id = self.tf_base
                    odom_msg.child_frame_id = self.tf_drone

                    try:
                        odom_msg.pose.pose.position.x = self.position[0]
                        odom_msg.pose.pose.position.y = self.position[1]
                        odom_msg.pose.pose.position.z = self.position[2]
                    except:
                        self.node.get_logger().info('except odom')
                        pass

                    odom_msg.pose.pose.orientation.x = q[0]
                    odom_msg.pose.pose.orientation.y = q[1]
                    odom_msg.pose.pose.orientation.z = q[2]
                    odom_msg.pose.pose.orientation.w = q[3]

                    odom_msg.twist.twist.linear.x = float(self.tello.get_speed_x()) / 100.0
                    odom_msg.twist.twist.linear.y = float(self.tello.get_speed_y()) / 100.0
                    odom_msg.twist.twist.linear.z = float(self.tello.get_speed_z()) / 100.0
                    self.pub_odom.publish(odom_msg)

                #tf pose
                t_tello = TransformStamped()
                t_tello.header.frame_id = self.tf_base
                t_tello.child_frame_id = self.tf_drone
                t_tello.header.stamp = self.node.get_clock().now().to_msg()

                t_tello.transform.translation.x = self.position[0]
                t_tello.transform.translation.y = self.position[1]
                t_tello.transform.translation.z = self.position[2]

                t_tello.transform.rotation.x = q[0]
                t_tello.transform.rotation.y = q[1]
                t_tello.transform.rotation.z = q[2]
                t_tello.transform.rotation.w = q[3]

                self.tf_broadcaster.sendTransform(t_tello)
                time.sleep(rate)

        thread = threading.Thread(target=status_odom)
        thread.start()
        return thread

    # Start drone info thread
    def start_tello_status(self, rate=1.0/2.0):
        def status_loop():
            while True:
                # Battery
                if self.pub_battery.get_subscription_count() > 0:
                    msg = BatteryState()
                    msg.header.frame_id = self.tf_drone
                    msg.percentage = float(self.tello.get_battery())
                    if msg.percentage % 10 == 0: self.node.get_logger().info(f'Battery: {msg.percentage}')
                    msg.voltage = 3.8
                    msg.design_capacity = 1.1
                    msg.present = True
                    msg.power_supply_technology = 2 # POWER_SUPPLY_TECHNOLOGY_LION
                    msg.power_supply_status = 2 # POWER_SUPPLY_STATUS_DISCHARGING
                    self.pub_battery.publish(msg)

                # Temperature
                if self.pub_temperature.get_subscription_count() > 0:
                    msg = Temperature()
                    msg.header.frame_id = self.tf_drone
                    msg.temperature = self.tello.get_temperature()
                    if msg.temperature % 10 == 0: self.node.get_logger().info(f'Temperature: {msg.temperature}')
                    msg.variance = 0.0
                    self.pub_temperature.publish(msg)

                # Tello Status
                if self.pub_status.get_subscription_count() > 0:
                    msg = TelloStatus()
                    msg.acceleration.x = self.tello.get_acceleration_x()
                    msg.acceleration.y = self.tello.get_acceleration_y()
                    msg.acceleration.z = self.tello.get_acceleration_z()

                    msg.speed.x = float(self.tello.get_speed_x())
                    msg.speed.y = float(self.tello.get_speed_y())
                    msg.speed.z = float(self.tello.get_speed_z())

                    msg.pitch = self.tello.get_pitch()
                    msg.roll = self.tello.get_roll()
                    msg.yaw = self.tello.get_yaw()

                    msg.barometer = int(self.tello.get_barometer())
                    msg.distance_tof = self.tello.get_distance_tof()

                    msg.fligth_time = self.tello.get_flight_time()

                    msg.battery = self.tello.get_battery()

                    msg.highest_temperature = self.tello.get_highest_temperature()
                    msg.lowest_temperature = self.tello.get_lowest_temperature()
                    msg.temperature = self.tello.get_temperature()

                    msg.wifi_snr = self.tello.query_wifi_signal_noise_ratio()

                    self.pub_status.publish(msg)

                # Tello ID
                if self.pub_id.get_subscription_count() > 0:
                    msg = TelloID()
                    msg.sdk_version = self.tello.query_sdk_version()
                    msg.serial_number = self.tello.query_serial_number()
                    self.pub_id.publish(msg)
 
                # Camera info
                if self.pub_camera_info.get_subscription_count() > 0:
                    msg = CameraInfo()
                    msg.height = self.camera_info["image_height"]
                    msg.width = self.camera_info["image_width"]
                    msg.distortion_model = self.camera_info["distortion_model"]
                    msg.D = self.camera_info["distortion_coefficients"]["data"]
                    msg.K = self.camera_info["camera_matrix"]["data"]
                    msg.R = self.camera_info["rectification_matrix"]["data"]
                    msg.P = self.camera_info["projection_matrix"]["data"]
                    self.pub_camera_info.publish(msg)

                # Sleep
                time.sleep(rate)

        thread = threading.Thread(target=status_loop)
        thread.start()
        return thread


    # Start video capture thread.
    def start_video_capture(self, rate=1.0/30.0):
        # Enable tello stream
        self.tello.streamon()

        # OpenCV bridge
        self.bridge = CvBridge()

        def video_capture_thread():
            frame_read = self.tello.get_frame_read()

            while True:
                # Get frame from drone
                frame = frame_read.frame

                # Publish opencv frame using CV bridge
                msg = self.bridge.cv2_to_imgmsg(np.array(frame), 'bgr8')
                msg.header.frame_id = self.tf_drone
                self.pub_image_raw.publish(msg)

                time.sleep(rate)


        # We need to run the recorder in a seperate thread, otherwise blocking options would prevent frames from getting added to the video
        thread = threading.Thread(target=video_capture_thread)
        thread.start()
        return thread

    # Terminate the code and shutdown node.
    def terminate(self, err):
        self.node.get_logger().error(str(err))
        self.tello.end()
        rclpy.shutdown()

    # Stop all movement in the drone
    def cb_emergency(self, msg):
        self.tello.emergency()

    # Drone takeoff message control
    def cb_takeoff(self, msg):
        self.tello.takeoff()

    # Land the drone message callback
    def cb_land(self, msg):
        self.tello.land()

    # Control messages received use to control the drone "analogically"
    #
    # This method of controls allow for more precision in the drone control.
    #
    # Receives the linear and angular velocities to be applied from -100 to 100.
    def cb_control(self, msg):
        self.tello.send_rc_control(int(msg.linear.x), int(msg.linear.y), int(msg.linear.z), int(msg.angular.z))
        #self.node.get_logger().info(f"cb_control: {msg}")

    # Configure the wifi credential that should be used by the drone.
    #
    # The drone will be restarted after the credentials are changed.
    def cb_wifi_config(self, msg):
        self.tello.set_wifi_credentials(msg.ssid, msg.password)

    # Perform a drone flip in a direction specified.
    #
    # Directions can be "r" for right, "l" for left, "f" for forward or "b" for backward.
    def cb_flip(self, msg):
        self.tello.flip(msg.data)

# Convert a rotation from euler to quaternion.
def euler_to_quaternion(r):
    (yaw, pitch, roll) = (r[0], r[1], r[2])
    qx = math.sin(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) - math.cos(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    qy = math.cos(roll/2) * math.sin(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.cos(pitch/2) * math.sin(yaw/2)
    qz = math.cos(roll/2) * math.cos(pitch/2) * math.sin(yaw/2) - math.sin(roll/2) * math.sin(pitch/2) * math.cos(yaw/2)
    qw = math.cos(roll/2) * math.cos(pitch/2) * math.cos(yaw/2) + math.sin(roll/2) * math.sin(pitch/2) * math.sin(yaw/2)
    return [qx, qy, qz, qw]

# Convert rotation from quaternion to euler.
def quaternion_to_euler(q):
    (x, y, z, w) = (q[0], q[1], q[2], q[3])
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return [yaw, pitch, roll]

def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('tello')
    drone = TelloNode(node)

    rclpy.spin(node)

    drone.cb_shutdown()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
