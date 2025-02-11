import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist
from sensor_msgs.msg import Image
import math
from cv_bridge import CvBridge
import cv2
import time
from collections import deque

class TelloMarkerFollower(Node):
    def __init__(self):
        super().__init__('tello_marker_follower')
        self.declare_parameter('target_distance', 1.0)
        self.declare_parameter('linear_speed', 0.5)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('pose_timeout', 2.0)
        self.declare_parameter('pose_filter_window', 5)

        self.target_distance = self.get_parameter('target_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.pose_timeout = self.get_parameter('pose_timeout').value
        self.filter_window = self.get_parameter('pose_filter_window').value

        self.pose_buffer = {}
        self.active_marker = None
        self.bridge = CvBridge()
        self.image = None

        self.pose_subscriber = self.create_subscription(PoseArray, '/pc/aruco_poses', self.aruco_pose_callback, 10)
        self.image_subscriber = self.create_subscription(Image, '/tello/camera/image_raw', self.camera_callback, 10)
        self.cmd_vel_publisher = self.create_publisher(Twist, '/tello/cmd_vel', 10)
        self.create_timer(0.1, self.navigate_to_marker)

    def aruco_pose_callback(self, msg: PoseArray):
        for idx, pose in enumerate(msg.poses):
            marker_id = f"marker_{idx}"
            self.pose_buffer[marker_id] = {
                "x": pose.position.x,
                "y": pose.position.y,
                "z": pose.position.z,
                "timestamp": time.time()
            }

        current_time = time.time()
        self.pose_buffer = {
            marker_id: pose
            for marker_id, pose in self.pose_buffer.items()
            if current_time - pose["timestamp"] < self.pose_timeout
        }

        if not self.active_marker and self.pose_buffer:
            self.active_marker = next(iter(self.pose_buffer.keys()))

    def camera_callback(self, msg: Image):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def navigate_to_marker(self):
        if self.active_marker and self.active_marker in self.pose_buffer:
            marker_pose = self.pose_buffer[self.active_marker]
            x, y, z = marker_pose["x"], marker_pose["y"], marker_pose["z"]
            distance = math.sqrt(x**2 + y**2 + z**2)
            yaw_angle = math.atan2(y, x)

            twist = Twist()
            if distance > self.target_distance:
                twist.linear.x = min(self.linear_speed, self.linear_speed * (distance - self.target_distance))
                twist.angular.z = min(self.angular_speed, self.angular_speed * yaw_angle)
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                self.save_image(self.active_marker)
                self.active_marker = None

            self.cmd_vel_publisher.publish(twist)

    def save_image(self, marker_id):
        if self.image is not None:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            filename = f"{timestamp}_{marker_id}.jpg"
            cv2.imwrite(filename, self.image)
            self.get_logger().info(f"Immagine salvata: {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = TelloMarkerFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()