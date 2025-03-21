import rclpy, math, time, cv2, yaml
import tf_transformations as tf
import numpy as np
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ros2_aruco_interfaces.msg import ArucoMarkers

class TelloController(Node):
    def __init__(self):
        super().__init__('tello_controller')
        # PID's parameters
        self.kp, self.ki, self.kd = 1.5, 0.05, 0.2
        self.error_sum_x = self.error_sum_y = self.error_sum_z = self.error_sum_yaw = 0.0
        self.last_error_x = self.last_error_y = self.last_error_z = self.last_error_yaw = 0.0

        # Home
        self.home_x, self.home_y, self.home_z, self.home_yaw = 0.0, 0.0, 1.5, 0.0
        self.target_x, self.target_y, self.target_z, self.target_yaw = self.home_x, self.home_y, self.home_z, self.home_yaw

        # State of drone
        self.tello_x = self.tello_y = self.tello_z = self.tello_yaw = 0.0
        self.drone_pose = None
        self.image = None
        self.pose_buffer = {}
        self.bridge = CvBridge()
        self.seen_markers = set()

        # Timeout marker
        self.last_marker_time = time.time()
        self.marker_timeout = 2.0  # secondi

        # Marker's parameters
        self.dist_tolerance = 0.1
        self.dist_from_marker = 0.3
        self.linear_speed = 25.0
        self.angular_speed = 50.0
        self.pose_timeout = 3.0

        # Flag
        self.flag_target = 0
        self.flag_tello_aruco = False

        # Load offset camera
        self.offset_camera = self.load_transform_config('/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/config/offset_camera.yaml')
        self.pc_x = self.offset_camera['translation']['x']
        self.pc_y = self.offset_camera['translation']['y']
        self.pc_z = self.offset_camera['translation']['z']

        # Publisher e Subscriber
        self.cmd_pub = self.create_publisher(Twist, 'control', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/vicon/Tello_42/Tello_42', self.vicon_callback, 10)
        self.marker_sub = self.create_subscription(ArucoMarkers, '/pc/aruco_markers', self.aruco_pose_callback, 10)
        self.marker_tello_sub = self.create_subscription(ArucoMarkers, '/tello/aruco_markers', self.aruco_pose_tello_callback, 10)
        self.create_timer(0.1, self.control_loop)

    def load_transform_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def vicon_callback(self, msg):
        self.tello_x = msg.pose.position.x
        self.tello_y = msg.pose.position.y
        self.tello_z = msg.pose.position.z
        q = msg.pose.orientation
        _, _, self.tello_yaw = tf.euler_from_quaternion([q.x, q.y, q.z, q.w]) # CHECK ORIENTATION
        self.drone_pose = {"x": self.tello_x, "y": self.tello_y, "z": self.tello_z}

    def aruco_pose_callback(self, msg):
        if 0 < msg.marker_ids[0] < 50:
            current_time = time.time()
            for idx, pose in enumerate(msg.poses):
                marker_id = f"marker_{msg.marker_ids[0]}"
                if marker_id not in self.seen_markers:
                    self.pose_buffer[marker_id] = {
                        "x": pose.position.x + self.pc_x, 
                        "y": pose.position.z + self.pc_y, 
                        "z": -pose.position.y + self.pc_z,
                        "timestamp": current_time
                    }
                    #self.get_logger().info(f"add marker: {self.pose_buffer[marker_id]}")
                    #self.get_logger().info(f"add marker id: {msg.marker_ids[0]}")
            self.pose_buffer = {k: v for k, v in self.pose_buffer.items() if current_time - v["timestamp"] < self.pose_timeout}
            if self.pose_buffer:
                self.last_marker_time = current_time

    def aruco_pose_tello_callback(self, msg):
        if 0 < msg.marker_ids[0] < 50:
            self.flag_tello_aruco = True
            self.last_marker_detection_time = time.time()

    def camera_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def compute_pid(self, error, error_sum, last_error):
        error_sum += error
        delta_error = error - last_error
        error_sum = np.clip(error_sum, -5.0, 5.0)
        control = np.clip(self.kp * error + self.ki * error_sum + self.kd * delta_error, -1.0, 1.0)
        return control, error_sum, error

    def control_loop(self):
        if not self.drone_pose:
            return

        if time.time() - self.last_marker_detection_time > self.marker_timeout:
            self.flag_tello_aruco = False
        if time.time() - self.last_marker_time > self.marker_timeout:
            self.pose_buffer.clear()
            self.target_x, self.target_y, self.target_z = self.home_x, self.home_y, self.home_z
            if self.flag_target == 0 or self.flag_target == 2:
                self.get_logger().info(f"Target to home, sec {time.time():.2f}")
                self.flag_target = 1

        if self.pose_buffer:
            closest_marker = min(self.pose_buffer.items(), key=lambda m: self.dist_from_tello(m[1]))
            marker_id, marker_pose = closest_marker
            dir_x = self.pc_x - marker_pose["x"]
            dir_y = self.pc_y - marker_pose["y"]
            dir_z = self.pc_z - marker_pose["z"]
            norm = np.sqrt(dir_x**2 + dir_y**2 + dir_z**2)
            if norm > 0: dir_x, dir_y, dir_z = dir_x / norm, dir_y / norm, dir_z / norm
            self.target_x = marker_pose["x"] + dir_x * self.dist_from_marker
            self.target_y = marker_pose["y"] + dir_y * self.dist_from_marker
            self.target_z = marker_pose["z"] + dir_z * self.dist_from_marker
            #self.target_x, self.target_y, self.target_z = marker_pose["x"], marker_pose["y"] - 0.30, marker_pose["z"]
            if self.flag_target == 0 or self.flag_target == 1:
                self.get_logger().info(f"Target pos: {self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}")
                self.get_logger().info(f"Marker id: {marker_id}")
                self.flag_target = 2
    
        error_x = self.target_x - self.tello_x
        error_y = self.target_y - self.tello_y
        error_z = self.target_z - self.tello_z
        distance = self.dist_from_tello({"x": self.target_x, "y": self.target_y, "z": self.target_z})
        speed_factor = np.clip(distance, 0.1, 1.0)

        if distance <= self.dist_tolerance and self.pose_buffer:
            # marker = min(self.pose_buffer.values(), key=lambda m: self.dist_from_tello(m[1]))
            error_yaw = math.atan2(marker_pose["y"] - self.tello_y, marker_pose["x"] - self.tello_x) - self.tello_yaw # CHECK, +/- math.pi ?
            self.get_logger().info(f'YAW. cur: {self.tello_yaw:.2f}, err: {error_yaw:.2f}')
        else: error_yaw = math.atan2(error_y, error_x) - self.tello_yaw # CHECK + math.pi ?
        # error_yaw = (math.atan2(error_y, error_x) - self.tello_yaw + math.pi) % (2 * math.pi) - math.pi
            
        if abs(error_yaw) >= math.pi/4: control_x, control_y, control_z = 0, 0, 0
        else:
            control_x, self.error_sum_x, self.last_error_x = self.compute_pid(error_x, self.error_sum_x, self.last_error_x)
            control_y, self.error_sum_y, self.last_error_y = self.compute_pid(error_y, self.error_sum_y, self.last_error_y)
            control_z, self.error_sum_z, self.last_error_z = self.compute_pid(error_z, self.error_sum_z, self.last_error_z)

        control_yaw, self.error_sum_yaw, self.last_error_yaw = self.compute_pid(error_yaw, self.error_sum_yaw, self.last_error_yaw)
        #self.get_logger().info(f'YAW. err: {error_yaw:.2f}, contr: {control_yaw}') #cur: {self.tello_yaw:.2f}, err: {error_yaw:.2f}')

        if distance <= self.dist_tolerance and error_yaw <= math.pi/4 and self.flag_tello_aruco:
            if self.image is None: self.get_logger().info(f"Video not received")
            else:
                self.save_image(marker_id)
                self.seen_markers.add(marker_id)
                self.get_logger().info(f"Marker processed: {marker_id}")
                self.pose_buffer.pop(marker_id, None)

        cmd = Twist()
        cmd.linear.x = 0.0 # control_y * self.linear_speed * speed_factor
        if error_x > 0: cmd.linear.y = control_x * self.linear_speed * speed_factor
        else: cmd.linear.y = - control_x * self.linear_speed * speed_factor
        cmd.linear.z = 0.0 # control_z * self.linear_speed * speed_factor
        if error_yaw > - math.pi: cmd.angular.z = - control_yaw * self.angular_speed
        else: cmd.angular.z = control_yaw * self.angular_speed
        self.cmd_pub.publish(cmd)

    def dist_from_tello(self, msg):
        return math.sqrt((msg["x"] - self.tello_x) ** 2 +
                         (msg["y"] - self.tello_y) ** 2 +
                         (msg["z"] - self.tello_z) ** 2)

    def save_image(self, marker_id):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_marker_{marker_id}.jpg"
        cv2.imwrite(filename, self.image)
        self.get_logger().info(f"Img saved: {filename}")

def main(args=None):
    rclpy.init(args=args)
    node = TelloController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
