import rclpy, math, time, cv2, yaml, os
import tf_transformations as tf
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseStamped, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ros2_aruco_interfaces.msg import ArucoMarkers

class TelloController(Node):
    def __init__(self):
        super().__init__('tello_controller')
        # Publisher e Subscriber
        self.cmd_pub = self.create_publisher(Twist, 'control', 10)
        self.pose_sub = self.create_subscription(PoseStamped, '/vicon/Tello_42/Tello_42', self.vicon_callback, 10)
        self.marker_sub = self.create_subscription(ArucoMarkers, '/pc/aruco_markers', self.aruco_pose_callback, 10)
        self.marker_tello_sub = self.create_subscription(ArucoMarkers, '/tello/aruco_markers', self.aruco_pose_tello_callback, 10)
        self.image_tello_sub = self.create_subscription(Image, 'tello/image_raw/Image', self.camera_callback, 10)

        # PID's parameters
        self.kp, self.ki, self.kd = 1.5, 0.05, 0.2
        self.error_sum_x = self.error_sum_y = self.error_sum_z = self.error_sum_yaw = 0.0
        self.last_error_x = self.last_error_y = self.last_error_z = self.last_error_yaw = 0.0

        # Home
        self.home_x, self.home_y, self.home_z, self.home_yaw = 0.0, 0.0, 2.0, 0.0
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
        self.marker_timeout = 1.0  # sec

        #Timeout vicon
        self.last_vicon_time = 0

        # Marker's parameters
        self.dist_tolerance = 0.3
        self.yaw_tolerance = math.pi/4
        self.dist_from_marker = 0.4
        self.linear_speed = 25.0
        self.angular_speed = 50.0
        self.pose_timeout = 3.0
        
        # Flag
        self.flag_target = 0
        self.flag_tello_aruco = False
        self.last_marker_detection_time = 0

        # Load offset camera
        self.offset_camera = self.load_transform_config('/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/config/offset_camera.yaml')
        self.pc_x = self.offset_camera['translation']['x']
        self.pc_y = self.offset_camera['translation']['y']
        self.pc_z = self.offset_camera['translation']['z']

        # Save data, {marker_id: {x, y, z, timestamp}
        self.data_dir = self.create_data_directory()
        self.pc_marker_positions = {}
        self.tello_marker_positions = {}
        
        # Graph
        self.drone_trajectory = []  # (timestamp, x, y, z)
        self.last_plot_time = time.time()
        self.plot_interval = 10.0  # sec
        self.marker_errors = {}
        
        self.create_timer(0.01, self.control_loop)
        self.create_timer(1.0, self.data_logging_loop)

    def create_data_directory(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_path = "tello_ws/src/tello_pkg/tello_data"
        data_dir = os.path.join(base_path, f"tello_data_{timestamp}")
        os.makedirs(base_path, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        return data_dir

    def load_transform_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def vicon_callback(self, msg):
        self.tello_x = msg.pose.position.x
        self.tello_y = msg.pose.position.y
        self.tello_z = msg.pose.position.z
        q = msg.pose.orientation
        _, _, self.tello_yaw = tf.euler_from_quaternion([q.x, q.y, q.z, q.w])# CHECK ORIENTATION
        self.drone_pose = {"x": self.tello_x, "y": self.tello_y, "z": self.tello_z}
        self.last_vicon_time = time.time()
        #self.get_logger().info(f'YAW. err: {self.tello_yaw:.2f}')

        self.drone_trajectory.append((time.time(), self.tello_x, self.tello_y, self.tello_z))
        if time.time() - self.last_plot_time >= self.plot_interval:
            self.generate_trajectory_plots()
            self.generate_error_plots()
            self.last_plot_time = time.time()

    def aruco_pose_callback(self, msg):
        if msg.marker_ids and 0 < msg.marker_ids[0] < 50:
            current_time = time.time()
            for idx, pose in enumerate(msg.poses):
                marker_id = f"marker_{msg.marker_ids[0]}"
                global_x = pose.position.x + self.pc_x
                global_y = pose.position.z + self.pc_y
                global_z = -pose.position.y + self.pc_z

                if marker_id not in self.seen_markers:
                    self.pose_buffer[marker_id] = {
                        "x": global_x, 
                        "y": global_y, 
                        "z": global_z,
                        "timestamp": current_time
                    }
                    #self.get_logger().info(f"add marker: {self.pose_buffer[marker_id]}")
                    #self.get_logger().info(f"add marker id: {msg.marker_ids[0]}")
                self.pc_marker_positions[marker_id] = {
                    "x": global_x,
                    "y": global_y,
                    "z": global_z,
                    "timestamp": current_time
                }
                self.save_marker_positions(marker_id)

            self.pose_buffer = {k: v for k, v in self.pose_buffer.items() if current_time - v["timestamp"] < self.pose_timeout}
            if self.pose_buffer:
                self.last_marker_time = current_time

    def aruco_pose_tello_callback(self, msg):
        if msg.marker_ids and 0 < msg.marker_ids[0] < 50:
            current_time = time.time()
            self.flag_tello_aruco = True
            self.last_marker_detection_time = time.time()
            
            for idx, pose in enumerate(msg.poses):
                marker_id = f"marker_{msg.marker_ids[idx]}"
                
                local_x = pose.position.x
                local_y = pose.position.y
                local_z = pose.position.z
                
                global_x = self.tello_x + local_x
                global_y = self.tello_y + local_y
                global_z = self.tello_z + local_y
                
                self.tello_marker_positions[marker_id] = {
                    "x": global_x,
                    "y": global_y,
                    "z": global_z,
                    "timestamp": current_time
                }
                self.save_marker_positions(marker_id)

    def save_marker_positions(self, marker_id):
        if marker_id in self.pc_marker_positions and marker_id in self.tello_marker_positions:
            pc_data = self.pc_marker_positions[marker_id]
            tello_data = self.tello_marker_positions[marker_id]
            marker_file_path = os.path.join(self.data_dir, f"{marker_id}_positions.csv")
            file_exists = os.path.exists(marker_file_path)
            
            error_x = pc_data['x'] - tello_data['x']
            error_y = pc_data['y'] - tello_data['y']
            error_z = pc_data['z'] - tello_data['z']
            euclidean_error = math.sqrt(error_x**2 + error_y**2 + error_z**2)
            
            with open(marker_file_path, 'a') as f:
                if not file_exists:
                    f.write("timestamp,pc_x,pc_y,pc_z,tello_x,tello_y,tello_z,error_x,error_y,error_z,euclidean_error\n")
                timestamp = time.time()
                f.write(f"{timestamp:.3f},{pc_data['x']:.4f},{pc_data['y']:.4f},{pc_data['z']:.4f},")
                f.write(f"{tello_data['x']:.4f},{tello_data['y']:.4f},{tello_data['z']:.4f},")
                f.write(f"{error_x:.4f},{error_y:.4f},{error_z:.4f},{euclidean_error:.4f}\n")
            
            if marker_id not in self.marker_errors:
                self.marker_errors[marker_id] = []
            
            self.marker_errors[marker_id].append({
                'timestamp': timestamp,
                'error_x': error_x,
                'error_y': error_y,
                'error_z': error_z,
                'euclidean_error': euclidean_error
            })

    def data_logging_loop(self):
        if self.drone_pose is not None:
            drone_log_path = os.path.join(self.data_dir, "drone_trajectory.csv")
            file_exists = os.path.exists(drone_log_path)
            
            with open(drone_log_path, 'a') as f:
                if not file_exists:
                    f.write("timestamp,x,y,z,yaw\n")
                timestamp = time.time()
                f.write(f"{timestamp:.3f},{self.tello_x:.4f},{self.tello_y:.4f},{self.tello_z:.4f},{self.tello_yaw:.4f}\n")

    def generate_trajectory_plots(self):
        if len(self.drone_trajectory) < 2:
            return
        
        timestamps = [entry[0] - self.drone_trajectory[0][0] for entry in self.drone_trajectory]
        x_values = [entry[1] for entry in self.drone_trajectory]
        y_values = [entry[2] for entry in self.drone_trajectory]
        z_values = [entry[3] for entry in self.drone_trajectory]
        
        plt.figure(figsize=(10, 6))
        plt.plot(y_values, x_values, 'r-', label='XY Position')
        plt.title('Drone X-Y Position Over Time')
        plt.xlabel('Y (meters)')
        plt.ylabel('X (meters)')
        plt.legend()
        plt.grid(True)
        max_limit = max(abs(max(y_values)), abs(min(y_values)), abs(max(x_values)), abs(min(x_values)))
        plt.xlim(-max_limit, max_limit)
        plt.ylim(-max_limit, max_limit)
        plt.axis('equal')
        xy_plot_path = os.path.join(self.data_dir, f"xy_trajectory_{time.strftime('%Y%m%d')}.png")
        plt.savefig(xy_plot_path)
        plt.close()
        
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, z_values, 'b-', label='Z Position')
        plt.title('Drone Altitude Over Time')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Altitude (meters)')
        plt.legend()
        plt.grid(True)
        z_plot_path = os.path.join(self.data_dir, f"z_trajectory_{time.strftime('%Y%m%d')}.png")
        plt.savefig(z_plot_path)
        plt.close()
        
        #self.get_logger().info(f"Graphs saved in {self.data_dir}")

    def generate_error_plots(self):
        for marker_id, errors in self.marker_errors.items():
            if len(errors) < 2:
                continue
                
            timestamps = [entry['timestamp'] - errors[0]['timestamp'] for entry in errors]
            error_x = [entry['error_x'] for entry in errors]
            error_y = [entry['error_y'] for entry in errors]
            error_z = [entry['error_z'] for entry in errors]
            euclidean_errors = [entry['euclidean_error'] for entry in errors]
            
            plt.figure(figsize=(12, 8))
            plt.subplot(2, 1, 1)
            plt.plot(timestamps, error_x, 'r-', label='X Error')
            plt.plot(timestamps, error_y, 'g-', label='Y Error')
            plt.plot(timestamps, error_z, 'b-', label='Z Error')
            plt.title(f'Coordinate Errors for {marker_id} Over Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Error (meters)')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(timestamps, euclidean_errors, 'k-', label='Euclidean Error')
            plt.title(f'Euclidean Error for {marker_id} Over Time')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Error (meters)')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            error_plot_path = os.path.join(self.data_dir, f"{marker_id}_errors_{time.strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(error_plot_path)
            plt.close()
            
            # self.get_logger().info(f"Error graphs saved for {marker_id} at {error_plot_path}")

    def camera_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

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
            if self.flag_target == 0 or self.flag_target == 1:
                self.get_logger().info(f"Target pos: {self.target_x:.2f}, {self.target_y:.2f}, {self.target_z:.2f}")
                self.get_logger().info(f"Marker id: {marker_id}")
                self.flag_target = 2
    
        error_x = self.target_x - self.tello_x
        error_y = self.target_y - self.tello_y
        error_z = self.target_z - self.tello_z - 0.1
        distance = self.dist_from_tello({"x": self.target_x, "y": self.target_y, "z": self.target_z})
        error_yaw = (math.atan2(error_y, error_x) - self.tello_yaw + math.pi) % (2 * math.pi) - math.pi

        if error_yaw < self.yaw_tolerance and distance < 0.2 and self.flag_tello_aruco:
            if self.image is None: self.get_logger().info(f"Video not received")
            else:
                try:
                    self.save_image(marker_id)
                    self.seen_markers.add(marker_id)
                    self.get_logger().info(f"Marker processed: {marker_id}")
                    self.pose_buffer.pop(marker_id, None)
                except: pass

        cmd = Twist()
        if self.last_vicon_time > time.time() - 0.5:
            cmd.linear.x = 0.0
            if abs(error_yaw) >= self.yaw_tolerance: cmd.linear.y = 0.0
            else: cmd.linear.y = self.linear_speed * np.clip(abs(distance), 0.0, 1.0)
            cmd.linear.z = self.linear_speed * np.clip(error_z, -1.0, 1.0)
            cmd.angular.z = - self.angular_speed * np.clip(error_yaw, -1.0, 1.0)

            #self.get_logger().info(f'X: {error_x:.1f}, Y: {error_y:.1f}, Z: {error_z:.1f}, YAW: {error_yaw:.1f}.')
            #self.get_logger().info(f'DIST: {distance:.2f}, YAW err: {error_yaw:.2f}')
        else: 
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.linear.z = 1.0
            cmd.angular.z = 1.0
            #self.get_logger().info(f'Vicon error.')
        
        self.cmd_pub.publish(cmd)
        
    def dist_from_tello(self, msg):
        return math.sqrt((msg["x"] - self.tello_x) ** 2 +
                         (msg["y"] - self.tello_y) ** 2 +
                         (msg["z"] - self.tello_z) ** 2)

    def save_image(self, marker_id):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}_marker_{marker_id}.jpg"
        full_path = os.path.join(self.data_dir, filename)
        cv2.imwrite(full_path, self.image)
        self.get_logger().info(f"Img saved: {full_path}")

def main(args=None):
    rclpy.init(args=args)
    node = TelloController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
