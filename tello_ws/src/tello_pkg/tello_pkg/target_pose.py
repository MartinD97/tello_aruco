import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Twist, PoseStamped, Point
from sensor_msgs.msg import Image
import math
import time
import cv2
from cv_bridge import CvBridge
import numpy as np
import tf_transformations

class TelloMarkerFollower(Node):
    def __init__(self):
        super().__init__('tello_marker_follower')

        # Parametri configurabili
        self.declare_parameter('target_distance', 1.0)  # Distanza desiderata dal marker
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.3)
        self.declare_parameter('pose_timeout', 3.0)  # Timeout per considerare un marker "non più valido"

        self.target_distance = self.get_parameter('target_distance').value
        self.linear_speed = self.get_parameter('linear_speed').value
        self.angular_speed = self.get_parameter('angular_speed').value
        self.pose_timeout = self.get_parameter('pose_timeout').value

        self.offset_camera = self.load_transform_config('/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/offset_camera.yaml')
        self.pc_rotation_matrix = tf_transformations.quaternion_matrix(self.offset_camera['rotation'])[:3, :3]

        self.bridge = CvBridge()
        self.image = None
        self.pose_buffer = {}  # Salviamo i marker con timestamp
        self.drone_pose = None  # Posizione globale del drone

        # Sottoscrizione ai topic
        self.create_subscription(PoseArray, '/pc/aruco_poses', self.aruco_pose_callback, 10)  # Marker in coordinate globali
        self.create_subscription(PoseStamped, '/vicon/Tello_42/Tello_42', self.vicon_callback, 10)  # Posizione globale del drone
        self.create_subscription(Image, '/tello/image_raw/Image', self.camera_callback, 10)  # Visione della telecamera del drone

        # Publisher per il controllo del drone
        self.cmd_vel_publisher = self.create_publisher(Twist, 'control', 10) #cmd_vel ??
        
        # Timer per navigare verso i marker
        self.create_timer(0.1, self.navigate_to_marker)

    def aruco_pose_callback(self, msg: PoseArray):
        current_time = time.time()
        for idx, pose in enumerate(msg.poses):
            global_position = self.transform_marker_to_global(pose)
            marker_id = f"marker_{idx}"
            self.pose_buffer[marker_id] = {
                "x": global_position[0],
                "y": global_position[1],
                "z": global_position[2]
            }

        # Rimuove marker scaduti
        self.get_logger().info(f"Marker {marker_id} in coordinate globali: {global_position}")
        self.pose_buffer = {k: v for k, v in self.pose_buffer.items() if current_time - v["timestamp"] < self.pose_timeout}

    def transform_marker_to_global(self, marker_pose):
        marker_position = np.array([marker_pose.position.x, marker_pose.position.y, marker_pose.position.z])
        global_position = self.pc_rotation_matrix @ marker_position + self.offset_camera['translation']
        return global_position

    def vicon_callback(self, msg: PoseStamped):
        self.drone_pose = {
            "x": msg.pose.position.x,
            "y": msg.pose.position.y,
            "z": msg.pose.position.z
        }

    def camera_callback(self, msg: Image):
        """Riceve l'immagine dalla telecamera del drone."""
        self.image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

    def navigate_to_marker(self):
        """Naviga verso il marker attivo e si centra su di esso quando lo vede con la telecamera."""
        if not self.drone_pose or not self.pose_buffer:
            return  # Se il drone o i marker non sono disponibili, non fare nulla

        # Seleziona il marker più vicino
        closest_marker = min(self.pose_buffer.items(), key=lambda m: self.distance_to_marker(m[1]))
        marker_id, marker_pose = closest_marker
        x, y, z = marker_pose["x"], marker_pose["y"], marker_pose["z"]

        # Calcola la distanza tra il drone e il marker
        drone_x, drone_y, drone_z = self.drone_pose["x"], self.drone_pose["y"], self.drone_pose["z"]
        distance = math.sqrt((x - drone_x)**2 + (y - drone_y)**2 + (z - drone_z)**2)
        yaw_angle = math.atan2(y - drone_y, x - drone_x)

        # Comando di velocità per avvicinarsi
        twist = Twist()
        if distance > self.target_distance:
            twist.linear.x = min(self.linear_speed, self.linear_speed * (distance - self.target_distance))
            twist.angular.z = min(self.angular_speed, self.angular_speed * yaw_angle)
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

            # Se il drone vede il marker con la telecamera, scatta la foto e si ferma
            if self.image is not None:
                self.save_image(marker_id)
                self.pose_buffer.pop(marker_id, None)  # Rimuove il marker raggiunto

        self.cmd_vel_publisher.publish(twist)

    def distance_to_marker(self, marker_pose):
        """Calcola la distanza tra il drone e un marker dato."""
        if not self.drone_pose:
            return float('inf')
        return math.sqrt(
            (marker_pose["x"] - self.drone_pose["x"])**2 +
            (marker_pose["y"] - self.drone_pose["y"])**2 +
            (marker_pose["z"] - self.drone_pose["z"])**2
        )

    def save_image(self, marker_id):
        """Salva l'immagine quando il drone è centrato sul marker."""
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
