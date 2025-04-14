import rclpy
import cv2
import numpy as np
import pickle
import tf2_ros
import math
import yaml
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf_transformations import quaternion_from_matrix

class PcNode(Node):
    def __init__(self):
        super().__init__('pc_node')
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.offset_camera = self.load_transform_config('/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/config/offset_camera.yaml')
        self.publisher_image = self.create_publisher(Image, 'pc/image_raw', 10)
        self.pub_camera_info = self.create_publisher(CameraInfo, 'pc/camera_info', 10)
        self.publisher_pose = self.create_publisher(PoseStamped, 'pc/pose', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.bridge = CvBridge()
        self.create_pc_frame()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open pc camera')
        self.camera_info = self.load_calibration_file()

    def load_transform_config(self, file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

    def load_calibration_file(self):
        try:
            calibration_data = self.load_transform_config('/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/config/calibration_pc.yaml')
            self.get_logger().info(f"Calibration data loaded from calibration_pc.yaml")
            camera_matrix = np.array(calibration_data['camera_matrix']['data']).reshape((3, 3))
            dist_coeffs = np.array(calibration_data['distortion_coefficients']['data']).flatten()
            projection_matrix = np.zeros((3, 4))
            projection_matrix[:3, :3] = camera_matrix

            return {
                'distortion_coefficients': dist_coeffs.flatten().tolist(),
                'camera_matrix': camera_matrix.flatten().tolist(),
                'rectification_matrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],  # Matrice identità 3x3
                'projection_matrix': projection_matrix.flatten().tolist()
            }

        except Exception as e:
            self.get_logger().error(f"Errore durante il caricamento del file di calibrazione: {e}")
            return None


    def create_pc_frame(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'pc_frame'
        t.transform.translation.x = self.offset_camera['translation']['x']
        t.transform.translation.y = self.offset_camera['translation']['y']
        t.transform.translation.z = self.offset_camera['translation']['z']
        
        t.transform.rotation.x = self.offset_camera['rotation']['x']
        t.transform.rotation.y = self.offset_camera['rotation']['y']
        t.transform.rotation.z = self.offset_camera['rotation']['z']
        t.transform.rotation.w = self.offset_camera['rotation']['w']
        self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        if len(self.camera_info)  > 0:
            camera_info_msg = CameraInfo()
            camera_info_msg.height = 720
            camera_info_msg.width = 1280
            camera_info_msg.distortion_model = 'plumb_bob'
            camera_info_msg.d = self.camera_info['distortion_coefficients']
            camera_info_msg.k = self.camera_info['camera_matrix']
            camera_info_msg.r = self.camera_info['rectification_matrix']
            camera_info_msg.p = self.camera_info['projection_matrix']
            camera_info_msg.header.frame_id = 'pc_frame'
            camera_info_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_camera_info.publish(camera_info_msg)

        ret, frame = self.cap.read()
        if ret:
            ros_image = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            ros_image.header.frame_id = 'pc_frame'
            ros_image.header.stamp = self.get_clock().now().to_msg()
            self.publisher_image.publish(ros_image)
        else:
            self.get_logger().warn('Failed to capture image from pc camera.')

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = PcNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
