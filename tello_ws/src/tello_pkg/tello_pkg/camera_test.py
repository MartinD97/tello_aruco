import rclpy
import cv2
import numpy as np
import pickle
import tf2_ros
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped, Point

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.publisher_image = self.create_publisher(Image, 'camera', 10)
        self.pub_camera_info = self.create_publisher(CameraInfo, 'camera_info', 10)
        self.publisher_pose = self.create_publisher(PoseStamped, 'pose', 10)
        self.camera_pose = self.create_subscription(Point, 'camera_position', self.pointmsg, 10)
        self.bridge = CvBridge()
        self.position = [0.0, 0.0, 0.0]
        self.create_map_frame()
        self.create_camera_frame()
        self.timer = self.create_timer(0.1, self.timer_callback)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error('Cannot open camera')
            rclpy.shutdown()
            return

        self.calibration_file = "/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/calibration.pckl"
        self.camera_info = self.load_calibration_file()
        if not self.camera_info:
            self.get_logger().warn("No camera info.")
            return

    def pointmsg(self, msg):
        self.position = [msg.x, msg.y, msg.z]
        self.get_logger().info(f"Received: x={self.position[0]:.3f}, y={self.position[1]:.3f}, z={self.position[2]:.3f}")

    def create_map_frame(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'
        t.child_frame_id = 'map'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def load_calibration_file(self):
        try:
            with open(self.calibration_file, 'rb') as f:
                calibration_data = pickle.load(f)
                self.get_logger().info(f"Calibration data loaded from {self.calibration_file}")
                camera_matrix = calibration_data[0]
                projection_matrix = np.zeros((3, 4))
                projection_matrix[:3, :3] = camera_matrix
                return {
                    'distortion_coefficients': calibration_data[1].flatten().tolist(),
                    'camera_matrix': calibration_data[0].flatten().tolist(),
                    'rectification_matrix': [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], 
                    'projection_matrix': projection_matrix.flatten().tolist() 
                }
        except Exception as e:
            self.get_logger().error(f"Errore durante il caricamento del file di calibrazione: {e}")
            return None

    def create_camera_frame(self):
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_frame'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

    def timer_callback(self):
        if self.camera_info and len(self.camera_info)  > 0:
            camera_info_msg = CameraInfo()
            camera_info_msg.height = 720
            camera_info_msg.width = 1280
            camera_info_msg.distortion_model = 'plumb_bob'
            camera_info_msg.d = self.camera_info['distortion_coefficients']
            camera_info_msg.k = self.camera_info['camera_matrix']
            camera_info_msg.r = self.camera_info['rectification_matrix']
            camera_info_msg.p = self.camera_info['projection_matrix']
            camera_info_msg.header.frame_id = 'camera_frame'
            camera_info_msg.header.stamp = self.get_clock().now().to_msg()
            self.pub_camera_info.publish(camera_info_msg)

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'map'
        t.child_frame_id = 'camera_frame'
        t.transform.translation.x = self.position[0]
        t.transform.translation.y = self.position[1]
        t.transform.translation.z = self.position[2]
        t.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(t)

        ret, frame = self.cap.read()
        if ret:
            ros_image = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            ros_image.header.frame_id = 'camera_frame'
            ros_image.header.stamp = self.get_clock().now().to_msg()
            self.publisher_image.publish(ros_image)
        else:
            self.get_logger().warn('Failed to capture image from camera.')

    def destroy_node(self):
        super().destroy_node()
        self.cap.release()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
