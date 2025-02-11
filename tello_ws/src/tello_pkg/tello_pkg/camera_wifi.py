import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import time
import tf2_ros
import numpy as np
import requests
from geometry_msgs.msg import TransformStamped, Point
from tf_transformations import quaternion_from_matrix

class CameraNode(Node):
    def __init__(self, ip_address):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'camera', 1)
        self.xyz_orbslam = self.create_subscription(Point, 'camera_position', self.pointmsg, 1)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)
        self.bridge = CvBridge()
        self.cap = cv2.VideoCapture(f"http://{ip_address}:8080/video")
        self.sensor_url = f"http://{ip_address}:8080/sensors.json"
        #self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.cap.isOpened(): self.get_logger().info(f'Successfully opened camera at {ip_address}')
        else: self.get_logger().warn(f'Failed to open camera at {ip_address}')
        self.position = [0.0, 0.0, 0.0]
        self.orientation = [0.0, 0.0, 0.0]
        self.timer = self.create_timer(0.1, self.timer_callback)

    def pointmsg(self, msg):
        # self.position = [-msg.z, msg.x, msg.y]
        self.position = [msg.x, msg.y, msg.z]
        self.get_logger().info(f"Pos: x={self.position[0]:.3f}, y={self.position[1]:.3f}, z={self.position[2]:.3f}")

    def get_orientation_from_sensors(self):
        try:
            response = requests.get(self.sensor_url, timeout=0.5)
            if response.status_code == 200:
                data = response.json()
                rot_vector = data.get("rot_vector", {}).get("data", [])
                if len(rot_vector) > 0 and len(rot_vector[0]) > 1:
                    quaternion = rot_vector[0][1] 
                if len(quaternion) >= 4:
                    return self.quaternion_to_matrix([
                        float(quaternion[0]),  # x*sin(θ/2)
                        float(quaternion[1]),  # y*sin(θ/2)
                        float(quaternion[2]),  # z*sin(θ/2)
                        float(quaternion[3])   # cos(θ/2)
                    ])
        except Exception as e:
            self.get_logger().warn(f"Failed to fetch sensor data: {e}")
        return [0.0, 0.0, 0.0, 1.0]

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            #resized_frame = cv2.resize(frame, (1280, 720))
            ros_image = self.bridge.cv2_to_imgmsg(np.array(frame), 'bgr8') # MOOODD!!
            ros_image.header.frame_id = 'tello_frame'
            self.publisher_.publish(ros_image)
            #cv2.imshow('Camera View', resized_frame)
            
            angle = -np.radians(90)
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(angle), -np.sin(angle)],
                [0, np.sin(angle), np.cos(angle)]
            ])
            sensor_quaternion = self.get_orientation_from_sensors()
            #combined_rotation_matrix = np.dot(sensor_quaternion, rotation_matrix)

            # quat_mult = quaternion_from_matrix(
            #     np.block([
            #         [combined_rotation_matrix, np.zeros((3, 1))],
            #         [np.zeros((1, 3)), np.array([[1]])]
            #     ])
            # )

            t = TransformStamped()
            t.header.frame_id = "map"
            t.child_frame_id = "tello_frame"
            t.transform.translation.x = self.position[0]
            t.transform.translation.y = self.position[1]
            t.transform.translation.z = self.position[2]
            # t.transform.rotation.x = quat_mult[0]
            # t.transform.rotation.y = quat_mult[1]
            # t.transform.rotation.z = quat_mult[2]
            # t.transform.rotation.w = quat_mult[3]
            t.header.stamp = self.get_clock().now().to_msg()
            self.tf_broadcaster.sendTransform(t)
            #cv2.waitKey(1)  # wait 1 ms for update
        else:
            self.get_logger().warn('Failed to capture image from camera.')
            time.sleep(2)

    def quaternion_to_matrix(self, quat):
        x, y, z, w = quat
        r1 = [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - z * w), 2 * (x * z + y * w)]
        r2 = [2 * (x * y + z * w), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - x * w)]
        r3 = [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x ** 2 + y ** 2)]
        return np.array([r1, r2, r3])

def main(args=None):
    rclpy.init(args=args)

    if len(sys.argv) < 2:
        print("Error: Camera IP address ID is missing.")
        print("Use: ros2 run tello_pkg camera_wifi <ID>")
        return
    ip_address = sys.argv[1]
    node = CameraNode(ip_address)
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
