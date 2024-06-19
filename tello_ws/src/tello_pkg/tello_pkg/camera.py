import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import requests

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.bridge = CvBridge()
        self.video_url = 'http://192.168.33.78:4747'  # Modifica con l'URL corretto

    def timer_callback(self):
        try:
            resp = requests.get(self.video_url, stream=True)
            if resp.status_code == 200:
                arr = np.asarray(bytearray(resp.raw.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)
                if image is not None:
                    ros_image = self.bridge.cv2_to_imgmsg(image, 'bgr8')
                    self.publisher_.publish(ros_image)
            else:
                self.get_logger().warn('Failed to get image from camera.')
        except Exception as e:
            self.get_logger().error(f'Error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
