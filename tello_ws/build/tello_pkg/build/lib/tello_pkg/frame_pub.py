import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class VideoPublisher(Node):
    def __init__(self):
        super().__init__('frame_pub')
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.get_logger().info('frame_pub started')

        #video_path = os.path.join(
        #    os.path.dirname(__file__), 
        #    '..', 
        #    'videos', 
        #    'detect_phone.mp4'
        #)
        video_path = '/root/docker_tello_ros2/tello_ws/src/tello_pkg/videos/detect_phone.mp4'
        self.get_logger().info(video_path)

        self.cap = cv2.VideoCapture(video_path)
        self.bridge = CvBridge()

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            msg = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    video_publisher = VideoPublisher()
    rclpy.spin(video_publisher)
    video_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
