import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.bridge = CvBridge()
        
        self.cap = None
        for i in range(5):  # Prova ad aprire le prime 5 fotocamere
            self.cap = cv2.VideoCapture(i)
            if self.cap.isOpened():
                self.get_logger().info(f'Successfully opened camera with index {i}')
                break
            else:
                self.get_logger().warn(f'Failed to open camera with index {i}')
        
        if self.cap is None or not self.cap.isOpened():
            self.get_logger().error('Failed to open any camera.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if ret:
            ros_image = self.bridge.cv2_to_imgmsg(frame, 'bgr8')
            self.publisher_.publish(ros_image)
            self.get_logger().info('Image captured and published.')

            # Visualizza l'immagine
            cv2.imshow('Camera View', frame)
            cv2.waitKey(1)  # Attende 1 ms per permettere l'aggiornamento della finestra
        else:
            self.get_logger().warn('Failed to capture image from camera.')

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
