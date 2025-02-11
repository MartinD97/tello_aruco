#!/usr/bin/env python3

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from cv2 import aruco

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.subscription = self.create_subscription(
            Image,
            'camera',  # Assicurati che questo topic corrisponda a quello pubblicato dal nodo video_publisher
            self.listener_callback,
            10)
        self.subscription  # Prevent unused variable warning
        self.bridge = CvBridge()
        self.ARUCO_DICT = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
        self.corners_all = []
        self.ids_all = []
        self.get_logger().info('aruco_detect started')

    def listener_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        aruco_params = aruco.DetectorParameters()
        aruco_detector = cv2.aruco.ArucoDetector(dictionary=self.ARUCO_DICT)
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        
        if len(corners) > 0:
            ids = ids.flatten()
            for (marker_corner, marker_id) in zip(corners, ids):
                self.corners_all.append(marker_corner)
                self.ids_all.append(marker_id)
                marker_corner = marker_corner.reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = marker_corner

                # Convert the (x,y) coordinate pairs to integers
                top_right = (int(top_right[0]), int(top_right[1]))
                bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
                bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
                top_left = (int(top_left[0]), int(top_left[1]))

                # Draw the bounding box of the ArUco detection
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # Calculate and draw the center of the ArUco marker
                center_x = int((top_left[0] + bottom_right[0]) / 2.0)
                center_y = int((top_left[1] + bottom_right[1]) / 2.0)
                cv2.circle(frame, (center_x, center_y), 4, (0, 0, 255), -1)

                # Draw the ArUco marker ID on the image
                cv2.putText(frame, str(marker_id), (top_left[0], top_left[1] - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.namedWindow('Aruco Detector', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Aruco Detector', 1270, 720)
        cv2.imshow('Aruco Detector', frame)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    aruco_detector = ArucoDetector()
    rclpy.spin(aruco_detector)
    aruco_detector.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
