#! /usr/bin/env python3

import numpy as np
import cv2
from cv2 import aruco
import glob

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
corners_all = []
ids_all = []
images = glob.glob('*.jpg')
for iname in images:
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    aruco_params = aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(dictionary=ARUCO_DICT)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    if len(corners) > 0: 
        ids = ids.flatten()
        for (marker_corner, marker_id) in zip(corners, ids):
            corners_all.append(marker_corner)
            ids_all.append(marker_id)
            marker_corner = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = marker_corner
            
            # Convert the (x,y) coordinate pairs to integers
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            top_left = (int(top_left[0]), int(top_left[1]))
            
            # Draw the bounding box of the ArUco detection
            cv2.line(img, top_left, top_right, (0, 255, 0), 2)
            cv2.line(img, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(img, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(img, bottom_left, top_left, (0, 255, 0), 2)
            
            # Calculate and draw the center of the ArUco marker
            center_x = int((top_left[0] + bottom_right[0]) / 2.0)
            center_y = int((top_left[1] + bottom_right[1]) / 2.0)
            cv2.circle(img, (center_x, center_y), 4, (0, 0, 255), -1)
            
            # Draw the ArUco marker ID on the image
            cv2.putText(img, str(marker_id), (top_left[0], top_left[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    cv2.imshow('frame', img)
    cv2.waitKey(0)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break

cv2.destroyAllWindows()