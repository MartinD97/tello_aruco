#! /usr/bin/env python3

import numpy as np
import cv2
from cv2 import aruco
import pickle
import yaml
import glob


# ChAruco board variables
CHARUCOBOARD_ROWCOUNT = 7
CHARUCOBOARD_COLCOUNT = 9
ARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Create constants to be passed into OpenCV and Aruco methods
CHARUCO_BOARD = cv2.aruco.CharucoBoard((
        CHARUCOBOARD_COLCOUNT,
        CHARUCOBOARD_ROWCOUNT),
        squareLength=0.015,
        markerLength=0.011,
        dictionary=ARUCO_DICT)

# Create the arrays and variables we'll use to store info like corners and IDs from images processed
corners_all = [] # Corners discovered in all images processed
ids_all = [] # Aruco ids corresponding to corners discovered
object_points = [] #punti 3D
image_points = [] #punti 2D


# This requires a set of images or a video taken with the camera you want to calibrate
# I'm using a set of images taken with the camera with the naming convention:
# 'camera-pic-of-charucoboard-<NUMBER>.jpg'
# All images used should be the same size, which if taken with the same camera shouldn't be a problem
images = glob.glob('*.jpg')

for iname in images:
    img = cv2.imread(iname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find aruco markers in the query image
    aruco_detector = cv2.aruco.ArucoDetector(dictionary=ARUCO_DICT)
    corners, ids, _ = aruco_detector.detectMarkers(gray)

    # Outline the aruco markers found in our query image
    img = cv2.aruco.drawDetectedMarkers(
            image=img, 
            corners=corners)

    # Get charuco corners and ids from detected aruco markers
    charucodetector = cv2.aruco.CharucoDetector(CHARUCO_BOARD)
    charuco_corners, charuco_ids, marker_corners, marker_ids = charucodetector.detectBoard(gray)

    # Draw the Charuco board we've detected to show our calibrator the board was properly detected
    corners_all.append(charuco_corners)
    ids_all.append(charuco_ids)
    charuco_object_points = CHARUCO_BOARD.getChessboardCorners()
    object_points.append(charuco_object_points)

    # Draw the Charuco board we've detected to show our calibrator the board was properly detected
    img = cv2.aruco.drawDetectedCornersCharuco(
        image=img,
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids)

    # Reproportion the image, maxing width or height at 1000
    proportion = max(img.shape) / 1000.0
    img = cv2.resize(img, (int(img.shape[1]/proportion), int(img.shape[0]/proportion)))
    cv2.imshow('Charuco board', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
if len(images) < 1:
    print("Calibration was unsuccessful. No images of charucoboards were found. Add images of charucoboards and use or alter the naming conventions used in this file.")
    exit()

object_points = np.array(object_points)
corners_all = np.array(corners_all)
ret, cameraMatrix, distCoeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        corners_all,
        gray.shape[::-1],
        None,
        None)
    
# Print matrix and distortion coefficient to the console
print(cameraMatrix)
print(distCoeffs)
    
# Save values to be used where matrix+dist is required, for instance for posture estimation
# I save files in a pickle or yaml


# f = open('calibration.pckl', 'wb')
# pickle.dump((cameraMatrix, distCoeffs, rvecs, tvecs), f)
# f.close()

calibration_data = {
    'camera_matrix': cameraMatrix.tolist(),
    'dist_coeffs': distCoeffs.tolist(),
    'rotation_vectors': [vec.tolist() for vec in rvecs],
    'translation_vectors': [vec.tolist() for vec in tvecs]
}

with open('calibration_pc.yaml', 'w') as f:
    yaml.dump(calibration_data, f, default_flow_style=False)
    
# Print to console our success
print('Calibration successful. Calibration file used: {}'.format('calibration.pckl'))