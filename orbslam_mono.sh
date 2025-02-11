#!/bin/bash

export LD_LIBRARY_PATH=/root/tello_MD/wrk_src/tello-ros2/libs/ORB_SLAM2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

source install/setup.bash

#ros2 run orbslam2 mono tello-ros2/libs/ORB_SLAM2/Vocabulary/ORBvoc.txt tello-ros2/slam/src/orbslam2/config.yaml # camera tello
ros2 run orbslam2 mono tello-ros2/libs/ORB_SLAM2/Vocabulary/ORBvoc.txt tello-ros2/slam/src/orbslam2/config_camera_pc.yaml # camera pc