#!/bin/bash

colcon build --packages-ignore pangolin g2o ORB_SLAM2
export LD_LIBRARY_PATH=/root/tello_MD/wrk_src/tello-ros2/libs/ORB_SLAM2/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

source install/setup.bash