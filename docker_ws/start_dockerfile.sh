#!/bin/bash
IMAGE=tello

docker run --rm -it\
 --privileged \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix/:/tmp/.X11-unix \
 -v ~/.Xauthority:/root/.Xauthority \
 -e XAUTHORITY=/root/.Xauthority \
 -v /dev/video0:/dev/video0 \
 -v ~/docker_tello_ros2/tello_ws:/root/tello_ws/\
 --name tello \
 $IMAGE bash
