#!/bin/bash
xhost +
IMAGE=tello

docker run --rm -it\
 --privileged \
 --ipc host \
 --net host \
 -e DISPLAY=$DISPLAY \
 -v /tmp/.X11-unix/:/tmp/.X11-unix \
 -v ~/.Xauthority:/root/.Xauthority \
 -e XAUTHORITY=/root/.Xauthority \
 -v /dev/video0:/dev/video0 \
 -v ~/Tello-detect-marker-Aruco:/root/tello_MD/wrk_src\
 --name tello \
 $IMAGE bash