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
 -v ~/tello_aruco:/root/tello_MD/wrk_src\
 --name tello \
 $IMAGE bash -c "cp /root/tello_MD/wrk_src/colcon.sh /root/tello_MD && \
 bash"
 #source /root/tello_MD/wrk_src/install/setup.bash \