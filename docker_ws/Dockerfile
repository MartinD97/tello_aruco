FROM osrf/ros:humble-desktop

RUN apt-get -y update && apt-get install -y git
RUN apt-get -y upgrade
WORKDIR /root/tello_MD

## install.sh
RUN apt-get -y update && apt-get -y upgrade
RUN apt-get install -y build-essential gdb
RUN apt install -y locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN apt-get -y update && apt-get -y upgrade
RUN apt install -y curl gnupg2 lsb-release
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN apt install -y python3-pip
RUN pip3 install -U argcomplete
RUN apt install -y python3-colcon-common-extensions
RUN apt install -y python3-rosdep
RUN rosdep update
RUN rosdep fix-permissions
RUN echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc
RUN echo "export _colcon_cd_root=~/ros2_install" >> ~/.bashrc
RUN echo "source ~/.bashrc"
RUN pip3 install catkin_pkg rospkg av image djitellopy2 pyyaml
RUN apt install -y python3-tf*
RUN apt install -y ros-humble-ament-cmake* ros-humble-tf2* ros-humble-rclcpp* ros-humble-rosgraph*
RUN apt install -y ros-humble-rviz* ros-humble-rqt*

#for target
RUN pip3 install transforms3d
RUN apt install -y ros-humble-tf-transformations
RUN apt install nano

#phone camera
RUN apt update
RUN apt install -y ros-humble-image-transport ros-humble-cv-bridge ros-humble-sensor-msgs ros-humble-rqt-image-view
RUN pip install requests scipy
RUN sudo apt install -y adb
RUN sudo apt install -y scrcpy
RUN sudo apt-get install -y cheese

#camera tello calibration
RUN apt install -y ros-humble-camera-calibration ros-humble-camera-calibration-parsers ros-humble-camera-info-manager ros-humble-launch-testing-ament-cmake

RUN pip install opencv-python
RUN pip install opencv-contrib-python

WORKDIR /root/tello_MD/wrk_src