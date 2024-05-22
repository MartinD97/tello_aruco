FROM osrf/ros:foxy-desktop

RUN apt-get -y update && apt-get install -y git
RUN apt-get -y upgrade
WORKDIR /root/tello_ws
RUN git clone https://github.com/tentone/tello-ros2.git

## install.sh
RUN apt install -y build-essential gdb
RUN apt-get -y update && apt-get -y upgrade
RUN apt install -y locales
RUN locale-gen en_US en_US.UTF-8
RUN update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG=en_US.UTF-8
RUN apt-get -y update && apt-get -y upgrade
RUN apt install -y curl gnupg2 lsb-release
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
RUN sh -c 'echo "deb [arch=$(dpkg --print-architecture)] http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list'
RUN echo "source /opt/ros/foxy/setup.bash" >> ~/.bashrc
RUN apt install -y python3-pip
RUN pip3 install -U argcomplete
RUN apt install -y python3-colcon-common-extensions
RUN apt install -y python3-rosdep
RUN rosdep update
RUN rosdep fix-permissions
RUN echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc
RUN echo "export _colcon_cd_root=~/ros2_install" >> ~/.bashrc
RUN echo "source ~/.bashrc"
RUN pip3 install catkin_pkg rospkg av image opencv-python djitellopy2 pyyaml
RUN apt install -y python3-tf*
RUN apt install -y ros-foxy-ament-cmake* ros-foxy-tf2* ros-foxy-rclcpp* ros-foxy-rosgraph*
RUN apt install -y ros-foxy-rviz* ros-foxy-rqt*

#opencv.sh
RUN mkdir -p libs
RUN cd libs
RUN apt install -y cmake gcc g++ python3-dev python3-numpy libavcodec-dev libavformat-dev libswscale-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev
RUN cd ~
RUN apt install -y git
RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv
#RUN git fetch
#RUN git checkout 3.4
RUN cmake
#RUN mkdir build
#RUN cd build
#RUN make install -j8

#cameracalib.sh
RUN apt install -y ros-foxy-camera-calibration ros-foxy-camera-calibration-parsers ros-foxy-camera-info-manager ros-foxy-launch-testing-ament-cmake

#extra
RUN sudo usermod -aG video root
RUN sudo apt install -y cheese