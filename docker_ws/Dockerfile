FROM osrf/ros:foxy-desktop

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

#for target
RUN pip3 install opencv-contrib-python transforms3d
RUN apt install -y ros-foxy-tf-transformations
RUN apt install nano

#phone camera
RUN apt update
RUN apt install -y ros-foxy-image-transport ros-foxy-cv-bridge ros-foxy-sensor-msgs ros-foxy-rqt-image-view
RUN pip install requests
RUN sudo apt install -y adb
RUN sudo apt install -y scrcpy
RUN sudo apt-get install -y cheese

#camera tello calibration
RUN apt install -y ros-foxy-camera-calibration ros-foxy-camera-calibration-parsers ros-foxy-camera-info-manager ros-foxy-launch-testing-ament-cmake

#opencv.sh
WORKDIR /root/tello_MD/libs
RUN apt install -y cmake gcc g++ python3-dev python3-numpy libavcodec-dev libavformat-dev libswscale-dev libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libgtk-3-dev libpng-dev libjpeg-dev libopenexr-dev libtiff-dev libwebp-dev
RUN git clone https://github.com/opencv/opencv.git
RUN cd opencv && git checkout 3.4
RUN cd opencv && mkdir build && cd build && cmake .. -DWITH_QT=OFF -DWITH_OPENGL=ON -DBUILD_TIFF=ON -DBUILD_opencv_java=OFF -DWITH_CUDA=OFF -DWITH_GTK=ON -DWITH_V4L=ON -DWITH_FFMPEG=ON -DWITH_XINE=ON -DBUILD_EXAMPLES=OFF -DCMAKE_BUILD_TYPE=RELEASE -DCMAKE_INSTALL_PREFIX=/usr/local
RUN cd opencv/build && make && make install

#ORB_SLAM
RUN apt install -y ros-foxy-vision-opencv ros-foxy-message-filters libeigen3-dev
RUN mkdir -p ../libs && cd ../libs

RUN apt install -y libgl1-mesa-dev libglew-dev cmake libegl1-mesa-dev libwayland-dev libxkbcommon-dev wayland-protocols ffmpeg libavcodec-dev libavutil-dev libavformat-dev libswscale-dev libavdevice-dev libjpeg-dev libpng-dev libtiff5-dev libopenexr-dev
RUN git clone https://github.com/stevenlovegrove/Pangolin.git
RUN cd Pangolin && mkdir build && cd build && cmake ..
RUN cd Pangolin/build && make && make install

RUN apt install -y cmake libeigen3-dev libsuitesparse-dev qtdeclarative5-dev qt5-qmake libqglviewer-dev-qt5
RUN git clone https://github.com/RainerKuemmerle/g2o.git
RUN cd g2o && mkdir build && cd build && cmake ..
RUN cd g2o/build && make && make install

WORKDIR /root/tello_MD/wrk_src