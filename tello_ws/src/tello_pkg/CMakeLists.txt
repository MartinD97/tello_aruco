cmake_minimum_required(VERSION 3.5)
project(tello_pkg)

find_package(ament_cmake REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(
  PROGRAMS
  scripts/video_publisher.py
  scripts/detect_marker.py
  scripts/camera.py
  scripts/camera_usb.py
  scripts/aruco_node.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()