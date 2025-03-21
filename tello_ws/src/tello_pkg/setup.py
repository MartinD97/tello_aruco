from setuptools import setup

package_name = 'tello_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='A package to publish and detect aruco markers from video frames',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'video_frame_pub = tello_pkg.video_frame_pub:main',
            'detect_marker = tello_pkg.detect_marker:main',
            'camera = tello_pkg.camera:main',
            'camera_wifi = tello_pkg.camera_wifi:main',
            'aruco_node = tello_pkg.aruco_node:main',
            'aruco_node_pc = tello_pkg.aruco_node_pc:main',
            'aruco_node_tello = tello_pkg.aruco_node_tello:main',
            'keep_alive = tello_pkg.keep_alive:main',
            'camera_test = tello_pkg.camera_test:main',
            'target_pose = tello_pkg.target_pose:main',
            'convert_pckl_yaml = tello_pkg.convert_pckl_yaml:main',
            'camera_for_calib = tello_pkg.camera_for_calib:main'
        ],
    },
)
