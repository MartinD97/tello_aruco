import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import pickle
import tf_transformations
import yaml
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import PoseArray, Pose
from ros2_aruco_interfaces.msg import ArucoMarkers
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformBroadcaster

class ArucoNode(Node):
    def __init__(self):
        super().__init__("aruco_node")
        self.tf_broadcaster = TransformBroadcaster(self)
        self.bridge = CvBridge()

        # Pipeline for PC Camera
        self.pc_pipeline = self.create_pipeline(
            "pc", "/pc/image_raw", "/pc/camera_info", "/pc/aruco_poses", "/pc/aruco_markers",
            calibration_file="/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/calibration_camera_pc.pckl"
        )

        # Pipeline for Tello Camera
        self.tello_pipeline = self.create_pipeline(
            "tello", "/tello/image_raw/Image", "/tello/camera_info", "/tello/aruco_poses", "/tello/aruco_markers",
            calibration_file="/root/tello_MD/wrk_src/tello_ws/src/tello_pkg/tello_pkg/calibration.yaml"
        )

    def create_pipeline(self, prefix, image_topic, info_topic, poses_topic, markers_topic, calibration_file):
        pipeline = {}
        
        # Load calibration file
        if calibration_file.endswith(".pckl"):
            try:
                with open(calibration_file, 'rb') as f:
                    cameraMatrix, distCoeffs, _, _ = pickle.load(f)
                    pipeline["intrinsic_mat"] = cameraMatrix
                    pipeline["distortion"] = distCoeffs
                    self.get_logger().info(f"Loaded calibration for {prefix} from {calibration_file}")
            except Exception as e:
                self.get_logger().error(f"Failed to load {prefix} calibration file: {e}")
        elif calibration_file.endswith(".yaml"):
            try:
                with open(calibration_file, 'r') as f:
                    calibration_data = yaml.safe_load(f)
                    pipeline["intrinsic_mat"] = np.array(calibration_data['camera_matrix']['data']).reshape((3, 3))
                    pipeline["distortion"] = np.array(calibration_data['distortion_coefficients']['data'])
                    self.get_logger().info(f"Loaded calibration for {prefix} from {calibration_file}")
            except Exception as e:
                self.get_logger().error(f"Failed to load {prefix} calibration file: {e}")

        # Aruco settings
        pipeline["aruco_dictionary"] = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        pipeline["aruco_parameters"] = cv2.aruco.DetectorParameters() # parameters
        pipeline["aruco_parameters"].adaptiveThreshWinSizeMin = 5
        pipeline["aruco_parameters"].adaptiveThreshWinSizeMax = 19
        pipeline["aruco_parameters"].minMarkerPerimeterRate = 0.05
        pipeline["aruco_parameters"].polygonalApproxAccuracyRate = 0.03
        pipeline["aruco_parameters"].minCornerDistanceRate = 0.05
        pipeline["aruco_parameters"].minMarkerDistanceRate = 0.05
        pipeline["aruco_parameters"].adaptiveThreshConstant = 7
        pipeline["detector"] = cv2.aruco.ArucoDetector(pipeline["aruco_dictionary"], pipeline["aruco_parameters"])
        pipeline["poses_pub"] = self.create_publisher(PoseArray, poses_topic, 10)
        pipeline["markers_pub"] = self.create_publisher(ArucoMarkers, markers_topic, 10)

        # Subscriptions
        self.create_subscription(CameraInfo, info_topic, lambda msg: self.info_callback(msg, pipeline), qos_profile_sensor_data)
        self.create_subscription(Image, image_topic, lambda msg: self.image_callback(msg, pipeline), qos_profile_sensor_data)

        return pipeline

    def info_callback(self, info_msg, pipeline):
        pipeline["intrinsic_mat"] = np.reshape(np.array(info_msg.k), (3, 3))
        pipeline["distortion"] = np.array(info_msg.d)

    def image_callback(self, img_msg, pipeline):
        intrinsic_mat = pipeline.get("intrinsic_mat")
        distortion = pipeline.get("distortion")

        if intrinsic_mat is None or distortion is None:
            self.get_logger().warn("No camera calibration data loaded.")
            return

        frame = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        marker_corners, marker_ids, _ = pipeline["detector"].detectMarkers(frame)

        if marker_ids is not None:
            pose_array = PoseArray()
            markers = ArucoMarkers()
            pose_array.header.stamp = img_msg.header.stamp
            pose_array.header.frame_id = img_msg.header.frame_id
            markers.header = pose_array.header

            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                marker_corners, 0.0625, intrinsic_mat, distortion
            )
            
            MIN_MARKER_AREA = 1000 
            for i, marker_id in enumerate(marker_ids):
                if cv2.contourArea(marker_corners[i]) < MIN_MARKER_AREA: continue
                pose = Pose()
                pose.position.x = tvecs[i][0][0]
                pose.position.y = tvecs[i][0][1]
                pose.position.z = tvecs[i][0][2]

                # rot_matrix_add = np.array([
                #     [1, 0, 0],
                #     [0, np.cos(np.pi / 2), -np.sin(np.pi / 2)],
                #     [0, np.sin(np.pi / 2), np.cos(np.pi / 2)]
                # ])
                rot_matrix_add = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]
                ])

                rot_matrix = cv2.Rodrigues(np.array(rvecs[i][0]))[0]
                rot_matrix = np.dot(rot_matrix_add, rot_matrix)
                rot_matrix_hom = np.eye(4)
                rot_matrix_hom[:3, :3] = rot_matrix
                quat = tf_transformations.quaternion_from_matrix(rot_matrix_hom)

                pose.orientation.x = -quat[2]
                pose.orientation.y = quat[0]
                pose.orientation.z = -quat[1]
                pose.orientation.w = quat[3]

                pose_array.poses.append(pose)
                markers.poses.append(pose)
                markers.marker_ids.append(marker_id[0])

                self.publish_marker_tf(marker_id[0], pose, img_msg.header.frame_id)

            if len(pose_array.poses) > 0:
                pipeline["poses_pub"].publish(pose_array)
                pipeline["markers_pub"].publish(markers)
            else: return

    def publish_marker_tf(self, marker_id, pose, frame_id):
        if not frame_id or frame_id == "": frame_id = "tello_frame"
        if frame_id == "pc_frame":
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = frame_id
            t.child_frame_id = f"aruco_marker_{marker_id}"
            offset_x = 0.0
            offset_y = -0.13
            offset_z = -0.08
            t.transform.translation.x = pose.position.z + offset_x
            t.transform.translation.y = -pose.position.x + offset_y
            t.transform.translation.z = -pose.position.y + offset_z
            t.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(t)
        elif frame_id == "tello_frame":
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = frame_id
            t.child_frame_id = f"aruco_marker_{marker_id}"
            offset_x = 0.0
            offset_y = 0.0
            offset_z = 0.0
            t.transform.translation.x = pose.position.z + offset_x
            t.transform.translation.y = -pose.position.x + offset_y
            t.transform.translation.z = -pose.position.y + offset_z
            t.transform.rotation = pose.orientation
            self.tf_broadcaster.sendTransform(t)

def main():
    rclpy.init()
    node = ArucoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()