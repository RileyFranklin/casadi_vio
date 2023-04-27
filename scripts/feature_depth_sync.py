#!/bin/env python3

# Visual Odometry Node
#
# This file runs the entire visual odometry algorithm

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped, Point
from visualization_msgs.msg import MarkerArray, Marker
from std_msgs.msg import Int32, ColorRGBA
from rclpy.duration import Duration
from tf2_ros import TransformBroadcaster
import tf_transformations
from cv_bridge import CvBridge
from voLib import *
from feature_points_pnpransac_debug import FeaturePoints
import sys
sys.path.insert(0, '/home/scoops/git/Riley_Fork/pyecca/notebooks/BA')
import BF_PCA


# Debug flags
stream_raw = False
stream_keypoints = False
stream_flann = False
skip_match = False
skip_estimate = False
print_pose = False
save_pose = True
box_test = True

class Odometry(Node):
    def __init__(self):
        super().__init__('odom_root')

        # Subscribers
        self.subscription = self.create_subscription(
                Image,
                '/camera/color/image_raw',
                self.callback_rgb,
                10)
        self.subscription = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.callback_depth,
            10)
        
        # Publishers
        self.pub_pose_ = self.create_publisher(PoseStamped, 'camera_pose', 10)
        self.pub_tf_broadcaster_ = TransformBroadcaster(self)
        self.pub_marker_ = self.create_publisher(Marker, 'feature_markers', 10)

        # Defining Data
        self.trajectory = np.zeros((3, 1))
        self.br = CvBridge()
        self.pointCloudFrame = None
        self.imageFrame = None
        self.pointCloudFrame_last = None
        self.kp_last = None
        self.des_last = None
        self.imageFrame_last = None
        self.pose = np.eye(4)
        self.pose_prev = np.eye(4)
        self.k = np.array([[607.79150390625, 0, 319.30987548828125],
                           [0, 608.1211547851562, 236.9514617919922],
                           [0, 0, 1]], dtype=np.float32)
        self.box_test_counter = 0
        self.xyz_test_prev = None
        self.xyz_test = None

    def callback_rgb(self, image_msg):
        self.imageFrame = self.br.imgmsg_to_cv2(image_msg)


    def callback_depth(self, pc_msg):
        if self.pointCloudFrame_last == None:
            self.pointCloudFrame_last = pc_msg
            self.pointCloudFrame_last_img = self.imageFrame

            self.pointCloudFrame = pc_msg
            self.pointCloudFrame_img = self.imageFrame
        else:
            # Assign previous pc and img
            self.pointCloudFrame_last = self.pointCloudFrame
            self.pointCloudFrame_last_img = self.pointCloudFrame_img

            # Assign current pc and img
            self.pointCloudFrame = pc_msg
            self.pointCloudFrame_img = self.imageFrame

            # Hardcode box test problem
            if box_test:
                points_map, points_meas, points_meas_prev = self.box_test_case()
                
                xyz_points = points_meas
                xyz_points_prev = points_meas_prev

                # # If you are not running frame to frame, you'll need to apply Top_ to all your previous points
                # homo_xyz_points_prev = np.hstack((xyz_points_prev, np.ones([xyz_points_prev.shape[0],1])))
                # xyz_points_prev = np.linalg.inv(self.Top_)@homo_xyz_points_prev.T
                # xyz_points_prev = xyz_points_prev[0:3,:].T

                self.xyz_test_prev = xyz_points_prev
                self.xyz_test = xyz_points

            # Calculate Pose
            self.calc_pose()

    def calc_pose(self):
        # Stream RGB Video
        if stream_raw:
            stream_rgb(self.pointCloudFrame_img)

        # Detect Features
        kp, des = detect_features(self.pointCloudFrame_img, 1000)

        # Stream Detected Features
        if stream_keypoints:
            stream_features(self.pointCloudFrame_img, kp)
        # Do Not Continue If First Frame
        if not skip_match and self.kp_last is not None and self.des_last is not None:
            # Detect Matches
            matches = detect_matches(self.des_last, des)

            # Filter Matches
            matches = filter_matches(matches, 0.7)

            # Stream Matches
            if stream_flann:
                stream_matches(self.pointCloudFrame_last_img, self.kp_last, self.pointCloudFrame_img, kp, matches)

            # Estimate Motion
            if not skip_estimate and self.pointCloudFrame is not None:
                # Estimate Change in Pose
                pose_perturb = estimate_motion(matches, self.kp_last, kp, self.k, self.pointCloudFrame)

                pose_perturb_bar = estimate_motion_barfoot_ransac(matches, self.kp_last, kp, self.k, self.pointCloudFrame_last, self.pointCloudFrame,  self.xyz_test_prev, self.xyz_test)
                #self.pose_prev=pose_perturb_bar
                # pose_perturb_bar = estimate_motion_barfoot(matches, self.kp_last, kp, self.k, self.pointCloudFrame_last, self.pointCloudFrame,  self.xyz_test_prev, self.xyz_test)
                # Update Current Position
                # print('step barfoot:',pose_perturb_bar)
                self.pose = self.pose @ pose_perturb_bar
                #self.pose = self.pose @ pose_perturb_bar
                #self.pose_test = self.pose_test @ np.linalg.inv(pose_perturb)
                #print(self.pose)
                #print(self.pose_test)
                # print("barfoot",pose_perturb_bar)
                #print("pnpransac",np.linalg.inv(pose_perturb))
                
                # Build Trajectory
                coordinates = np.array([[self.pose[0, 3], self.pose[1, 3], self.pose[2, 3]]])
                self.trajectory = np.concatenate((self.trajectory, coordinates.T), axis=1)


        # Save Frame Data for Next Frame
        self.kp_last = kp
        self.des_last = des
        # self.imageFrame_last = self.imageFrame

        # Publish content
        bigT = np.array([[0.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
        if box_test:
            bigT = np.eye(4)
        bigT = np.eye(4)

        corrected_pose = bigT @ self.pose

        q = tf_transformations.quaternion_from_matrix(self.pose)

        t_vec = corrected_pose[0:3,3]
        # print(t_rot.shape)
        # t_act = -self.Top_[:3,:3].T@t_rot

        msg = PoseStamped()
        msg.pose.position.x = float(t_vec[0])
        msg.pose.position.y = float(t_vec[1])
        msg.pose.position.z = float(t_vec[2])
        msg.header.frame_id = "map"
        if box_test:
            msg.pose.orientation.x = q[0]
            msg.pose.orientation.y = q[1]
            msg.pose.orientation.z = q[2]
        else:
            msg.pose.orientation.x = -q[2]
            msg.pose.orientation.y = -q[0]
            msg.pose.orientation.z = -q[1]
        msg.pose.orientation.w = q[3]
        self.pub_pose_.publish(msg)

        t = TransformStamped()
        t.transform.translation.x = float(t_vec[0])
        t.transform.translation.y = float(t_vec[1])
        t.transform.translation.z = float(t_vec[2])
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "vehicle_frame"
        if box_test:
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
        else:
            t.transform.rotation.x = -q[2]
            t.transform.rotation.y = -q[0]
            t.transform.rotation.z = -q[1]
        t.transform.rotation.w = q[3]
        self.pub_tf_broadcaster_.sendTransform(t)

    # def save_data(self): 
        np.save("out", self.trajectory)


    def box_test_case(self):
        # Define points in world ("map") frame
        points_map = np.array([
            [1, 2, 3],
            [-1, 2, 3],
            [-1, -2, 3],
            [1,-2, 3],
            [1, 2, -3],
            [-1, 2, -3],
            [-1, -2, -3],
            [1, -2, -3],
            ])
        points_map = np.tile(points_map,(20,1))

        # Measure points with current position
        points_meas_prev = self.meas_points(self.pose, points_map)

        # User inputs here:
        t_vec = [0.01, 0.0, 0.0]         # x, y, z
        angles_vec = [0.0, 0.0, 0.01]    # roll, pitch, yaw euler angles (1-2-3 sequence)

        # Propogate movement and measure points again with new position
        T_frame2frame = self.calc_T_frame2frame(t_vec, angles_vec)
        new_pose = self.robot_move(self.pose, T_frame2frame)
        points_meas = self.meas_points(new_pose, points_map)

        # For comparison with self.pose
        # Note these are both for the previous time step. -1 is needed because first time box_test is run barfoot is skipped (first starting node)
        truth_pose = np.linalg.matrix_power(T_frame2frame, self.box_test_counter-1)
        print('truth_pose: ', np.round(truth_pose,6))
        print('self.pose: ', np.round(self.pose,6))

        # Debugging
        # T_frame2frame can compare against pose_perturb_bar
        # print('T_frame2frame: ', T_frame2frame)

        msg = Marker()
        msg.type = Marker.POINTS
        msg.scale.x = 0.05
        msg.scale.y = 0.05
        msg.scale.z = 0.05
        msg.lifetime = Duration(seconds=10).to_msg()
        msg.header.frame_id = "map"
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.action = Marker.ADD
        white = ColorRGBA()
        white.r = 1.0
        white.g = 1.0
        white.b = 1.0
        white.a = 1.0
        for point in points_map:
            msg_point = Point()
            msg_point.x = float(point[0])
            msg_point.y = float(point[1])
            msg_point.z = float(point[2])

            msg.points.append(msg_point)
            msg.colors.append(white)
        self.pub_marker_.publish(msg)

        # Increment box_counter
        self.box_test_counter += 1

        return points_map, points_meas, points_meas_prev

    def euler2rot(self, phi, theta, psi):
        # Assumes roll, pitch, yaw sequence (1-2-3)
        # phi - roll angle in radians
        # theta - pitch angle in radians
        # psi - yaw angle in radians
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(phi), -np.sin(phi)],
                    [0, np.sin(phi), np.cos(phi)],
                    ])
        Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                    [0, 1, 0],
                    [-np.sin(theta), 0, np.cos(theta)],
                    ])
        Rz = np.array([[np.cos(psi), -np.sin(psi), 0],
                    [np.sin(psi), np.cos(psi), 0],
                    [0, 0, 1],
                    ])
        
        R = Rx @ Ry @ Rz
        
        return R
    
    def calc_T_frame2frame(self, t_change_body, euler_change_body):
        # Calculates an SE(3) element
        # t_change_body - (3,1), (1,3), or (3,) vector with xyz translation
        # euler_change_body - (3,1), (1,3), or (3,) vector with roll, pitch, yaw Euler angles for XYZ rotation sequence.
        
        R = self.euler2rot(euler_change_body[0], euler_change_body[1], euler_change_body[2])
        
        T = np.eye(4)
        
        T[0:3, 0:3] = R
        T[0, 3] = t_change_body[0]
        T[1, 3] = t_change_body[1]
        T[2, 3] = t_change_body[2]
        
        return T
    
    def meas_points(self, T_map2body_map, points_map):
        # Parse T_map2body_map
        t_map = T_map2body_map[0:3, 3]
        R_map2body = T_map2body_map[0:3, 0:3]
        
        y_map = np.zeros([len(points_map), 3])
        y_body = np.zeros([len(points_map), 3])
        for lcv, point in enumerate(points_map):
            # Point - Body = vector from body to point in map frame
            y_map[lcv, :] = points_map[lcv, :] - t_map
            
            # Convert map frame measurements to body frame via R_map2body
            y_body[lcv, :] = np.linalg.inv(R_map2body) @ y_map[lcv, :]
        
        return y_body
    
    def robot_move(self, T_start_map, T_frame2frame):
        # Translate then rotate
        # Translate in current body frame, then rotate once you arrive to the new location.
        
        # Result is in map frame
        return T_start_map @ T_frame2frame

def main(args=None):
    rclpy.init(args=args)
    odom_node = Odometry()

    try:
        rclpy.spin(odom_node)
    except:
        # Save Data
        # This is such a terrible way of doing this, but it works
        # TODO: maybe try something more sensible one day
        odom_node.save_data()
        odom_node.destroy_node()
        rclpy.shutdown()

    odom_node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
