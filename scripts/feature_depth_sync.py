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
        self.k = np.array([[607.79150390625, 0, 319.30987548828125],
                           [0, 608.1211547851562, 236.9514617919922],
                           [0, 0, 1]], dtype=np.float32)
        self.motion_counter = 0
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
                self.motion_counter += 1
                points_map, points_meas, points_meas_prev = self.box_test_case(self.motion_counter)
                
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

                pose_perturb_bar = estimate_motion_barfoot(matches, self.kp_last, kp, self.k, self.pointCloudFrame_last, self.pointCloudFrame, np.linalg.inv(pose_perturb), self.xyz_test_prev, self.xyz_test)
                # Update Current Position
                print('step rasnac:',np.linalg.inv(pose_perturb))
                print('step barfoot:',pose_perturb_bar)
                self.pose = self.pose @ pose_perturb_bar
                # self.pose = self.pose @ np.linalg.inv(pose_perturb)
                #print(self.pose)

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
            msg.pose.orientation.x = -q[0]
            msg.pose.orientation.y = -q[1]
            msg.pose.orientation.z = -q[2]
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
            t.transform.rotation.x = -q[0]
            t.transform.rotation.y = -q[1]
            t.transform.rotation.z = -q[2]
        else:
            t.transform.rotation.x = -q[2]
            t.transform.rotation.y = -q[0]
            t.transform.rotation.z = -q[1]
        t.transform.rotation.w = q[3]
        self.pub_tf_broadcaster_.sendTransform(t)

    # def save_data(self): 
        np.save("out", self.trajectory)


    def box_test_case(self, motion_counter):
        # Define points in world ("map") frame
        points_map = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0],
            [1, 1, 1],
            ])*2
        points_map = np.tile(points_map,(20,1))

        # Define velocity and euler angle rotations as a function of time_step
        # roll z, roll y, roll x
        omega = -np.array([0.0, 0.0, 0.0])
        # trans x, trans y, trans z
        v = -np.array([[0.0],
                    [0.1],
                    [0.0],
                    ])

        # Define rotation and translation for current time step
        R_true = BF_PCA.euler2rot(omega[0]*motion_counter, omega[1]*motion_counter, omega[2]*motion_counter)
        t_true = v*motion_counter
        
        # Craft the "truth" transformation matrix
        T_01_true = np.vstack([np.hstack([R_true, t_true]), np.array([[0, 0, 0, 1]])])
        
        # Apply the "truth" transformation matrix to all points_map
        points_meas = BF_PCA.applyT(points_map, T_01_true)

        # Define rotation and translation for current time step
        motion_counter = motion_counter - 1
        R_true_prev = BF_PCA.euler2rot(omega[0]*motion_counter, omega[1]*motion_counter, omega[2]*motion_counter)
        t_true_prev = v*motion_counter
        
        # Craft the "truth" transformation matrix
        T_01_true_prev = np.vstack([np.hstack([R_true_prev, t_true_prev]), np.array([[0, 0, 0, 1]])])
        
        # Apply the "truth" transformation matrix to all points_map
        points_meas_prev = BF_PCA.applyT(points_map, T_01_true_prev)
        
        # print('T_01_true:', T_01_true)

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

        return points_map, points_meas, points_meas_prev


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
