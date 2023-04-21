#!/bin/env python3

# Visual Odometry Node
#
# This file runs the entire visual odometry algorithm

import rclpy
import cv2
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
from cv_bridge import CvBridge
from voLib import *

# Debug flags
stream_raw = False
stream_keypoints = False
stream_flann = False
skip_match = False
skip_estimate = False
print_pose = False
save_pose = True

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

    def callback_rgb(self, image_msg):
        self.imageFrame = self.br.imgmsg_to_cv2(image_msg)

        # Calculate Pose
        self.calc_pose()

    def callback_depth(self, pc_msg):
        if self.pointCloudFrame_last == None:
            self.pointCloudFrame=pc_msg
            self.pointCloudFrame_last=pc_msg
        else:
            self.pointCloudFrame_last=self.pointCloudFrame
            self.pointCloudFrame = pc_msg

    def calc_pose(self):
        # Stream RGB Video
        if stream_raw:
            stream_rgb(self.imageFrame)

        # Detect Features
        kp, des = detect_features(self.imageFrame, 1000)

        # Stream Detected Features
        if stream_keypoints:
            stream_features(self.imageFrame, kp)
        # Do Not Continue If First Frame
        if not skip_match and self.kp_last is not None and self.des_last is not None:
            # Detect Matches
            matches = detect_matches(self.des_last, des)

            # Filter Matches
            matches = filter_matches(matches, 0.7)

            # Stream Matches
            if stream_flann:
                stream_matches(self.imageFrame_last, self.kp_last, self.imageFrame, kp, matches)

            # Estimate Motion
            if not skip_estimate and self.pointCloudFrame is not None:
                # Estimate Change in Pose
                pose_perturb = estimate_motion(matches, self.kp_last, kp, self.k, self.pointCloudFrame)

                pose_perturb_bar = estimate_motion_barfoot(matches, self.kp_last, kp, self.k, self.pointCloudFrame,self.pointCloudFrame ,np.linalg.inv(pose_perturb))
                # Update Current Position
                print('step rasnac:',np.linalg.inv(pose_perturb))
                print('step barfoot:',pose_perturb_bar)
                self.pose = self.pose @ pose_perturb_bar
                #print(self.pose)

                # Build Trajectory
                coordinates = np.array([[self.pose[0, 3], self.pose[1, 3], self.pose[2, 3]]])
                self.trajectory = np.concatenate((self.trajectory, coordinates.T), axis=1)


        # Save Frame Data for Next Frame
        self.kp_last = kp
        self.des_last = des
        self.imageFrame_last = self.imageFrame

        # Publish content
        bigT = np.array([[0.0, 0.0, 1.0, 0.0],
                        [-1.0, 0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0]])
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
        msg.pose.orientation.x = -q[0]
        msg.pose.orientation.y = -q[1]
        msg.pose.orientation.z = -q[2]
        msg.pose.orientation.w = q[3]
        self.pub_pose_.publish(msg)

        t = TransformStamped()
        t.transform.translation.x = float(t_vec[0])
        t.transform.translation.y = float(t_vec[1])
        t.transform.translation.z = float(t_vec[2])
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "vehicle_frame"
        t.transform.rotation.x = -q[0]
        t.transform.rotation.y = -q[1]
        t.transform.rotation.z = -q[2]
        t.transform.rotation.w = q[3]
        self.pub_tf_broadcaster_.sendTransform(t)

    # def save_data(self): 
        np.save("out", self.trajectory)

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
