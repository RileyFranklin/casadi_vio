#!/bin/env python3

# Visual Odometry Node
#
# This file runs the entire visual odometry algorithm

import rclpy
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import PoseStamped, TransformStamped
from tf2_ros import TransformBroadcaster
import tf_transformations
from cv_bridge import CvBridge
from voLib import *

import sys
sys.path.insert(0, '/home/purt/Github/RileyFranklin/pyecca')
from pyecca.lie_numpy import se3, so3

# Debug flags
stream_raw = False
stream_keypoints = False
stream_flann = False
skip_match = False
print_pose = False
save_pose = False

class Odometry(Node):

    #Constructor
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
            self.callback_pc,
            10)
        
        # Publishers
        self.pub_pose_ = self.create_publisher(PoseStamped, 'camera_pose', 10)
        self.pub_tf_broadcaster_ = TransformBroadcaster(self)

        # Defining Data
        self.SE3 = se3._SE3()
        self.SO3 = so3._Dcm()
        self.trajectory = np.zeros((3, 1))
        self.br = CvBridge()
        self.pointCloudFrame = None
        self.pointCloudFrame_last = None
        self.imageFrame = None
        self.kp_last = None
        self.des_last = None
        self.skip_estimate = False
        self.pose = np.eye(4)
        self.pose_bar = np.eye(4)
        self.k = np.array([[611.2769775390625, 0, 434.48028564453125],
                    [0, 609.7720336914062, 237.57313537597656],
                    [0, 0, 1]], dtype=np.float32)
        self.q = tf_transformations.quaternion_from_matrix(np.eye(4))
        self.t_vec = np.array([0,0,0])


    def callback_rgb(self, image_msg):
        self.imageFrame = self.br.imgmsg_to_cv2(image_msg)

        # Calculate Pose
        self.calc_pose()

    def callback_pc(self, pc_msg):

        #should we use pointcloud from current for location of previous points?
        #pointcloud is being published significantly slower than images
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
        kp, des = detect_features(self.imageFrame, 500)

        # Stream Detected Features
        if stream_keypoints:
            stream_features(self.imageFrame, kp)
        # Do Not Continue If First Frame
        if not skip_match and self.kp_last is not None and self.des_last is not None:
            # Detect Matches
            ransacking = False
            counter =0
            
            if ransacking:     
                while ransacking:

                    matches = detect_matches(self.des_last, des)
                    counter =counter+1

                    # Filter Matches
                    matches = filter_matches(matches, 0.7)
                    matches = ransac(matches, kp, self.kp_last)

                    if len(matches)>100:
                        self.skip_estimate =False
                        ransacking=False
                        # print(len(matches))
                    
                    if counter >9:
                        self.skip_estimate =True
                        ransacking=False
            
            else:
                matches = detect_matches(self.des_last, des)
                # Filter Matches
                matches = filter_matches(matches, 0.7)
                # print("matches before ransac:", len(matches))
                # matches = ransac(matches, kp, self.kp_last)
                # print("matches after ransac:", len(matches))


            # Stream Matches
            if stream_flann:
                stream_matches(self.imageFrame_last, self.kp_last, self.imageFrame, kp, matches)

            # Estimate Motion
            if not self.skip_estimate and self.pointCloudFrame is not None:
                # Estimate Change in Pose
                #pose_perturb_bar = estimate_motion_barfoot(matches, self.kp_last, kp, self.k, self.pointCloudFrame, self.pointCloudFrame,None,None)
                # pose_perturb = estimate_motion(matches, kp,self.kp_last, self.k, self.pointCloudFrame)
                # print("incremental ransasc: ",pose_perturb)
                pose_perturb_bar = estimate_motion_barfoot_ransac(matches, self.kp_last, kp, self.k, self.pointCloudFrame_last, self.pointCloudFrame,None,None)
                print("incremental barfoot: \n",pose_perturb_bar) 
                # Update Current Position
                # self.pose = pose_perturb_bar
                self.pose = pose_perturb_bar@self.pose
                # print('cumulated T: ', self.pose)
                  
                #print("Transformation: ",self.pose)
                #self.pose_bar=self.pose@np.linalg.inv(pose_perturb_bar)
                #print("barfoot Transformation: ",self.pose_bar)
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
        # bigT = np.eye(4)
        self.pose_corrected = bigT@np.linalg.inv(self.pose)
        # self.pose_corrected = bigT @ self.pose
        # self.pose_corrected = np.linalg.inv(self.pose_corrected)

        #print(self.pose)
        # lie_alg_op = self.SE3.vee(self.SE3.log(self.pose_corrected))
        # print("Lie algebra: ", lie_alg_op)
        # omega = lie_alg_op[3:6]
        # theta = np.linalg.norm(omega)
        # omega = (omega[0,:], omega[1,:], omega[2,:])
        # R = tf_transformations.rotation_matrix(theta, omega)

        #--------------------------------
        #Rotation
        R = np.eye(4)
        R[:3, :3] = self.pose_corrected[:3, :3]
        q = tf_transformations.quaternion_from_matrix(R)

        # - Update Rotation at each time instance
        # R0 = matrix_from_quaternion(self.q)
        # R = np.eye(4)
        # R[:3, :3] = self.pose_corrected[:3, :3]@R0
        # self.q = tf_transformations.quaternion_from_matrix(R)
        # q = self.q

        # - Update Euler Angle at each time instance
        # r, p, y = tf_transformations.euler_from_matrix(R, 'syxz')
        # current_rot = tf_transformations.euler_from_quaternion(self.q, 'syxz')
        # self.q = tf_transformations.quaternion_from_euler(current_rot[0] + r, current_rot[1] + p, current_rot[2] + y,  'syxz')
        # q = self.q 
        

        # - Update Quaternion at each time instance
        # q_diff = tf_transformations.quaternion_from_matrix(R)
        # self.q  = tf_transformations.quaternion_multiply(q_diff, self.q)
        # # self.q = quaternion_multiply(np.array(q_diff), np.array(self.q))
        # print("Quaternion", self.q)
        # q = self.q 
        # # print(t_rot.shape)

        # #Scaling Factor
        # factor, origin, direction  = tf_transformations.scale_from_matrix(self.pose_corrected)
        # print("scale:", factor)

        #--------------------------------
        #Translation
        t_vec = self.pose_corrected[:3,3]

        #Update translate at each time instance
        # t_vec_diff = self.pose_corrected[:3,3]
        # self.t_vec = self.t_vec + t_vec_diff
        # t_vec = self.t_vec

        p_inert = np.array([0,0,0,1])
        print('camera inertial position: ', self.pose_corrected @ p_inert)  

        #--------------------------------

        # msg = PoseStamped()
        # msg.pose.position.x = float(t_vec[0])
        # msg.pose.position.y = float(t_vec[1])
        # msg.pose.position.z = float(t_vec[2])
        # msg.header.frame_id = "map"
        # msg.pose.orientation.x = q[0]
        # msg.pose.orientation.y = q[1]
        # msg.pose.orientation.z = q[2]
        # msg.pose.orientation.w = q[3]
        # self.pub_pose_.publish(msg)

        t = TransformStamped()
        t.transform.translation.x = float(t_vec[0])
        t.transform.translation.y = float(t_vec[1])
        t.transform.translation.z = float(t_vec[2])
        # t.transform.translation.x = float(t_vec[2])
        # t.transform.translation.y = -1*float(t_vec[0])
        # t.transform.translation.z = -1*float(t_vec[1])
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "map"
        t.child_frame_id = "barfoot_vehicle"
        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]
        # t.transform.rotation.x = 0.0
        # t.transform.rotation.y = 0.0
        # t.transform.rotation.z = 0.0
        # t.transform.rotation.w = 1.0
        self.pub_tf_broadcaster_.sendTransform(t)

        # t = TransformStamped()
        # t.transform.translation.x = float(t_vec[0])
        # t.transform.translation.y = float(t_vec[1])
        # t.transform.translation.z = float(t_vec[2])
        # t.header.stamp = self.get_clock().now().to_msg()
        # t.header.frame_id = "map"
        # t.child_frame_id = "vehicle_frame"
        # t.transform.rotation.x = -q[0]
        # t.transform.rotation.y = -q[1]
        # t.transform.rotation.z = -q[2]
        # t.transform.rotation.w = q[3]
        # self.pub_tfransac_broadcaster_.sendTransform(t)

        self.skip_estimate=False
    # def save_data(self):
        np.save("test", self.trajectory)

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
