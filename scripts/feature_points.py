#!/bin/env python3

import array
from typing import Iterable, List, NamedTuple, Optional

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32
import struct
from sensor_msgs_py import point_cloud2
import casadi as ca


import sys
sys.path.insert(0, '/home/purt-admin/git/pyecca')
from pyecca.lie import se3, so3

class FeaturePoints(Node):

    def __init__(self):
        super().__init__('feature_points')
        self.subscription_ = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, 'feature_points', 10)
        self.subscription_ = self.create_subscription(
            PointCloud2,
            '/camera/depth/color/points',
            self.listener_callback_pc,
            10)
        self.publish = self.create_publisher(Int32, 'width', 10)
        self.br_ = CvBridge()
        
        FLANN_INDEX_LSH = 5
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   #key_size = 12,     # 20
                   #multi_probe_level = 1,
                   trees=5) #2
        search_params = dict(checks=50)
        self.first_img_=0
        self.flann_ = cv2.FlannBasedMatcher(index_params,search_params)
        self.kp_prev_=None
        self.img_prev_=None
        self.des_prev_=None
        self.point_cloud_=None
        self.pc_prev = None
        self.pc = None
        self.nfeatures = 200
        self.u_=10
        self.v_=12

        self.SE3 = se3._SE3()
        self.SO3 = so3._Dcm()
        self.Top_=None

    def Ad(self, T):
        C = T[:3,:3]
        r = T[:3,3]
        return ca.vertcat(ca.horzcat(C, self.SO3.wedge(r)@C), ca.horzcat(ca.SX.zeros(3,3),C))

    def barfoot_solve(self, Top, p, y):
        #the incorporated weights assume that every landmark is observed len(y) = len(w) = len(p)
        Tau = self.Ad(Top)
        Cop = Top[:3,:3]
        rop = (-Cop.T@Top[:3,3])
        
        P=p
        Y=y
        P = ca.SX(np.average(p,axis=0))
        Y = ca.SX(np.average(y,axis=0))
        
        I = 0
        for j in range(len(p)):
            pint0=(p[j] - P)
            I += self.SO3.wedge(pint0)@self.SO3.wedge(pint0)
        I=-I/len(p)
        
        M1 = ca.vertcat(ca.horzcat(ca.SX.eye(3), ca.SX.zeros(3,3)), ca.horzcat(self.SO3.wedge(P),ca.SX.eye(3)))
        M2 = ca.vertcat(ca.horzcat(ca.SX.eye(3), ca.SX.zeros(3,3)), ca.horzcat(ca.SX.zeros(3,3),I))
        M3 = ca.vertcat(ca.horzcat(ca.SX.eye(3), -self.SO3.wedge(P)), ca.horzcat(ca.SX.zeros(3,3),ca.SX.eye(3)))
        M=M1@M2@M3
        
        W = 0
        for j in range(len(y)):
            pj = p[j]
            yj = y[j]
            
            W += (yj-Y)@(pj-P).T  
        W = W/len(y)
        
        b=ca.SX.zeros(1,3)
        b[0] = ca.trace(self.SO3.wedge([1,0,0])@Cop@W.T)
        b[1] = ca.trace(self.SO3.wedge([0,1,0])@Cop@W.T)
        b[2] = ca.trace(self.SO3.wedge([0,0,1])@Cop@W.T) 

        a=ca.vertcat(Y-Cop@(P-rop),b.T-self.SO3.wedge(Y)@Cop@(P-rop))
        
        #Optimizied pertubation point
        eopt=Tau@ca.inv(M)@Tau.T@a
        
        return eopt   
    def listener_callback(self, msg):
        img = self.br_.imgmsg_to_cv2(msg)

        method = 'orb'

        if method == 'surf':
            surf = cv2.SURF_create()
            kp, des = surf.detectAndCompute(img, None)
            img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
        elif method == 'sift':
            sift = cv2.SIFT_create()
            kp, des = sift.detectAndCompute(img, None)
            img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
        elif method == 'orb':
            orb = cv2.ORB_create(nfeatures=self.nfeatures)
            kp, des = orb.detectAndCompute(img, None)
            if self.img_prev_ is None:
                img2=img
                self.kp_prev_ = kp
                self.des_prev_ = des
                self.img_prev_ = img
            # Catches if zero features are calculated.
            elif (len(kp) > 0) and (len(self.kp_prev_) > 0):
                matches = self.flann_.knnMatch(np.float32(des),np.float32(self.des_prev_),k=2)
                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for i in range(len(matches))]
                # ratio test as per Lowe's paper
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.60*n.distance:
                        matchesMask[i]=[1,0]
                draw_params = dict(matchColor = (0,255,0),
                                singlePointColor = (255,0,0),
                                matchesMask = matchesMask,
                                flags = cv2.DrawMatchesFlags_DEFAULT)
                img2 = cv2.drawMatchesKnn(img,kp,self.img_prev_,self.kp_prev_,matches,None,**draw_params)

                # Publish img2 to msg
                out_msg = self.br_.cv2_to_imgmsg(img2, encoding='rgb8')
                self.publisher_.publish(out_msg)

                # Get only the good matches
                good_m2 = []
                for i, match in enumerate(matchesMask):
                    if match[0] == 1:
                        good_m2.append(matches[i])

                # Loop through all good points and store their x and y pixel location
                list_pxl_prev = []
                list_pxl = []

                # Skips xyz point collection if not enough features are detected.
                if not (len(kp) < self.nfeatures) and not (len(self.kp_prev_) < self.nfeatures):

                    # For each match...
                    for mat in good_m2:

                        # Get the matching keypoints for each of the images
                        img0_idx = mat[0].queryIdx
                        img1_idx = mat[0].trainIdx

                        # x - columns
                        # y - rows
                        # Get the coordinates
                        (x0, y0) = self.kp_prev_[img0_idx].pt
                        (x1, y1) = kp[img1_idx].pt

                        # Append to each list
                        list_pxl_prev.append((int(round(x0)), int(round(y0))))
                        list_pxl.append((int(round(x1)), int(round(y1))))

                    # Use get_points_efficient to get xyz points for all good matches
                    if self.pc_prev is not None:
                        xyz_points_prev = self.read_points_efficient(self.pc_prev, uvs=list_pxl_prev, field_names = ("x", "y", "z"))
                        xyz_points = self.read_points_efficient(self.pc, uvs=list_pxl, field_names = ("x", "y", "z"))

                else:
                    print("Not enough features detected. Skipping frame...")

                # # Testing
                # img_idx=matches[0][0].queryIdx
                
                # (self.u_,self.v_)=kp[img_idx].pt
                # self.u_=int(round(self.u_))
                # self.v_=int(round(self.v_))

                # Update previous values in self
                self.kp_prev_ = kp
                self.des_prev_ = des
                self.img_prev_ = img 

            else:
                print("Not enough features detected. Skipping frame...")
            
        elif method == 'none':
            img2 = img
        else:
            raise ValueError('unknown method')
        algopt = np.array([0,0,0,0,0,0])
        
        algoptprev = None
        #----- Point Cloud Alignment, iterative optimization for each time step k -------
        counter = 0
        if self.pc_prev is not None and self.img_prev_ is not None:
            print('yep')
            while algoptprev is None or ca.norm_2(algopt-algoptprev)>1e-8:    
                algoptprev = algopt
                algopt = self.barfoot_solve(self.Top_,xyz_points_prev,xyz_points)
                self.Top_ = self.SE3.exp(self.SE3.wedge(algopt))@self.Top_
                counter +=1

        return

    def listener_callback_pc(self, msg):
        # Store current and previous PointCloud2 msg to self
        if self.pc_prev is None:
            self.pc = msg
            self.pc_prev = msg
            return
        else:
            self.pc_prev = self.pc
            self.pc = msg



        return
    
    def read_points_efficient(
        self,
        cloud: PointCloud2,
        field_names: Optional[List[str]] = None,
        uvs: Optional[Iterable] = None) -> np.ndarray:

        assert isinstance(cloud, PointCloud2), \
            'Cloud is not a PointCloud2'        

        # Grab only the data you want
        if uvs is not None:
            select_data = array.array('B', [])

            # Loop through all provided pixel locations
            for u,v in uvs:
                start_ind = u*cloud.point_step + v*cloud.row_step
                curr_data = cloud.data[start_ind:start_ind+cloud.point_step]

                select_data += curr_data
            points_len = len(uvs)
        else:
            select_data = cloud.data
            points_len = cloud.width * cloud.height

        # Cast bytes to numpy array
        points = np.ndarray(
            shape=(points_len, ),
            dtype=point_cloud2.dtype_from_fields(cloud.fields, point_step=cloud.point_step),
            buffer=select_data)
        
        # Keep only the requested fields
        if field_names is not None:
            assert all(field_name in points.dtype.names for field_name in field_names), \
                'Requests field is not in the fields of the PointCloud!'
            # Mask fields
            points = points[list(field_names)]

        return points

def main(args=None):
    print("opencv version", cv2.__version__, cv2.__file__)
    rclpy.init(args=args)
    feature_points = FeaturePoints()
    rclpy.spin(feature_points)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
