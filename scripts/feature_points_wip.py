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
        self.nfeatures = 10
        self.u_=10
        self.v_=12
        
        self.Top_=None

    def Ad(self, T):
        C = T[:3,:3]
        r = T[:3,3]
        return np.vstack(np.hstack(C, self.SO3_wedge(r)@C), np.hstack(np.zeros(3,3),C))
    def SE3_wedge(self, v):
        """
        This takes in an element of the se3 Lie Algebra and returns the se3 Lie Algebra matrix

        v: [x,y,z,theta0,theta1,theta2]
        """
        X = np.zeros(4, 4)
        X[0, 3] = v[0]
        X[1, 3] = v[1]
        X[2, 3] = v[2]
        X[:3, :3] = self.SO3_wedge(v[3:6])
        return X
    def SO3_wedge(self, v):
        X = np.ones(3, 3)
        theta0 = v[0]
        theta1 = v[1]
        theta2 = v[2]
        X[0, 1] = -theta2
        X[0, 2] = theta1
        X[1, 0] = theta2
        X[1, 2] = -theta0
        X[2, 0] = -theta1
        X[2, 1] = theta0
        return X
    
    def SO3_vee(self, X):
        v = np.zeros(3, 1)
        v[0, 0] = X[2, 1]
        v[1, 0] = X[0, 2]
        v[2, 0] = X[1, 0]
        return v
    
    def SE3_vee(self, X):
        """
        This takes in an element of the SE3 Lie Group (Wedge Form) and returns the se3 Lie Algebra elements
        """
        v = np.zeros(6, 1)
        v[0, 0] = X[0, 3]  # x
        v[1, 0] = X[1, 3]  # y
        v[2, 0] = X[2, 3]  # z
        v[3, 0] = X[2, 1]  # theta0
        v[4, 0] = X[0, 2]  # theta1
        v[5, 0] = X[1, 0]  # theta2
        return v
    
    # def SO3_exp(self, v):
    #     theta = np.linalg.norm(v)
    #     X = self.SO3_wedge(v)
    #     A = series_dict["sin(x)/x"]
    #     B = series_dict["(1 - cos(x))/x^2"]
    #     return np.eye(3) + A(theta) * X + B(theta) * X @ X
    
    def SE3_exp(self, v):  # accept input in wedge operator form
        v = self.SE3_vee(v)
        # v = [x,y,z,theta1,theta2,theta3]
        v_so3 = v[
            3:6
        ]  # grab only rotation terms for so3 uses ##corrected to v_so3 = v[3:6]
        X_so3 = self.SO3_wedge(v_so3)  # wedge operator for so3
        theta = np.linalg.norm(
            self.SO3_vee(X_so3)
        )  # theta term using norm for sqrt(theta1**2+theta2**2+theta3**2)

        # translational components u
        u = np.zeros(3, 1)
        u[0, 0] = v[0]
        u[1, 0] = v[1]
        u[2, 0] = v[2]

        R = self.SO3_exp(
            v_so3
        )  #'Dcm' for direction cosine matrix representation of so3 LieGroup Rotational

        A = series_dict["sin(x)/x"](theta)
        B = series_dict["(1 - cos(x))/x^2"](theta)
        C = (1 - A) / theta**2

        V = np.eye(3) + B * X_so3 + C * X_so3 @ X_so3

        horz = np.hstack(R, np.matmul(V, u))

        lastRow = np.array([0, 0, 0, 1]).T

        return np.vstack(horz, lastRow)

    def barfoot_solve(self, Top, p, y):
        #the incorporated weights assume that every landmark is observed len(y) = len(w) = len(p)
        Tau = self.Ad(Top)
        Cop = Top[:3,:3]
        rop = (-Cop.T@Top[:3,3])
        
        P=p
        Y=y
        P = np.average(p,axis=0)
        Y = np.average(y,axis=0)
        
        I = 0
        for j in range(len(p)):
            pint0=(p[j] - P)
            I += self.SO3_wedge(pint0)@self.SO3_wedge(pint0)
        I=-I/len(p)
        
        M1 = np.vstack(np.hstack(np.eye(3), np.zeros(3,3)), np.hstack(self.SO3_wedge(P),np.eye(3)))
        M2 = np.vstack(np.hstack(np.eye(3), np.zeros(3,3)), np.hstack(np.zeros(3,3),I))
        M3 = np.vstack(np.hstack(np.eye(3), -self.SO3_wedge(P)), np.hstack(np.zeros(3,3),np.eye(3)))
        M=M1@M2@M3
        
        W = 0
        for j in range(len(y)):
            pj = p[j]
            yj = y[j]
            
            W += (yj-Y)@(pj-P).T  
        W = W/len(y)
        
        b=np.zeros(1,3)
        b[0] = np.trace(self.SO3_wedge([1,0,0])@Cop@W.T)
        b[1] = np.trace(self.SO3_wedge([0,1,0])@Cop@W.T)
        b[2] = np.trace(self.SO3_wedge([0,0,1])@Cop@W.T) 

        a=np.vstack(Y-Cop@(P-rop),b.T-self.SO3_wedge(Y)@Cop@(P-rop))
        
        #Optimizied pertubation point
        eopt=Tau@np.linalg.inv(M)@Tau.T@a
        
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

                        algopt = np.array([0,0,0,0,0,0])
        
                        algoptprev = None
                        #----- Point Cloud Alignment, iterative optimization for each time step k -------
                        counter = 0
                        while algoptprev is None or np.linalg.norm(algopt-algoptprev)>1e-8:    
                            algoptprev = algopt
                            algopt = self.barfoot_solve(self.Top_,xyz_points_prev,xyz_points)
                            self.Top_ = self.SE3_exp(self.SE3.wedge(algopt))@self.Top_
                            counter +=1
                        
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

        points_out = np.vstack([points['x'], points['y'], points['z']])

        return points_out

def main(args=None):
    print("opencv version", cv2.__version__, cv2.__file__)
    rclpy.init(args=args)
    feature_points = FeaturePoints()
    rclpy.spin(feature_points)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
