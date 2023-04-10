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
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, AffineTransform

import sys
import time
sys.path.insert(0, '/home/scoops/git/Riley_Fork/pyecca')
from pyecca.lie_numpy import se3, so3

sys.path.insert(0, '/home/scoops/git/Riley_Fork/pyecca/notebooks/BA')
import BF_PCA

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
        
        FLANN_INDEX_LSH = 6
        index_params= dict(algorithm = FLANN_INDEX_LSH,
                   table_number = 6, # 12
                   key_size = 12,     # 20
                   multi_probe_level = 1) #2
        search_params = dict(checks=50)
        self.first_img_=0
        self.flann_ = cv2.FlannBasedMatcher(index_params,search_params)
        self.kp_prev_=None
        self.img_prev_=None
        self.des_prev_=None
        self.point_cloud_=None
        self.pc_prev = None
        self.pc = None
        self.nfeatures = 250
        self.u_=10
        self.v_=12
        self.depth_min_lim = 0.1
        self.depth_max_lim = 5
        self.ransac_T=None

        self.SE3 = se3._SE3()
        self.SO3 = so3._Dcm()
        self.Top_= self.SE3.exp(self.SE3.wedge([0,0,0,0,0,0]))

    def Ad(self, T):
        C = T[:3,:3]
        r = T[:3,3]
        return np.vstack((np.hstack((C, self.SO3.wedge(r)@C)), np.hstack((np.zeros([3,3]),C))))

    def barfoot_solve(self, Top, p, y):
        #the incorporated weights assume that every landmark is observed len(y) = len(w) = len(p)
        Tau = self.Ad(Top)
        Cop = Top[:3,:3]
        rop = np.expand_dims((-Cop.T@Top[:3,3]), axis=0)
        
        P = np.expand_dims(np.average(p,axis=0), axis=0)
        Y = np.expand_dims(np.average(y,axis=0), axis=0)
        
        I = 0
        for j in range(len(p)):
            pint0 = p[[j],:] - P

            I += self.SO3.wedge(pint0[0])@self.SO3.wedge(pint0[0])
        I=-I/len(p)
        
        M1 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((self.SO3.wedge(P[0]),np.eye(3)))))
        M2 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((np.zeros([3,3]),I))))
        M3 = np.vstack((np.hstack((np.eye(3), -self.SO3.wedge(P[0]))), np.hstack((np.zeros([3,3]),np.eye(3)))))
        M=M1@M2@M3

        W = 0
        for j in range(len(y)):
            pj = p[[j],:]
            yj = y[[j],:]

            W += (yj-Y).T@(pj-P)
        W = W/len(y)
        
        b=np.zeros([1,3])
        b[0,0] = np.trace(self.SO3.wedge([1,0,0])@Cop@W.T)
        b[0,1] = np.trace(self.SO3.wedge([0,1,0])@Cop@W.T)
        b[0,2] = np.trace(self.SO3.wedge([0,0,1])@Cop@W.T)

        a=np.vstack((Y.T-Cop@(P-rop).T, b.T-self.SO3.wedge(Y[0])@Cop@(P-rop).T))
        
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
                # matches = self.flann_.knnMatch(np.float32(des),np.float32(self.des_prev_),k=2)
                matches = self.flann_.knnMatch(des, self.des_prev_, k=2)

                # Chase filter (required)
                # print("Before Chase filter: ", len(matches))
                pruned_matches = [m for m in matches if len(m) == 2]
                matches = pruned_matches
                # print("After Chase filter: ", len(matches))

                # Need to draw only good matches, so create a mask
                matchesMask = [[0,0] for i in range(len(matches))]

                # ratio test as per Lowe's paper
                for i,(m,n) in enumerate(matches):
                    if m.distance < 0.7*n.distance:
                        matchesMask[i]=[1,0]
                draw_params = dict(matchColor = (0,255,0),
                                singlePointColor = (255,0,0),
                                matchesMask = matchesMask,
                                flags = cv2.DrawMatchesFlags_DEFAULT)
                img2 = cv2.drawMatchesKnn(img,kp,self.img_prev_,self.kp_prev_,matches,None,**draw_params)



                # Publish img2 to msg
                out_msg = self.br_.cv2_to_imgmsg(img2, encoding='rgb8')
                self.publisher_.publish(out_msg)

                # print("Before ratio test: ", len(matches))
                # Get only the good matches
                good_m = []
                for i, match in enumerate(matchesMask):
                    if match[0] == 1:
                        good_m.append(matches[i])
                matches = good_m
                # print("After ratio test: ", len(matches))
                
                if len(matches) > 1:

                    # Loop through all good points and store their x and y pixel location
                    list_pxl_prev = []
                    list_pxl = []
                    # Skips xyz point collection if not enough features are detected.
                    if not (len(kp) < self.nfeatures) and not (len(self.kp_prev_) < self.nfeatures):
                    # if True:

                        # For each match...
                        for mat in matches:

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

                            # # Hardcode box test problem
                            # points_0 = np.array([
                            #     [1, 0, 0],
                            #     [0, 1, 0],
                            #     [0, 0, 1],
                            #     [1, 0, 1],
                            #     [0, 1, 1],
                            #     [1, 1, 0],
                            #     [1, 1, 1],
                            #     [0, 1, 1],
                            # ])
                            # R_true = BF_PCA.euler2rot(1, 0.5, -1)
                            # t_true = np.array([[0],
                            #                 [0],
                            #                 [0],
                            #                 ])
                            # T_01_true = np.vstack([np.hstack([R_true, t_true]), np.array([[0, 0, 0, 1]])])
                            # T_01_true
                            # points_1 = BF_PCA.applyT(points_0, T_01_true)
                            # # print('T_01_true:', T_01_true)
                            # xyz_points_prev = points_0
                            # xyz_points = points_1


                            # Prune points that are too close or too far away from camera
                            print("Points before xyz distance pruning: ", len(xyz_points))
                            delete_ind_list = []
                            for lcv in range(len(xyz_points)):
                                # Lower tolerance
                                if (np.linalg.norm(xyz_points[[lcv],:]) < self.depth_min_lim) or (np.linalg.norm(xyz_points_prev[[lcv],:]) < self.depth_min_lim):
                                    delete_ind_list.append(lcv)
                                # Upper tolerance
                                elif (np.linalg.norm(xyz_points[[lcv],:]) > self.depth_max_lim) or (np.linalg.norm(xyz_points_prev[[lcv],:]) > self.depth_max_lim):
                                    delete_ind_list.append(lcv)
                            # Delete garbage points based on min/max depth limits
                            xyz_points_prev = np.delete(xyz_points_prev, delete_ind_list, 0)
                            xyz_points = np.delete(xyz_points, delete_ind_list, 0)
                            print("Points remaining after xyz distance pruning: ", len(xyz_points))

                            ransacking = True
                            counter2=0                            
                            while ransacking ==True:
                                rand_ints=np.sort(np.random.choice(len(xyz_points),5,replace=False))
                                rand_ints= rand_ints[::-1]
                                prev_points_rand=xyz_points_prev[rand_ints,:]
                                points_rand=xyz_points[rand_ints,:]
                                algopt = np.array([0,0,0,0,0,0])
                                self.ransac_T=self.SE3.exp(self.SE3.wedge(algopt))
                                algoptprev=None
                                unsorted = np.array(range(len(xyz_points)))
                                for i in rand_ints:
                                    unsorted = np.delete(unsorted,i)
                
                                inliers=rand_ints
                                counter2 += 1
                                counter = 0

                                while (algoptprev is None or np.linalg.norm(algopt-algoptprev)>1e-1) and counter<900:    
                                    algoptprev = algopt
                                    algopt = self.barfoot_solve(self.ransac_T,prev_points_rand,points_rand)
                                    self.ransac_T = self.SE3.exp(self.SE3.wedge(algopt))@self.ransac_T
                                    counter +=1
                                print(self.ransac_T)
                                for i in unsorted:
                                    if np.linalg.norm(np.expand_dims(np.append(xyz_points[i,:],1),axis=0).T - self.ransac_T@(np.expand_dims(np.append(xyz_points_prev[i,:],1),axis=0).T))<.1:
                                        inliers = np.append(inliers,i)
                                print(len(inliers))
                                if len(inliers)>40:
                                    xyz_points=xyz_points[inliers,:]
                                    xyz_points_prev=xyz_points_prev[inliers,:]
                                    ransacking=False
                                #else:
                                    #print('failure to find adequate guess')
                                if counter2 >10:
                                    print("I'm getting lost here!")
                                    return
                            print('num inliers',len(inliers))
                            

                            if len(xyz_points) > 2:
                                algopt = np.array([0,0,0,0,0,0])
                
                                algoptprev = None
                                #----- Point Cloud Alignment, iterative optimization for each time step k -------
                                counter = 0
                                if self.img_prev_ is not None:
                                    start_time = time.time()
                                    #  and counter<200
                                    while (algoptprev is None or np.linalg.norm(algopt-algoptprev)>1e-4):    
                                        algoptprev = algopt
                                        algopt = self.barfoot_solve(self.Top_,xyz_points_prev,xyz_points)
                                        self.Top_ = self.SE3.exp(self.SE3.wedge(algopt))@self.Top_
                                        counter += 1
                                    end_time = time.time()
                                    converge_time = end_time - start_time
                                    
                                    print("Converged in " + str(counter) + " iterations in " + str(converge_time) + " seconds.")
                                    print(self.Top_)
                                

                    else:
                        print("Detected number of features = ", len(kp), ". Skipping frame...")
                else:
                    print("Need at least 2 good matches to run. Skipping frame...")


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
                print("Zero features found. Skipping frame...")
            
        elif method == 'none':
            img2 = img
        else:
            raise ValueError('unknown method')
        # out_msg = self.br_.cv2_to_imgmsg(img2, encoding='rgb8')
        # self.publisher_.publish(out_msg)

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

        points_out = np.vstack([points['x'], points['y'], points['z']]).T

        return points_out

def main(args=None):
    print("opencv version", cv2.__version__, cv2.__file__)
    rclpy.init(args=args)
    feature_points = FeaturePoints()
    rclpy.spin(feature_points)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
