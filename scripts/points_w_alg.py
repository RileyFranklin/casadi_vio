#!/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

import cv2
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int32
import struct
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

        self.SE3 = se3._SE3()
        self.SO3 = so3._Dcm()

        
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
        self.u_=10
        self.v_=12

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
        print(p)
        print(len(p))
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
            orb = cv2.ORB_create(nfeatures=200)
            kp, des = orb.detectAndCompute(img, None)
            if self.img_prev_ is None:
                img2=img
                self.kp_prev_ = kp
                self.des_prev_ = des
                self.img_prev_ = img
                self.Top_ = self.SE3.exp(self.SE3.wedge(np.array([0,0,0,0,0,0])))
            else:
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
                self.kp_prev_ = kp
                self.des_prev_ = des
                self.img_prev_ = img 
                img_idx=matches[0][0].queryIdx
                
                (self.u_,self.v_)=kp[img_idx].pt
                self.u_=int(round(self.u_))
                self.v_=int(round(self.v_))
            
            
        elif method == 'none':
            img2 = img
        else:
            raise ValueError('unknown method')
       
        out_msg = self.br_.cv2_to_imgmsg(img2, encoding='rgb8')
        self.publisher_.publish(out_msg)
        landmarks=[[0,2,1]]
        y=[[0,2,1]]
        algopt = np.array([0,0,0,0,0,0])
        
        algoptprev = None
        #----- Point Cloud Alignment, iterative optimization for each time step k -------
        counter = 0
        while algoptprev is None or ca.norm_2(algopt-algoptprev)>1e-8:    
            algoptprev = algopt
            algopt = self.barfoot_solve(self.Top_,landmarks,y)
            self.Top_ = self.SE3.exp(self.SE3.wedge(algopt))@self.Top_
            counter +=1


        return



    def listener_callback_pc(self, msg):
        #maybe try to chop off rgb values
        ## get width and height of 2D point cloud data
        pCloud=msg
        width = 400 #pCloud.width
        height = 200#pCloud.height
        
        # Convert from u (column / width), v (row/height) to position in array
        # where X,Y,Z data starts
        arrayPosition = self.v_*pCloud.row_step + self.u_*pCloud.point_step
        
        #compute position in array where x,y,z data start
        PosX = arrayPosition + pCloud.fields[0].offset; #// X has an offset of 0
        PosY = arrayPosition + pCloud.fields[1].offset; #// Y has an offset of 4
        PosZ = arrayPosition + pCloud.fields[2].offset; #// Z has an offset of 8
        
        #X = struct.unpack('f',bytearray(pCloud.data[arrayPosX:arrayPosX+3]))
        Xbyte=bytearray([pCloud.data[PosX],pCloud.data[PosX+1],pCloud.data[PosX+2],pCloud.data[PosX+3]])
        Ybyte=bytearray([pCloud.data[PosY],pCloud.data[PosY+1],pCloud.data[PosY+2],pCloud.data[PosY+3]])
        Zbyte=bytearray([pCloud.data[PosZ],pCloud.data[PosZ+1],pCloud.data[PosZ+2],pCloud.data[PosZ+3]])
        X = struct.unpack('f',Xbyte)
        Y = struct.unpack('f',Ybyte)
        Z = struct.unpack('f',Zbyte)
        
        #print(X,',',Y,',',Z)

        



        
        return



def main(args=None):
    print("opencv version", cv2.__version__, cv2.__file__)
    rclpy.init(args=args)
    feature_points = FeaturePoints()
    rclpy.spin(feature_points)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
