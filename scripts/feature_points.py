#!/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

import cv2

from sensor_msgs.msg import Image


class FeaturePoints(Node):

    def __init__(self):
        super().__init__('feature_points')
        self.subscription_ = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(Image, 'feature_points', 10)
        self.br_ = CvBridge()
        
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)
        
        self.flann_ = cv2.FlannBasedMatcher(index_params,search_params)
        
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
            orb = cv2.ORB_create(nfeatures=50)
            kp, des = orb.detectAndCompute(img, None)
            kp_prev = kp
            des_prev = des
            #matches = self.flann_.knnMatch(des_c,des_prev,k=2) 
            img2 = cv2.drawKeypoints(img, kp, None, (255,0,0), 4)
        elif method == 'none':
            img2 = img
        else:
            raise ValueError('unknown method')
       
        out_msg = self.br_.cv2_to_imgmsg(img2, encoding='rgb8')
        self.publisher_.publish(out_msg)
        return


def main(args=None):
    print("opencv version", cv2.__version__, cv2.__file__)
    rclpy.init(args=args)
    feature_points = FeaturePoints()
    rclpy.spin(feature_points)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
