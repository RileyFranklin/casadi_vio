#!/bin/env python3

# Helper functions for run.py

import cv2
import numpy as np
import array
from sensor_msgs_py import point_cloud2
from  feature_points_pnpransac_debug import FeaturePoints
import random

def estimate_motion(matches, kp_last, kp, k, pointCloud):
    # Declare lists and arrays
    rmat = np.eye(3)
    tvec = np.zeros((3, 1))
    pxl_list_last = []
    pxl_list = []
    points = []
    valid_points = []
    valid_pxl_list = []

    # Collect feature pixel locations
    for match in matches:
        x1, y1 = kp_last[match[0].queryIdx].pt
        x2, y2 = kp[match[0].trainIdx].pt
        pxl_list_last.append([int(x1), int(y1)])
        pxl_list.append([int(x2), int(y2)])

    # Read point cloud data
    points = read_point_efficient(pointCloud, pxl_list_last)

    # Remove pixel locations if there is no corresponding point
    for point in range(len(points)):
        if np.linalg.norm(points[point]) > 0:
            valid_points.append(points[point])
            valid_pxl_list.append(pxl_list[point])

    # Solve for relative motion
    if len(valid_points) > 5:

        # Ransac and solve
        _, rvec, tvec, _ = cv2.solvePnPRansac(np.vstack(valid_points), np.array(valid_pxl_list).astype(np.float32), k, None)

        # Convert to Rodrigues
        rmat, _ = cv2.Rodrigues(rvec)

    # Pack
    pose_perturb = np.eye(4)
    pose_perturb[0:3, 0:3] = rmat
    pose_perturb[0:3, 3] = tvec.T

    return pose_perturb

def estimate_motion_barfoot(matches, kp_last, kp, k, pointCloud_last, pointCloud,  xyz_test_prev, xyz_test):
    # Declare lists and arrays
    # rmat = np.eye(3)
    # tvec = np.zeros((3, 1))
    pxl_list_last = []
    pxl_list = []
    points = []
    valid_points = []
    valid_points_prev=[]
    valid_pxl_list = []
    valid_pxl_list_prev = []
    feature =FeaturePoints()
    Top_=np.eye(4)
    # Collect feature pixel locations
    print(len(matches))
    for match in matches:
        x1, y1 = kp_last[match[0].queryIdx].pt
        x2, y2 = kp[match[0].trainIdx].pt
        pxl_list_last.append([int(x1), int(y1)])
        pxl_list.append([int(x2), int(y2)])
    
    # Read point cloud data
    points = read_point_efficient(pointCloud, pxl_list)
    points_prev = read_point_efficient(pointCloud_last, pxl_list_last)

    # Remove pixel locations if there is no corresponding point
    for point in range(len(points)):
        if np.linalg.norm(points[point]) > 0.05 and np.linalg.norm(points[point]) <20 :
            valid_points.append(points[point])
            valid_points_prev.append(points_prev[point])
            valid_pxl_list.append(pxl_list[point])
            valid_pxl_list_prev.append(pxl_list_last[point])
    valid_points_prev=np.vstack(valid_points_prev)    
    valid_points=np.vstack(valid_points)   
    # if len(valid_points) > 5:

    #     # Ransac and solve
    #     _, rvec, tvec, _ = cv2.solvePnPRansac(np.vstack(valid_points), np.array(valid_pxl_list).astype(np.float32), k, None)

    #     # Convert to Rodrigues
    #     rmat, _ = cv2.Rodrigues(rvec)

    # # Pack
    # pose_perturb = np.eye(4)
    # pose_perturb[0:3, 0:3] = rmat
    # pose_perturb[0:3, 3] = tvec.T

    inliers=np.array([])
    points_clean=inliers
    points_clean_prev=inliers
    
    # for i in range(len(valid_points)):
    #     y=np.append(valid_points[i],1)
    #     z=pose_perturb@(np.append(valid_points_prev[i],1)).T
    #     # print('observed',y)
    #     # print(pose_perturb)
    #     # print('transformed',z)
    #     # print('norm',np.linalg.norm(y-z))
    #     if np.linalg.norm(y-z) < 2:
    #         if len(points_clean) == 0:
    #             points_clean = valid_points[i]
    #             points_clean_prev = valid_points_prev[i]
    #         else:        
    #             points_clean=np.vstack((points_clean,valid_points[i]))
    #             points_clean_prev=np.vstack((points_clean_prev,valid_points[i]))
    # print('valid_points',len(points_clean))
    # valid_points=points_clean
    # valid_points_prev=points_clean_prev    

    if xyz_test is not None:
        valid_points = xyz_test
        valid_points_prev = xyz_test_prev
 
    if len(valid_points) > 5:
        print('success')
        P = np.average(valid_points_prev, axis=0)
        print(valid_points_prev[:10,:])
        Pexp_dim = np.average(valid_points_prev[:10,:], axis=0)
        print(Pexp_dim)
        Y = np.average(valid_points,axis=0)
        I = 0
        for j in range(len(valid_points_prev)):
            pint0 = valid_points_prev[j,:] - P
            I += feature.SO3.wedge(pint0)@feature.SO3.wedge(pint0)
        I=-I/len(valid_points_prev)
        
        M1 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((feature.SO3.wedge(P),np.eye(3)))))
        M2 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((np.zeros([3,3]),I))))
        M3 = np.vstack((np.hstack((np.eye(3), -feature.SO3.wedge(P))), np.hstack((np.zeros([3,3]),np.eye(3)))))
        M=M1@M2@M3
        algopt=None
        counter=0
        while (algopt is None or np.linalg.norm(algopt)>1e-10) and counter<100:    
            algopt = feature.barfoot_solve(Top_, valid_points_prev, valid_points, M)
            Top_ = feature.SE3.exp(feature.SE3.wedge(algopt))@Top_
            counter += 1
        #print('converged after ',counter,' iterations')
        r = Top_[0:3, 0:3]
        t = Top_[0:3, 3]
        # Top_[0:3,0:3]=r.T
        # Top_[0:3, 3] = -r.T@t
        print('algopt',feature.SE3.vee(feature.SE3.log(Top_)))
        p_test = np.array([1,2,3,1])
        print('T_op @ p_test: ', Top_ @ p_test)
        #print("algopt: ", algopt)
    else:
        print('fail')   

    
    #print('Top barfoot:',Top_)
    
    return Top_

def read_point_efficient(pointCloud, pxl_list):
    # Define points to use
    select_data = array.array('B', [])
    for x, y in pxl_list:
        start_idx = x * pointCloud.point_step + y * pointCloud.row_step
        select_data += pointCloud.data[start_idx:start_idx + pointCloud.point_step]
    points_len = len(pxl_list)

    points = np.ndarray(
        shape=(points_len, ),
        dtype=point_cloud2.dtype_from_fields(pointCloud.fields, point_step=pointCloud.point_step),
        buffer=select_data)
    points_out = np.vstack([points['x'], points['y'], points['z']]).T
    return points_out
def ransac(matches,kp1,kp2):
    src_pts = np.float32([kp2[m[0].queryIdx].pt for m in matches]).reshape(-1,1,2)
    dst_pts = np.float32([kp1[m[0].trainIdx].pt for m in matches]).reshape(-1,1,2)  

    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RHO,5.0)
    print('M',M)
    matchesMask = mask.ravel().tolist()
    good=[]
    
    for i in matchesMask:
        if matchesMask[i]:
            good.append(matches[i])
    print(len(good))
    return good

def filter_matches(matches, threshold):
    filtered = []
    for match in matches:
        if len(match) == 2 and match[0].distance < (threshold * match[1].distance):
            filtered.append(match)

    return filtered

def stream_matches(imageFrame_last, kp_last, imageFrame, kp, matches):
    # Define draw parameters
    draw_params = dict(matchColor=(255, 0, 0),
                       singlePointColor=(0, 255, 0),
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    # Create Images
    visual = cv2.drawMatchesKnn(imageFrame, kp, imageFrame_last, kp_last, matches, None, **draw_params)

    # Draw Images
    cv2.imshow("Matches", visual)
    cv2.waitKey(1)

def detect_matches(des_last, des):
    # Define FLANN parameters
    index_params = dict(algorithm=6,
                        table_number=6,
                        key_size=12,
                        multi_probe_level=1)
    search_params = dict(checks=50)

    # Initialize FLANN Matcher
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Detect Matches
    match = flann.knnMatch(des_last, des, k=2)

    return match

def stream_rgb(image):
    cv2.imshow("RGB Stream", image)
    cv2.waitKey(1)

def detect_features(image, num):
    # Initialize ORB Dectector
    orb = cv2.ORB_create(nfeatures=num)

    # Detect and Compute
    kp, des = orb.detectAndCompute(image, None)

    return kp, des

def stream_features(image, kp):
    # Create Image
    visual = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0), flags=0)

    # Draw Images
    cv2.imshow("Features", visual)
    cv2.waitKey(1)
def barfoot_solve(valid_points, valid_points_prev):
    Top_=np.eye(4)
    feature=FeaturePoints()
    # sample = random.sample(range(len(valid_points)),10)
    # temp1=[]
    # temp2=[]
    # for i in sample:
    #     temp1.append(valid_points[i])
    #     temp2.append(valid_points_prev[i])
    # valid_points_prev=np.array(temp2)
    # valid_points=np.array(temp1)
        
    if len(valid_points) > 5:
        P = np.average(valid_points_prev, axis=0)
        
        Y = np.average(valid_points,axis=0)
        I = 0
        for j in range(len(valid_points_prev)):
            pint0 = valid_points_prev[j,:] - P
            I += feature.SO3.wedge(pint0)@feature.SO3.wedge(pint0)
        I=-I/len(valid_points_prev)
        
        M1 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((feature.SO3.wedge(P),np.eye(3)))))
        M2 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((np.zeros([3,3]),I))))
        M3 = np.vstack((np.hstack((np.eye(3), -feature.SO3.wedge(P))), np.hstack((np.zeros([3,3]),np.eye(3)))))
        M=M1@M2@M3
        algopt=None
        counter=0
        while (algopt is None or np.linalg.norm(algopt)>1e-12) and counter<100:    
            algopt = feature.barfoot_solve(Top_, valid_points_prev, valid_points, M)
            Top_ = feature.SE3.exp(feature.SE3.wedge(algopt))@Top_
            counter += 1
        #print('converged after ',counter,' iterations')
        #r = Top_[0:3, 0:3]
        #t = Top_[0:3, 3]

        #Top_[0:3, 3] = -r.T@t
        #print("algopt: ", algopt)
    else:
        print('not enough points seen')   

    
    #print('Top barfoot:',Top_)
    
    return Top_

def estimate_motion_barfoot_ransac(matches, kp_last, kp, k, pointCloud_last, pointCloud,xyz_test_prev,xyz_test):
    # Declare lists and arrays
    # rmat = np.eye(3)
    # tvec = np.zeros((3, 1))
    pxl_list_last = []
    pxl_list = []
    points = []
    valid_points = []
    valid_points_prev=[]
    valid_pxl_list = []
    valid_pxl_list_prev = []
    feature =FeaturePoints()
    # Collect feature pixel locations
    for match in matches:
        x1, y1 = kp_last[match[0].queryIdx].pt
        x2, y2 = kp[match[0].trainIdx].pt
        pxl_list_last.append([int(x1), int(y1)])
        pxl_list.append([int(x2), int(y2)])
    
   
    # Read point cloud data
    points = read_point_efficient(pointCloud, pxl_list)
    points_prev = read_point_efficient(pointCloud_last, pxl_list_last)

    # Remove pixel locations if there is no corresponding point
    for point in range(len(points)):
        if np.linalg.norm(points[point]) > 0.05 and np.linalg.norm(points[point]) <40 :
            valid_points.append(points[point])
            valid_points_prev.append(points_prev[point])
            valid_pxl_list.append(pxl_list[point])
            valid_pxl_list_prev.append(pxl_list_last[point])
    valid_points_prev=np.vstack(valid_points_prev)    
    valid_points=np.vstack(valid_points)   
    print('pre-ransac',len(valid_points))

    if xyz_test is not None:
        valid_points = xyz_test
        valid_points_prev = xyz_test_prev

    ransacing = True
    d=0
    Top_d=np.eye(4)
    counter=0
    points_hold=[]
    points_hold_prev=[]
    hold=valid_points
    print('matches',len(valid_points))
    while ransacing:
        orig_points=len(valid_points)
        sample = random.sample(range(orig_points),int(orig_points/10))
        #print('sample',sample)
        temp1=[]
        temp2=[]
        for i in sample:
            temp1.append(valid_points[i])
            temp2.append(valid_points_prev[i])
        
        ransac_Top = barfoot_solve(np.array(temp1), np.array(temp2))
        
   
        points_clean=[]
        points_clean_prev=[]

        
        for i in range(orig_points):
            y=np.append(valid_points[i],1)
            z=ransac_Top@(np.append(valid_points_prev[i],1)).T
            
            # print('observed',y)
            # print(ransac_Top)
            # print('transformed',z)
            # print('norm',np.linalg.norm(y-z))
            if np.linalg.norm(y-z) < .01:
                if len(points_clean) == 0:
                    points_clean = valid_points[i]
                    points_clean_prev = valid_points_prev[i]
                    print('y',y)
                    print('z',z)
                else:        
                    points_clean=np.vstack((points_clean,valid_points[i]))
                    points_clean_prev=np.vstack((points_clean_prev,valid_points_prev[i]))

        
        counter +=1
        if len(points_clean)>d:

            d=len(points_clean)
            points_hold=points_clean
            points_hold_prev=points_clean_prev
            if d>orig_points*2/3:
                print('pass')
                ransacing=False
        print(d)
        if counter>10:
            
            valid_points=points_hold
            valid_points_prev=points_hold_prev   
            print('removed via ransac',orig_points-len(valid_points))
            ransacing = False 

    Top_ = barfoot_solve(valid_points, valid_points_prev)
    Top_=np.linalg.inv(Top_)
    # print("Calculated Top", Top_)
    # r = Top_[0:3, 0:3]
    # t = Top_[0:3, 3]

    # Top_[0:3, 3] = -r.T@t
    # Top_[0:3, 0:3]=r
    # print("algopt: ", algopt)

    
    #print('Top barfoot:',Top_)
    
    return Top_