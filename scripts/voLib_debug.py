#!/bin/env python3

# Helper functions for run.py

import cv2
import numpy as np
import array
from sensor_msgs_py import point_cloud2
from  feature_points_pnpransac_debug import FeaturePoints
import random
from math import isinf
import readPointCloud2

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
        if points[point, 2] > 0.05 and points[point, 2] < 20 :
        # if np.linalg.norm(points[point]) > 0.05 and np.linalg.norm(points[point]) < 20 :

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

    #----------------------------
    #Create weighted factor for the observed "y" points
    weight = []
    for i in range(len(valid_points)):
        #further away the object, the less weight
        w = 1/valid_points[i, 2]
        weight.append(w)

        # print("pts", valid_points[i])
        # print("Weight: ", w)

    weight = np.array(weight)
    #----------------------------
 
    if len(valid_points) > 5:
        print('success')

        wg = np.sum(weight)
        P = np.average(valid_points_prev, axis=0, weights=weight)
        Y = np.average(valid_points,axis=0, weights=weight)
        # print(valid_points_prev[:10,:])
        # Pexp_dim = np.average(valid_points_prev[:10,:], axis=0)
        # print(Pexp_dim)
        
        I = 0
        for j in range(len(valid_points_prev)):
            pint0 = valid_points_prev[j,:] - P
            I += weight[j]*feature.SO3.wedge(pint0)@feature.SO3.wedge(pint0)
        I=-I/wg
        
        M1 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((feature.SO3.wedge(P),np.eye(3)))))
        M2 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((np.zeros([3,3]),I))))
        M3 = np.vstack((np.hstack((np.eye(3), -feature.SO3.wedge(P))), np.hstack((np.zeros([3,3]),np.eye(3)))))
        M = M1@M2@M3

        algopt = None
        counter = 0

        # print("y", valid_points)
        # print("p", valid_points_prev)

        while (algopt is None or np.linalg.norm(algopt)>1e-10) and counter<100:    
            algopt = feature.barfoot_solve(Top_, valid_points_prev, valid_points, M, weight)
            Top_ = feature.SE3.exp(feature.SE3.wedge(algopt))@Top_
            counter += 1
        #print('converged after ',counter,' iterations')
        r = Top_[0:3, 0:3]
        t = Top_[0:3, 3]
        Top_[0:3,0:3]=r.T
        Top_[0:3, 3] = -r.T@t

        # print('algopt',feature.SE3.vee(feature.SE3.log(Top_)))
        p_test = np.array([1,2,3,1])
        # print('T_op @ p_test: ', Top_ @ p_test)
        #print("algopt: ", algopt)
    else:
        print('fail')   

    
    # print('Top barfoot:',Top_)
    
    return Top_

def read_point_efficient(pointCloud, pxl_list):
    # Define points to use
    select_data = array.array('B', [])
    for x, y in pxl_list:
        start_idx = x * pointCloud.point_step + y * pointCloud.row_step
        select_data += pointCloud.data[start_idx:(start_idx + pointCloud.point_step)]
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

    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    matchesMask = mask.ravel().tolist()
    good=[]
    
    for i in matchesMask:
        if matchesMask[i]:
            good.append(matches[i])
    # print(len(good))
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
    visual = cv2.drawMatchesKnn(imageFrame_last, kp_last, imageFrame, kp, matches, None, **draw_params)

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

    #----------------------------
    #Create weighted factor for the observed "y" points
    # print("in barfoot_solve, valid_point: \n", valid_points)
    weight = 1/valid_points[:, 2]
    # weight = np.ones(len(valid_points))
    # print("in barfoot_solve, weight: \n", weight)
    #----------------------------
    
        
    if len(valid_points) > 5:

        wg = np.sum(weight)
        P = np.average(valid_points_prev, axis=0, weights=weight)
        I = 0
        for j in range(len(valid_points_prev)):
            pint0 = valid_points_prev[j,:] - P
            I += weight[j] * feature.SO3.wedge(pint0)@feature.SO3.wedge(pint0)
        I=-I/wg
        
        M1 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((feature.SO3.wedge(P),np.eye(3)))))
        M2 = np.vstack((np.hstack((np.eye(3), np.zeros([3,3]))), np.hstack((np.zeros([3,3]),I))))
        M3 = np.vstack((np.hstack((np.eye(3), -feature.SO3.wedge(P))), np.hstack((np.zeros([3,3]),np.eye(3)))))
        M=M1@M2@M3

        #Barfoot iterations process
        algopt=None
        counter=0
        while (algopt is None or np.linalg.norm(algopt)>1e-10) and counter<18:    
            algopt = feature.barfoot_solve(Top_, valid_points_prev, valid_points, M, weight)
            Top_ = feature.SE3.exp(feature.SE3.wedge(algopt))@Top_
            counter += 1
        #print('converged after ',counter,' iterations')
        #r = Top_[0:3, 0:3]
        #t = Top_[0:3, 3]

        #Top_[0:3, 3] = -r.T@t
        #print("algopt: ", algopt)
    else:
        print('not enough points seen')
    
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
    # feature =FeaturePoints()
    
    # Collect feature pixel locations
    for match in matches:
        x1, y1 = kp_last[match[0].queryIdx].pt
        x2, y2 = kp[match[0].trainIdx].pt
        pxl_list_last.append([int(x1), int(y1)])
        pxl_list.append([int(x2), int(y2)])

    if xyz_test is not None:
        points = xyz_test
        points_prev = xyz_test_prev
    else:
        # # Read point cloud data
        points = read_point_efficient(pointCloud, pxl_list)
        points_prev = read_point_efficient(pointCloud_last, pxl_list_last)

        # points = np.array(list(readPointCloud2.read_points(pointCloud, 
        #                                                     field_names=None, 
        #                                                     skip_nans=True, 
        #                                                     uvs=pxl_list)))
        
        # points_prev = np.array(list(readPointCloud2.read_points(pointCloud_last, 
        #                                                         field_names=None, 
        #                                                         skip_nans=True, 
        #                                                         uvs=pxl_list_last)))

                                
        # print("points: \n", points[0:10])
        # print("pcd: \n", pcd_as_numpy_array[0:10])

    # Remove pixel locations if there is no corresponding point
    for point in range(len(points)):
        if np.linalg.norm(points[point]) > 0 and np.linalg.norm(points[point]) < 20:
        # if np.linalg.norm(points[point, 2]) > 0.03 and np.linalg.norm(points[point, 2]) < 20:
            valid_points.append(points[point])
            valid_points_prev.append(points_prev[point])

            
            if xyz_test is None:
                valid_pxl_list.append(pxl_list[point])
                valid_pxl_list_prev.append(pxl_list_last[point])

    valid_points_prev = np.vstack(valid_points_prev)    
    valid_points = np.vstack(valid_points)   
    
    # print("valid point: \n", valid_points)

    # print('pre-ransac',len(valid_points))
    Top_ = np.eye(4)
    ransacing = True
    bestError = 1E4
    counter=0
    points_hold=valid_points
    points_hold_prev=valid_points_prev
    orig_points=len(valid_points)
    d = orig_points * 0.6
    print('matches',len(valid_points))

    while ransacing and len(valid_points) > 20:

        
        sample = random.sample(range(orig_points), 20)
        # print('sample',sample)
        # temp1=[]
        # temp2=[]
        # for i in sample:
        #     temp1.append(valid_points[i])
        #     temp2.append(valid_points_prev[i])
        
        ransac_Top = barfoot_solve(valid_points[sample], valid_points_prev[sample])
        
        thisError = 0
        points_clean=[]
        points_clean_prev=[]

        # print("In RANSAC, validpoint: \n", valid_points)

        for i in range(orig_points):
            y = np.append(valid_points[i],1)
            z = ransac_Top@(np.append(valid_points_prev[i],1)).T
            
            e = np.linalg.norm(y-z)
            thisError += e
            # print('observed',y)
            # print("ransac T: \n",ransac_Top)
            # print('transformed',z)
            # print('norm',np.linalg.norm(y-z))
            if e < 0.07:
                if len(points_clean) == 0:
                    points_clean = valid_points[i]
                    points_clean_prev = valid_points_prev[i]
                    # print('y',y)
                    # print('z',z)
                else:        
                    points_clean = np.vstack((points_clean,valid_points[i]))
                    points_clean_prev = np.vstack((points_clean_prev,valid_points_prev[i]))



        # counter +=1
        # if len(points_clean)>d:

        #     d=len(points_clean)
        #     points_hold=points_clean
        #     points_hold_prev=points_clean_prev

        #     if d>orig_points*0.7:
        #         valid_points=points_hold
        #         valid_points_prev=points_hold_prev  
        #         Top_ = barfoot_solve(valid_points, valid_points_prev)
        #         print('pass')
        #         ransacing=False
                

        # # print("inliers: " , d)
        # if counter>10:
            
        #     valid_points=points_hold
        #     valid_points_prev=points_hold_prev  
        #     Top_ = barfoot_solve(valid_points, valid_points_prev) 
        #     print('removed via ransac',orig_points-len(valid_points))
        #     ransacing = False 
        
        
        #-----------------------------------
        """
        Dit RANSAC
        """

        if len(points_clean) > d:
            print('pass')

            if thisError < bestError:
                print('better data found!')
                points_hold = points_clean
                points_hold_prev = points_clean_prev

                bestError = thisError

        counter +=1
        if counter > 2:
            Top_ = barfoot_solve(points_hold, points_hold_prev)
            ransacing = False 

        #-----------------------------------


    
    # Top_=np.linalg.inv(Top_)
    print("Final Top: \n", Top_)
    # r = Top_[0:3, 0:3]
    # t = Top_[0:3, 3]
    # Top_[0:3, 3] = -r.T@t
    # Top_[0:3, 0:3] = r
    # print("algopt: ", algopt)

    
    #print('Top barfoot:',Top_)
    
    return Top_

def quaternion_multiply(Q0,Q1):
    """
    Multiplies two quaternions.
 
    Input
    :param Q0: A 4 element array containing the first quaternion (q01,q11,q21,q31) 
    :param Q1: A 4 element array containing the second quaternion (q02,q12,q22,q32) 
 
    Output
    :return: A 4 element array containing the final quaternion (q03,q13,q23,q33) 
 
    """


    # Extract the values from Q0
    w0 = Q0[3]
    x0 = Q0[0]
    y0 = Q0[1]
    z0 = Q0[2]
     
    # Extract the values from Q1
    w1 = Q1[3]
    x1 = Q1[0]
    y1 = Q1[1]
    z1 = Q1[2]
     
    # Computer the product of the two quaternions, term by term
    Q0Q1_w = w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1
    Q0Q1_x = w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1
    Q0Q1_y = w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1
    Q0Q1_z = w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1
     
    # Create a 4 element array containing the final quaternion
    final_quaternion = np.array([Q0Q1_x, Q0Q1_y, Q0Q1_z, Q0Q1_w])
     
    # Return a 4 element array containing the final quaternion (q02,q12,q22,q32) 
    return final_quaternion

def matrix_from_quaternion(Q):
    """
    Covert a quaternion into a full three-dimensional rotation matrix.
 
    Input
    :param Q: A 4 element array representing the quaternion (w,x,y,z) 
 
    Output
    :return: A 3x3 element matrix representing the full 3D rotation matrix. 
             This rotation matrix converts a point in the local reference 
             frame to a point in the global reference frame.
    """
    # Extract the values from Q
    q0 = Q[3]
    q1 = Q[0]
    q2 = Q[1]
    q3 = Q[2]
     
    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)
     
    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)
     
    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1
     
    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])
                            
    return rot_matrix