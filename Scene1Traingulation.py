#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 15:01:20 2023

@author: quentinadolphe
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D


### https://www.instructables.com/Object-Tracking-With-Opencv-and-Python-With-Just-5/
def get_points(video):
    cap=cv2.VideoCapture(video)
    vid = []
        
        
    success, img = cap.read()
    bboxes = []
    tracked = 3
    points = [[] for i in range(tracked)]
    
    for i in range(tracked):
        bboxes.append(cv2.selectROI("Multi-tracker {}".format(i+1) ,img,False))
    
    ### https://learnopencv.com/multitracker-multiple-object-tracking-using-opencv-c-python/ 
    # Create MultiTracker object
    multiTracker = cv2.legacy.MultiTracker_create()
     
    # Initialize MultiTracker
    for bbox in bboxes:
      multiTracker.add(cv2.legacy.TrackerKCF_create(), img, bbox)
    
    
    while cap.isOpened():
      success, frame = cap.read()
      if not success:
        break
     
      # get updated location of objects in subsequent frames
      success, boxes = multiTracker.update(frame)
     
      # draw tracked objects
      for i, newbox in enumerate(boxes):
        p1 = (int(newbox[0]), int(newbox[1]))
        p2 = (int(newbox[0] + newbox[2]), int(newbox[1] + newbox[3]))
        cv2.rectangle(frame, p1, p2, (225,225,0), 2, 1)
        
        #draw centriod
        pc = (int((p1[0] + p2[0])/2),  int((p1[1] + p2[1])/2))
        points[i % tracked].append(pc)
        frame = cv2.circle(frame, pc , 3, (0,0,255), 3)
     
      # show frame
      cv2.imshow('MultiTracker', frame)
     
      # quit on ESC button
      if cv2.waitKey(1) & 0xFF == 27:  # Esc pressed
        break
    
    cap.release()
    cv2.destroyAllWindows()
    return [np.array(points[0]), np.array(points[1]), np.array(points[2])]


def cam_calibrate(vid1, vid2):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*7,3), np.float32)
    objp[:,:2] = np.mgrid[0:7,0:6].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    cap = cv2.VideoCapture(vid1)
    dap = cv2.VideoCapture(vid2)
    
    found = 0
    while(found < 240):  # Here, 10 can be changed to whatever number you like to choose
        ret, img1 = cap.read() # Capture frame-by-frame

        ret, img2 = dap.read() # Capture frame-by-frame
        
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret1, corners1 = cv2.findChessboardCorners(gray1, (7,6),None)
        ret2, corners2 = cv2.findChessboardCorners(gray2, (7,6),None)

        # If found, add object points, image points (after refining them)
        if ret1 == True and ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)
 
            cv2.drawChessboardCorners(img1, (5,8), corners1, ret1)
            cv2.drawChessboardCorners(img2, (5,8), corners2, ret2)
            
 
            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)
            found += 1
        
        cv2.imshow('img', img1)
        cv2.imshow('img2', img2)
        k = cv2.waitKey(500)

    # When everything done, release the capture
    h, w, _ = img1.shape
    print(h)
    print(w)
    cap.release()
    cv2.destroyAllWindows()

    ret, mtx1, dist1, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_left, gray1.shape[::-1], None, None)
    ret, mtx2, dist2, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints_right, gray2.shape[::-1], None, None)
    
    return mtx1, dist1, mtx2, dist2, imgpoints_left, imgpoints_right, objpoints, h, w

def ster_calibrate(vid1, vid2):
    mtx1, dist1, mtx2, dist2, imgpoints_left, imgpoints_right, objpoints, height, width = cam_calibrate(vid1, vid2)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    

    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist11, CM2, dist22, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx1, dist1,
                                                                 mtx2, dist2, (width, height), criteria = criteria, flags = stereocalibration_flags)
 
    print(ret)
    return R, T, mtx1, dist1, mtx2, dist2

### https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html
def DLT(P1, P2, point1, point2):
 
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

 
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
 
    print('Triangulated point: ')
    print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]

#%%
''' 
### camera and stereo calibration
R, T, mtx1, dist1, mtx2, dist2 = ster_calibrate("leftcalib.mp4", "rightcalib.mp4")

### calibration is complete so this no longer has to be run'''

#%%

mtx1 = np.array([[1661.49, 0, 990.275],
                [0, 1674.68, 483.835],
                [0,0,1]])

mtx2 = np.array([[1635.05, 0, 1010.56],
                [0, 1656.91, 422.097],
                [0,0,1]])

R = np.array([[.224832, -0.0170742, 0.974396],
             [0.0167305, 0.999858, -0.0210836],
             [-0.974254, 0.0167761, 0.224829]])

T = np.array([[-62.7286],
             [1.74351],
             [46.1935]])


#%%
#### triangulation

#RT matrix for C1 is identity.
RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
P1 = mtx1 @ RT1 #projection matrix for C1
 
#RT matrix for C2 is the R and T obtained from stereo calibration.
RT2 = np.concatenate([R, T], axis = -1)
P2 = mtx2 @ RT2 #projection matrix for C2

#%%
### Get points from stereo pair
points1 = get_points("lefttags.mp4")
#%%
points2 = get_points("righttags.mp4")

#%%



#i = 480 # delete for loop below and set i to any value to see desired moment in time

for i in range(0,800,3):
    
    p3ds = []
    for j in range(len(points1)): #tags
        _p3d = DLT(P1, P2, points1[j][i], points2[j][i])
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)
    p3ds = np.transpose(p3ds)
    
    
     
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-10, 30)
    ax.set_ylim3d(-20, 30)
    ax.set_zlim3d(30, 80)
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
     
    
    ax.plot(p3ds[0], p3ds[1], p3ds[2])
    ax.scatter3D(p3ds[0], p3ds[1], p3ds[2], 'red')
    plt.show()
    





