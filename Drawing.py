##    VideoF2B v0.4 - Draw F2B figures from video
##    Copyright (C) 2018  Alberto Solera Rico - albertoavion(a)gmail.com
##
##    This program is free software: you can redistribute it and/or modify
##    it under the terms of the GNU General Public License as published by
##    the Free Software Foundation, either version 3 of the License, or
##    (at your option) any later version.
##
##    This program is distributed in the hope that it will be useful,
##    but WITHOUT ANY WARRANTY; without even the implied warranty of
##    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
##    GNU General Public License for more details.
##
##    You should have received a copy of the GNU General Public License
##    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import math
import cv2
import numpy as np

def PointsInCircum(r,n=100):
    pi = math.pi
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in range(0,n+1)]

def PointsInHalfCircum(r,n=100):
    pi = math.pi
    return [(math.cos(pi/n*x)*r,math.sin(pi/n*x)*r) for x in range(0,n+1)]

def draw_axis(img, rvec, tvec, cameramtx, dist):
    # unit is m
    points = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, 5], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255,0,0), 1)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0,255,0), 1)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0,0,255), 1)
    return img

def draw_level(img, rvec, tvec, cameramtx, dist, r=25):
    # unit is m
    n=100
    coords = np.asarray(PointsInCircum(r=r, n=n), np.float32)
    points = np.c_[ coords, np.zeros(1+n) ]
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
    
    twoDPoints = twoDPoints.astype(int)
    
    for i in range(np.shape(twoDPoints)[0] - 1 ):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), (255,255,255), 1)
        
    return img

def draw_merid(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255)):
    # unit is m
    n=100
    pi = math.pi
    angle = angle *pi/180
    c = math.cos(angle)
    s = math.sin(angle)
    RotMatrix = [[c, s, 0],
                 [s, c, 0],
                 [0, 0, 1]]
    
    coords = np.asarray(PointsInHalfCircum(r=r, n=n), np.float32)
    points = np.c_[ np.zeros(1+n), coords ]
    
    points = np.matmul(points, RotMatrix)
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
    
    twoDPoints = twoDPoints.astype(int)
    
    for i in range(np.shape(twoDPoints)[0] - 1 ):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)
        
    return img

def draw_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255)):
    # unit is m
    n=100
    pi = math.pi
    YawAngle = angle *pi/180
    c = math.cos(YawAngle)
    s = math.sin(YawAngle)
    
    center = [0,
              0.85356*r,
              0.35355*r]
    
    TiltMatrix = [[1,       0,        0],
                  [0,  0.92388, 0.38268],
                  [0, -0.38268, 0.92388]]
    
    YawMatrix = [[c,-s,0],
                 [s,c,0],
                 [0,0,1]]
    
    rLoop = r*0.382683
    
    coords = np.asarray(PointsInCircum(r=rLoop, n=n), np.float32)
    points = np.c_[ np.zeros(1+n), coords ]
    points = np.matmul(points, [[0,-1,0],[1,0,0],[0,0,1]])
    points = np.matmul(points, TiltMatrix)+center
    points = np.matmul(points, YawMatrix)
    
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
    
    twoDPoints = twoDPoints.astype(int)
    
    for i in range(np.shape(twoDPoints)[0] - 1 ):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)
        
    return img

def draw_top_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255)):
    # unit is m
    n=100
    pi = math.pi
    YawAngle = angle *pi/180
    c = math.cos(YawAngle)
    s = math.sin(YawAngle)
    
    center = [0,
              0.35355*r,
              0.85356*r]
    
    TiltMatrix = [[1,       0,        0],
                  [0,  0.38268, 0.92388],
                  [0, -0.92388, 0.38268]]
    
    YawMatrix = [[c,-s,0],
                 [s,c,0],
                 [0,0,1]]
    
    rLoop = r*0.382683
    
    coords = np.asarray(PointsInCircum(r=rLoop, n=n), np.float32)
    points = np.c_[ np.zeros(1+n), coords ]
    points = np.matmul(points, [[0,-1,0],[1,0,0],[0,0,1]])
    points = np.matmul(points, TiltMatrix)+center
    points = np.matmul(points, YawMatrix)
    
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
    
    twoDPoints = twoDPoints.astype(int)
    
    for i in range(np.shape(twoDPoints)[0] - 1 ):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)
        
    return img

def drawHorEight(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255)):
    draw_loop(img, angle+24.47, rvec, tvec, cameramtx, dist, r, color=(255,255,255))
    draw_loop(img, angle-24.47, rvec, tvec, cameramtx, dist, r, color=(255,255,255))
    
    return img

def drawVerEight(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255)):
    draw_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255))
    draw_top_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255))
    
    return img
    
def drawOverheadEight(img, angle, rvec, tvec, cameramtx, dist, r, color=(255,255,255)):
    draw_top_loop(img, angle+90, rvec, tvec, cameramtx, dist, r, color=(255,255,255))
    draw_top_loop(img, angle-90, rvec, tvec, cameramtx, dist, r, color=(255,255,255))
    
    return img

def draw_all_merid(img, rvec, tvec, cameramtx, dist, r, offsettAngle):
    
    for angle in range(0,180,45):
        gray = 255-angle*2
        color = (gray, gray, gray)
        draw_merid(img, angle + offsettAngle, rvec, tvec, cameramtx, dist, r, color)
    
    return img

def draw_45(img, rvec, tvec, cameramtx, dist, r=20, color=(255,255,255)):
    # unit is m
    n=100
    pi = math.pi
    r45 = math.cos(pi/4) * r
    coords = np.asarray(PointsInCircum(r=r45, n=n), np.float32)
    points = np.c_[ coords, np.ones(1+n)*r45 ]
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
    
    twoDPoints = twoDPoints.astype(int)
    
    for i in range(np.shape(twoDPoints)[0] - 1 ):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)
        
    return img

def draw_points(img, rvec, tvec, cameramtx, dist, r=25, markRadius=25):
    rcos45 = markRadius * 0.70710678
    RealPoints = np.array([[0, 0, 0],
                           [0, 0, -1.5],
                           [0, 0, r],
                           [rcos45, rcos45, 0],
                           [-rcos45, rcos45, 0],
                           [markRadius, 0, 0],
                           [-markRadius, 0, 0],
                           [0,-markRadius, 0],
                           [0, markRadius, 0]], dtype=np.float32)
    imgpts, _ = cv2.projectPoints(RealPoints, rvec, tvec, cameramtx, dist)

    imgpts = np.int32(imgpts).reshape(-1,2)
    
    for p in range(RealPoints.shape[0]):
        cv2.circle(img,(imgpts[p,0],imgpts[p,1]),1,(0,0,255),-1)
        
    return img


def draw_all_geometry(img, Camera, offsettAngle=0, axis = False):
    
    distZero = np.zeros_like(Camera.dist)
    
    draw_points(img, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght, Camera.markRadius)
    draw_level(img, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght)
    draw_all_merid(img, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght, offsettAngle)
    if axis: draw_axis(img, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero)
    draw_45(img, Camera.rvec, Camera.tvec, Camera.newcameramtx, distZero, Camera.cableLenght)
    
    return img

def draw_track(img,pts_scaled,maxlen):
    # loop over the set of tracked points
    for i in range(1, len(pts_scaled)):
        # if either of the tracked points are None, ignore them
        if pts_scaled[i - 1] is None or pts_scaled[i] is None:
            continue
        
        # draw the lines
        thickness = 1
        color = int((255*float(i))/float(maxlen))
        
        cv2.line(img, pts_scaled[i - 1], pts_scaled[i], (0,min(255,color*2),min(255,(255-color)*2)), thickness)
        
    return img


