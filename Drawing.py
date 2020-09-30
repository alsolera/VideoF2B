# VideoF2B - Draw F2B figures from video
# Copyright (C) 2018  Alberto Solera Rico - albertoavion(a)gmail.com
# Copyright (C) 2020  Andrey Vasilik - basil96@users.noreply.github.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math

import cv2
import numpy as np


def PointsInCircum(r, n=100):
    pi = math.pi
    return [(math.cos(2*pi/n*x)*r, math.sin(2*pi/n*x)*r) for x in range(0, n+1)]


def PointsInHalfCircum(r, n=100):
    pi = math.pi
    return [(math.cos(pi/n*x)*r, math.sin(pi/n*x)*r) for x in range(0, n+1)]


def draw_axis(img, rvec, tvec, cameramtx, dist):
    # unit is m
    points = np.float32([[2, 0, 0], [0, 2, 0], [0, 0, 5], [0, 0, 0]]).reshape(-1, 3)
    axisPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[0].ravel()), (255, 0, 0), 1)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[1].ravel()), (0, 255, 0), 1)
    img = cv2.line(img, tuple(axisPoints[3].ravel()), tuple(axisPoints[2].ravel()), (0, 0, 255), 1)
    return img


def draw_level(img, rvec, tvec, cameramtx, dist, r=25):
    # unit is m
    n = 100
    coords = np.asarray(PointsInCircum(r=r, n=n), np.float32)
    points = np.c_[coords, np.zeros(1+n)]
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)

    twoDPoints = twoDPoints.astype(int)

    for i in range(np.shape(twoDPoints)[0] - 1):
        img = cv2.line(
            img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), (255, 255, 255), 1)

    return img


def draw_merid(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255)):
    # unit is m
    n = 100
    pi = math.pi
    angle = angle * pi/180
    c = math.cos(angle)
    s = math.sin(angle)
    RotMatrix = [[c, s, 0],
                 [s, c, 0],
                 [0, 0, 1]]

    coords = np.asarray(PointsInHalfCircum(r=r, n=n), np.float32)
    points = np.c_[np.zeros(1+n), coords]

    points = np.matmul(points, RotMatrix)
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)

    twoDPoints = twoDPoints.astype(int)

    for i in range(np.shape(twoDPoints)[0] - 1):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)

    return img


def draw_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255)):
    # unit is m
    n = 100
    pi = math.pi
    YawAngle = angle * pi/180
    c = math.cos(YawAngle)
    s = math.sin(YawAngle)

    center = [0,
              0.85356*r,
              0.35355*r]

    TiltMatrix = [[1,       0,        0],
                  [0,  0.92388, 0.38268],
                  [0, -0.38268, 0.92388]]

    YawMatrix = [[c, -s, 0],
                 [s, c, 0],
                 [0, 0, 1]]

    rLoop = r*0.382683

    coords = np.asarray(PointsInCircum(r=rLoop, n=n), np.float32)
    points = np.c_[np.zeros(1+n), coords]
    points = np.matmul(points, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    points = np.matmul(points, TiltMatrix)+center
    points = np.matmul(points, YawMatrix)

    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)

    twoDPoints = twoDPoints.astype(int)

    for i in range(np.shape(twoDPoints)[0] - 1):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)

    return img


def draw_top_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255)):
    # unit is m
    n = 100
    pi = math.pi
    YawAngle = angle * pi/180
    c = math.cos(YawAngle)
    s = math.sin(YawAngle)

    center = [0,
              0.35355*r,
              0.85356*r]

    TiltMatrix = [[1,       0,        0],
                  [0,  0.38268, 0.92388],
                  [0, -0.92388, 0.38268]]

    YawMatrix = [[c, -s, 0],
                 [s, c, 0],
                 [0, 0, 1]]

    rLoop = r*0.382683

    coords = np.asarray(PointsInCircum(r=rLoop, n=n), np.float32)
    points = np.c_[np.zeros(1+n), coords]
    points = np.matmul(points, [[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    points = np.matmul(points, TiltMatrix)+center
    points = np.matmul(points, YawMatrix)

    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)

    twoDPoints = twoDPoints.astype(int)

    for i in range(np.shape(twoDPoints)[0] - 1):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)

    return img


def drawHorEight(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255)):
    draw_loop(img, angle+24.47, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255))
    draw_loop(img, angle-24.47, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255))

    return img


def drawVerEight(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255)):
    draw_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255))
    draw_top_loop(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255))

    return img


def drawOverheadEight(img, angle, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255)):
    draw_top_loop(img, angle+90, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255))
    draw_top_loop(img, angle-90, rvec, tvec, cameramtx, dist, r, color=(255, 255, 255))

    return img


def draw_all_merid(img, rvec, tvec, cameramtx, dist, r, offsettAngle):

    for angle in range(0, 180, 45):
        gray = 255-angle*2
        color = (gray, gray, gray)
        draw_merid(img, angle + offsettAngle, rvec, tvec, cameramtx, dist, r, color)

    return img


def draw_45(img, rvec, tvec, cameramtx, dist, r=20, color=(255, 255, 255)):
    # unit is m
    n = 100
    pi = math.pi
    r45 = math.cos(pi/4) * r
    coords = np.asarray(PointsInCircum(r=r45, n=n), np.float32)
    points = np.c_[coords, np.ones(1+n)*r45]
    twoDPoints, _ = cv2.projectPoints(points, rvec, tvec, cameramtx, dist)

    twoDPoints = twoDPoints.astype(int)

    for i in range(np.shape(twoDPoints)[0] - 1):
        img = cv2.line(img, tuple(twoDPoints[i].ravel()), tuple(twoDPoints[i+1].ravel()), color, 1)

    return img


def draw_base_tol(img, cam, dist, R=21.):
    '''Draw the upper & lower limits of base flight envelope. Nominally these are 0.30m above and below the equator.'''
    n = 200
    tol = 0.3
    r = math.sqrt(R**2 - tol**2)
    coords = np.asarray(PointsInCircum(r, n))
    pts_lower = np.c_[coords, np.ones(1 + n) * (-tol)]
    pts_upper = np.c_[coords, np.ones(1 + n) * tol]
    img_pts_lower = cv2.projectPoints(pts_lower, cam.rvec, cam.tvec,
                                      cam.newcameramtx, dist)[0].astype(int)
    img_pts_upper = cv2.projectPoints(pts_upper, cam.rvec, cam.tvec,
                                      cam.newcameramtx, dist)[0].astype(int)
    color = (204, 204, 204)
    # Draw dashed lines
    num_on = 3
    num_off = 2
    counter = 0
    is_visible = True
    for i in range(n):
        counter += 1
        if is_visible:
            img = cv2.line(img,
                           tuple(img_pts_lower[i].ravel()),
                           tuple(img_pts_lower[i + 1].ravel()), color, 1)
            img = cv2.line(img,
                           tuple(img_pts_upper[i].ravel()),
                           tuple(img_pts_upper[i + 1].ravel()), color, 1)
            if counter == num_on:
                counter = 0
                is_visible = False
        else:
            if counter == num_off:
                counter = 0
                is_visible = True
    return img


def draw_edge(img, cam, dist, R):
    '''Draw the edge outline of the flight sphere as seen from camera's perspective.'''
    # normal vector of circle: points from sphere center to camera
    d = np.linalg.norm(cam.cam_pos)
    n = (cam.cam_pos / d).reshape((3,))
    phi = np.arctan2(n[1], n[0])
    rot_mat = np.array([
        [np.cos(phi), -np.sin(phi), 0.],
        [np.sin(phi), np.cos(phi), 0.],
        [0., 0., 1.]])
    u = np.array([0., 1., 0.])
    u = rot_mat.dot(u)
    v = np.cross(n, u)
    t = np.linspace(0., 1., 100)
    k = np.pi  # semi-circle
    c = (R**2 / d) * n
    det = d**2 - R**2
    if det < 0.:
        # guard against negative arg of sqrt
        r = 0.0
    else:
        r = R / d * np.sqrt(det)
    world_pts = np.array([c + r * (np.cos(k * t_i) * u + np.sin(k * t_i) * v) for t_i in t])
    img_pts, _ = cv2.projectPoints(world_pts, cam.rvec, cam.tvec, cam.newcameramtx, dist)
    img_pts = img_pts.astype(int)
    for i in range(img_pts.shape[0] - 1):
        img = cv2.line(
            img, tuple(img_pts[i].ravel()), tuple(img_pts[i+1].ravel()), (255, 0, 255), 1)
    return img


def draw_points(img, cam, dist):
    r = cam.flightRadius
    rcos45 = cam.markRadius * 0.70710678
    marker_size_x = 0.20  # marker width, in m
    marker_size_z = 0.60  # marker height, in m
    world_points = np.array([
        # Points on sphere centerline: sphere center, pilot's feet, top of sphere.
        [0, 0, 0],
        [0, 0, -cam.markHeight],
        [0, 0, r],
        # Points on equator: bottom of right & left marker, right & left antipodes, front & rear antipodes
        [rcos45, rcos45, 0],
        [-rcos45, rcos45, 0],
        [cam.markRadius, 0, 0],
        [-cam.markRadius, 0, 0],
        [0, -cam.markRadius, 0],
        [0, cam.markRadius, 0],
        # Points on corners of an imaginary marker at center of sphere (optional)
        [0.5 * marker_size_x, 0., 0.5 * marker_size_z],
        [-0.5 * marker_size_x, 0., 0.5 * marker_size_z],
        [-0.5 * marker_size_x, 0., -0.5 * marker_size_z],
        [0.5 * marker_size_x, 0., -0.5 * marker_size_z],
        # Points on corners of front marker
        [0.5 * marker_size_x, cam.markRadius, 0.5 * marker_size_z],
        [-0.5 * marker_size_x, cam.markRadius, 0.5 * marker_size_z],
        [-0.5 * marker_size_x, cam.markRadius, -0.5 * marker_size_z],
        [0.5 * marker_size_x, cam.markRadius, -0.5 * marker_size_z],
        # Points on corners of right marker
        [rcos45 + 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
        [rcos45 - 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
        [rcos45 - 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
        [rcos45 + 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
        # Points on corners of left marker
        [-rcos45 + 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
        [-rcos45 - 0.5 * marker_size_x, rcos45, 0.5 * marker_size_z],
        [-rcos45 - 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
        [-rcos45 + 0.5 * marker_size_x, rcos45, -0.5 * marker_size_z],
    ],
        dtype=np.float32)

    imgpts, _ = cv2.projectPoints(world_points, cam.rvec, cam.tvec, cam.newcameramtx, dist)
    imgpts = np.int32(imgpts).reshape(-1, 2)

    # Draw the world points in the video according to our color scheme
    for i, img_pt in enumerate(imgpts):
        if i < 9:
            # RED: points on centerline and equator
            pt_color = (0, 0, 255)
        elif 8 < i < 13:
            # CYAN: corners of imaginary marker at center of sphere
            pt_color = (255, 255, 0)
            # print(f'center marker: i={i}, img_pt={img_pt}')
        else:
            # GREEN: corners of the three outside markers
            pt_color = (0, 255, 0)
            # print(f'perimeter markers: i={i}, img_pt={img_pt}')
        cv2.circle(img, (img_pt[0], img_pt[1]), 1, pt_color, -1)

    return img


def draw_all_geometry(img, cam, offsettAngle=0, axis=False):

    distZero = np.zeros_like(cam.dist)

    draw_level(img, cam.rvec, cam.tvec, cam.newcameramtx, distZero, cam.flightRadius)
    draw_all_merid(img, cam.rvec, cam.tvec, cam.newcameramtx,
                   distZero, cam.flightRadius, offsettAngle)
    if axis:
        draw_axis(img, cam.rvec, cam.tvec, cam.newcameramtx, distZero)
    draw_45(img, cam.rvec, cam.tvec, cam.newcameramtx,
            distZero, cam.flightRadius, color=(0, 255, 0))
    draw_base_tol(img, cam, distZero, cam.flightRadius)
    draw_edge(img, cam, distZero, cam.flightRadius)

    draw_points(img, cam, distZero)

    return img


def get_track_color(x, x_max):
    color = int(255 * float(x) / float(x_max))
    return (0, min(255, color*2), min(255, (255-color)*2))


def draw_track(img, pts_scaled, maxlen):
    # loop over the set of tracked points
    for i in range(1, len(pts_scaled)):
        # If either of the tracked points are None, draw the available point
        f1 = pts_scaled[i] is None
        f2 = pts_scaled[i - 1] is None
        if f1 or f2:
            if not (f1 and f2):
                pt = pts_scaled[i-1] if f1 else pts_scaled[i]
                cv2.circle(img, pt, 1, get_track_color(i, maxlen), -1)
            continue
        # draw the lines
        cv2.line(img, pts_scaled[i - 1], pts_scaled[i], get_track_color(i, maxlen), 1)
    return img
