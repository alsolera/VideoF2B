# -*- coding: utf-8 -*-
# CamCalibration - Calibrate a camera for use with VideoF2B.
# Copyright (C) 2018  Alberto Solera Rico - albertoavion(a)gmail.com
# Copyright (C) 2020 - 2021  Andrey Vasilik - basil96@users.noreply.github.com
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
#
# https://markhedleyjones.com/storage/checkerboards/Checkerboard-A4-25mm-10x7.pdf

import os
import tkinter as Tkinter
from tkinter import filedialog as tkFileDialog

import cv2
import numpy as np

root = Tkinter.Tk()
root.withdraw()  # use to hide tkinter window
Path = tkFileDialog.askopenfilename(parent=root, initialdir='./', title='Select video')
print(Path)

width = 7
height = 10

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((height*width, 3), np.float32)
objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

cap = cv2.VideoCapture(Path)

num_used = 0
num_images = 0

while True:

    for i in range(1, 10):
        _, img = cap.read()
        if num_images == 0:
            image_1 = img

    if img is None:
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (width, height), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (width, height), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(100)
        num_used += 1
    num_images += 1

cv2.destroyAllWindows()

print(f'used: {num_used}/{num_images} images')

print('calibrating camera...')
_, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print('getting camera matrix...')
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)

print('creating images...')
out_dir = os.path.dirname(Path)
# undistort
dst = cv2.undistort(image_1, mtx, dist, None, newcameramtx)
cv2.imwrite(os.path.join(out_dir, 'calibresult_nocrop.png'), dst)
# cv2.imshow('calibresult no crop',dst)
# cv2.waitKey(1000)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv2.imwrite(os.path.join(out_dir, 'calibresult.png'), dst)

print('saving calibration data...')
fname = os.path.join(out_dir, 'CamCalibration')
np.savez(fname, mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi)

print('done.')
