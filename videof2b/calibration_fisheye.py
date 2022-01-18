# -*- coding: utf-8 -*-
# calibration_fisheye - Calibrate a camera using the fisheye model.
# Copyright (C) 2020 - 2021  Andrey Vasilik - basil96
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

import json
import os
import tkinter as Tkinter
from tkinter import filedialog as tkFileDialog

import cv2
import numpy as np

try:
    test = cv2.fisheye
except AttributeError as lib_err:
    raise AssertionError(
        'The cv2.fisheye module could not be found. Make sure you are using opencv version >= 3.x')


def calibrate_fisheye(path):
    '''
    Based on lots of sources...
        Translate from C:
            https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_camera_calibration.html
        Decent posts (two parts):
            https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
            https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
    '''
    cal_dir = os.path.dirname(path)
    chessboard_dim = (7, 10)
    subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    calibration_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    chessboard_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        # Fast check erroneously fails with high distortions like fisheye. Use for non-fisheye lenses.
        # cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    calibration_flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
        cv2.fisheye.CALIB_CHECK_COND +
        cv2.fisheye.CALIB_FIX_SKEW
    )
    objp = np.zeros((1, chessboard_dim[0]*chessboard_dim[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[:chessboard_dim[0], :chessboard_dim[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    cap = cv2.VideoCapture(path)
    # number of seconds between candidate frames
    read_interval = 1.
    # number of frames to skip so that we read candidate frames at read_interval.
    frame_skip_count = int(read_interval * cap.get(cv2.CAP_PROP_FPS))
    image_1 = None
    num_used = 0
    num_images = 0

    while True:
        for i in range(frame_skip_count):
            _, img = cap.read()
            if num_images == 0:
                image_1 = img
        if img is None:
            break
        if _img_shape is None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(
            gray,
            chessboard_dim,
            chessboard_flags
        )
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            refined_corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
            imgpoints.append(refined_corners)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, chessboard_dim, refined_corners, ret)
            cv2.imshow('cal1', img)
            cv2.waitKey(1)
            num_used += 1
        num_images += 1

    print(f'Found {num_used} of {num_images} valid images for calibration')
    print('Calibrating fisheye camera...')
    assert num_used == len(objpoints)
    cam_mat = np.zeros((3, 3))
    dist_coeffs = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num_used)]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num_used)]
    rms, _, _, _, _ = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        cam_mat,
        dist_coeffs,
        rvecs,
        tvecs,
        calibration_flags,
        calibration_criteria
    )
    img_dims = _img_shape[::-1]
    print(f"img_dims = {img_dims}")
    print(f"cam_mat =\n{cam_mat}")
    print(f"dist_coeffs =\n{dist_coeffs}")

    # Undistort a sample frame
    balance = 0.0
    dim2 = None
    dim3 = None
    img = image_1
    dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    if dim1[0] / dim1[1] != img_dims[0] / img_dims[1]:
        raise AssertionError(
            "Image to undistort must match aspect ratio of calibration images")
    if not dim2:
        dim2 = dim1
    if not dim3:
        dim3 = dim1
    # The values of K is to scale with image dimension.
    scaled_cam_mat = cam_mat * dim1[0] / img_dims[0]
    scaled_cam_mat[2][2] = 1.0  # Except that K[2][2] is always 1.0

    # This is how scaled_cam_mat, dim2 and balance are used to determine final_cam_mat used to un-distort image. OpenCV document failed to make this clear!
    final_cam_mat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        scaled_cam_mat, dist_coeffs, dim2, np.eye(3), balance=balance)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        scaled_cam_mat, dist_coeffs, np.eye(3), final_cam_mat, dim3, cv2.CV_16SC2)
    undistorted_img = cv2.remap(
        img, map1, map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT)

    print('getting camera matrix...')
    w = int(cap.get(3))
    h = int(cap.get(4))
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        final_cam_mat, dist_coeffs, (w, h), alpha=balance)

    data = dict(
        # TODO: we may or may not need all of these..
        is_fisheye=True,
        dim1=dim1,
        dim2=dim2,
        dim3=dim3,
        cam_mat=cam_mat,
        dist_coeffs=dist_coeffs,
        # final_cam_mat=final_cam_mat,
        scaled_cam_mat=scaled_cam_mat,
        balance=balance,
        mtx=final_cam_mat,
        dist=dist_coeffs,
        newcameramtx=newcameramtx,
        roi=roi
    )

    print("Writing calibration data:")
    for k, v in data.items():
        print(k, v)
    np.savez(os.path.join(cal_dir, "FisheyeCal.npz"), **data)

    print('creating images...')
    cv2.imwrite(os.path.join(cal_dir, 'calibresult.png'), undistorted_img)

    cv2.destroyAllWindows()


def main():
    root = Tkinter.Tk()
    root.withdraw()  # use to hide tkinter window
    path = tkFileDialog.askopenfilename(parent=root, initialdir='./', title='Select video')
    print(f'input calibration file: {path}')
    calibrate_fisheye(path)
    print('done.')


if __name__ == '__main__':
    main()
