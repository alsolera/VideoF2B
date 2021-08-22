# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
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

import logging
from os import path
from tkinter import filedialog as tkFileDialog
from tkinter import simpledialog as tkSimpleDialog

import cv2
import numpy as np
import videof2b.core.common as common
from videof2b.core.common.path import path_to_str

logger = logging.getLogger(__name__)


class CalCamera:

    PointNames = ('circle center', 'front marker', 'left marker', 'right marker')

    # TODO: notes on completing the rework in CalCamera:
    # * Move _loc_pts, flightRadius, markRadius, markHeight from here to Flight class.
    # * Use those only as inputs to the Locate() method.
    # * In fact, just pass a Flight instance as input to Locate().
    # * CalCamera does not need to store them permanently.
    # * It should only store its intrinsic/extrinsic calibration params and be able to serialize them.
    # * In fact, a Calibrate() method could be implemented here with all the functionality
    # * of CamCalibration.py so that everything is encapsulated here in one place.
    # * It only needs them to calculate the on-demand pose estimation for a given Flight.
    # * Move objectPoints, NumRefPoints, and related stuff to Flight as well. It makes more sense to define them there.

    def __init__(self, frame_size, calibrationPath,
                 flight_radius=None,
                 marker_radius=None,
                 marker_height=None,
                 marker_points=None):

        logger.info(f'calibrationPath: {calibrationPath}')
        self._loc_pts = []
        self.Calibrated = calibrationPath is not None
        self.Located = False
        self.AR = True
        self.flightRadius = flight_radius
        self.markRadius = marker_radius
        self.markHeight = marker_height
        self.frame_size = frame_size
        # Calibration default values
        self.is_fisheye = None
        self.mtx = None
        self.dist = None
        self.dist_zero = None
        self.roi = None
        self.newcameramtx = None
        self.map1 = None
        self.map2 = None

        if self.Calibrated:
            try:
                with open('cal.conf', 'w') as CF:
                    CF.write(path.dirname(calibrationPath))
            except Exception as writeErr:
                print(f'Error writing to cal.conf: {writeErr}')
            try:
                npzfile = np.load(calibrationPath)
                # is_fisheye: new in v0.6, default=False for compatibility with pre-v0.6 camera files.
                self.is_fisheye = npzfile.get('is_fisheye', False)
                self.mtx = npzfile['mtx']
                self.dist = npzfile['dist']
                self.roi = npzfile['roi']
                self.newcameramtx = npzfile['newcameramtx']
                self.dist_zero = np.zeros_like(self.dist)

                # Recalculate matrix and roi in case video from this camera was scaled after recording.
                # FIXME: I suspect this approach doesn't actually work for rescaled videos. Currently, it results in incorrectly scaled AR geometry.
                newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
                    self.mtx, self.dist, self.frame_size, 1)
                # For diagnostics only. If video was not scaled after recording, these comparisons should be exactly equal.
                # print(f'as recorded newcameramtx =\n{self.newcameramtx}')
                # print(f'scaled newcameramtx = \n{newcameramtx}')
                # print(f'as recorded roi =\n{self.roi}')
                # print(f'scaled roi =\n{roi}')
                self.newcameramtx = newcameramtx
                self.roi = roi

                # Calculate these undistortion maps just once
                self.map1, self.map2 = cv2.initUndistortRectifyMap(
                    self.mtx, self.dist, np.eye(3), self.newcameramtx, self.frame_size, cv2.CV_16SC2)

                if self.is_fisheye:
                    self.scaled_cam_mat = npzfile['scaled_cam_mat']
                    self.cam_mat = npzfile['cam_mat']
                    # 'balance' is a scalar, but numpy insists on serializing it as ndarray
                    self.balance = float(npzfile['balance'])

                self.flightRadius = self.flightRadius or tkSimpleDialog.askfloat(
                    'Input', f'Flight radius (m):', initialvalue=common.DEFAULT_FLIGHT_RADIUS)
                if self.flightRadius is None:
                    self.flightRadius = common.DEFAULT_FLIGHT_RADIUS

                self.markRadius = self.markRadius or tkSimpleDialog.askfloat(
                    'Input', f'Height markers distance to center (m):', initialvalue=common.DEFAULT_MARKER_RADIUS)
                if self.markRadius is None:
                    self.markRadius = common.DEFAULT_MARKER_RADIUS

                self.markHeight = self.markHeight or tkSimpleDialog.askfloat(
                    'Input', f'Height markers: height above center of circle (m):', initialvalue=common.DEFAULT_MARKER_HEIGHT)
                if self.markHeight is None:
                    self.markHeight = common.DEFAULT_MARKER_HEIGHT
            except:
                cal_err_str = 'Error loading calibration file'
                logger.error(cal_err_str)
                print(cal_err_str)
                self.Calibrated = False
                input("Press <ENTER> to continue without calibration...")

        logger.info(f'flight radius = {self.flightRadius} m')
        logger.info(f'  mark radius = {self.markRadius} m')
        logger.info(f'  mark height = {self.markHeight} m')
        logger.info(f'Using calibration: {"YES" if self.Calibrated else "NO"}')

    def Undistort(self, img):
        x, y, w, h = self.roi
        if self.is_fisheye:
            # img = cv2.fisheye.undistortImage(img, self.mtx, self.dist, None, self.newcameramtx) # kinda works, but something's still off...
            #
            # try the other approach
            dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
            dim2 = w, h
            dim3 = dim2
            if not dim2:
                dim2 = dim1
            if not dim3:
                dim3 = dim1
            # The values of scaled_cam_mat is to scale with image dimension.
            scaled_cam_mat = self.cam_mat * dim1[0] / 1920.  # TODO: need more generic scaling here
            scaled_cam_mat[2][2] = 1.0  # Except that mat[2][2] is always 1.0
            final_cam_mat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                scaled_cam_mat, self.dist, dim2, np.eye(3), balance=self.balance)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(
                scaled_cam_mat, self.dist, np.eye(3), final_cam_mat, dim3, cv2.CV_16SC2)
            img = cv2.remap(
                img, map1, map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)
        else:
            # img_slow = cv2.undistort(img.copy(), self.mtx, self.dist, None, self.newcameramtx)
            # # crop the image
            # img_slow = img_slow[y:y+h, x:x+w]

            # Faster method: calculate undistortion maps on init, then only call remap per frame.
            img = cv2.remap(img, self.map1, self.map2, interpolation=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT)
            # crop it
            img = img[y:y+h, x:x+w]

            # Diagnostics for testing the equality of the faster approach against the original.
            # Uncomment above and below sections to verify.
            # np.save('img_undistort_slow', img_slow)
            # np.save('img_undistort_fast', img)
            # if not np.allclose(img_slow, img):
            #     raise ArithmeticError('img_slow and img are not exactly equal!')

        return img

    def Locate(self, img):
        self.calWindowName = 'Calibration (Center, Front, Left, Right)'

        rcos45 = self.markRadius * 0.70710678
        objectPoints = np.array([[0, 0, -self.markHeight],
                                 [0, self.markRadius, 0],
                                 [-rcos45, rcos45, 0],
                                 [rcos45, rcos45, 0]], dtype=np.float32)

        self.point = 0
        NumRefPoints = np.shape(objectPoints)[0]
        self.imagePoints = np.zeros((NumRefPoints, 2), dtype=np.float32)

        cv2.namedWindow(self.calWindowName, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.calWindowName, self.CB_mouse, param=img)

        while(1):

            cv2.imshow(self.calWindowName,
                       cv2.putText(img.copy(),
                                   f'Click {self.PointNames[self.point]}', (15, 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 0, 255), 2))

            k = cv2.waitKey(1) & 0xFF

            if self.imagePoints[NumRefPoints-1, 1] > 0:
                _ret, self.rvec, self.tvec = cv2.solvePnP(objectPoints, self.imagePoints,
                                                          self.newcameramtx,
                                                          self.dist_zero,
                                                          cv2.SOLVEPNP_ITERATIVE)
                # precalculate all pieces necessary for line/sphere intersections
                self.rmat = None
                self.rmat, _ = cv2.Rodrigues(self.rvec, self.rmat)
                rmat_inv = np.linalg.inv(self.rmat)
                m_inv = np.linalg.inv(self.newcameramtx)
                self.qmat = rmat_inv.dot(m_inv)
                self.rtvec = rmat_inv.dot(self.tvec)
                # Direct result: camera location in world coordinates is where the scaling factor = 0
                self.cam_pos = -self.rtvec
                cam_d = np.linalg.norm(self.cam_pos)

                logger.info(f'imagePoints =\n{self.imagePoints}')
                logger.debug('Matrices and vectors for 3D tracking: =====================')
                logger.debug(f'm = {type(self.newcameramtx)}\n{self.newcameramtx}')
                logger.debug(f'rvec = {type(self.rvec)}\n{self.rvec}')
                logger.debug(f'tvec = {type(self.tvec)}\n{self.tvec}')
                logger.debug(f'rmat = {type(self.rmat)}\n{self.rmat}')
                logger.debug(f'rmat_inv = {type(rmat_inv)}\n{rmat_inv}')
                logger.debug(f'm_inv = {type(m_inv)}\n{m_inv}')
                logger.debug(f'qmat = {type(self.qmat)}\n{self.qmat}')
                logger.debug(f'rtvec = {type(self.rtvec)}\n{self.rtvec}')
                logger.debug('End of matrices and vectors for 3D tracking ===============')
                logger.info(f'world cam location =\n{self.cam_pos}')
                logger.info(f'world cam straight distance from sphere center = {cam_d:.3f}')

                self.Located = True
                cv2.destroyWindow(self.calWindowName)
                break

            if k == 27 or cv2.getWindowProperty(self.calWindowName, 1) < 0:
                self.AR = False
                if cv2.getWindowProperty(self.calWindowName, 1) >= 0:
                    cv2.destroyWindow(self.calWindowName)
                break

    # mouse callback function
    def CB_mouse(self, event, x, y, flags, param):
        '''NOTE: param is the image frame.'''
        # print(event, x, y, flags, param)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.imagePoints[self.point, 0], self.imagePoints[self.point, 1] = x, y
            self.point += 1
            # curr_pts_str = f'Point {self.point} entered. Points =\n{self.imagePoints}'
            # logger.info(curr_pts_str)
            # print(curr_pts_str)
            if param is not None:
                cv2.circle(param, (x, y), 6, (0, 255, 0))

        elif event == cv2.EVENT_RBUTTONDOWN:
            if self.point > 0:
                self.point -= 1
                if param is not None:
                    cv2.circle(param, (self.imagePoints[self.point, 0], self.imagePoints[self.point, 1]),
                               5, (0, 0, 255))
                self.imagePoints[self.point, 0] = 0
                self.imagePoints[self.point, 1] = 0
                # print(self.imagePoints)
