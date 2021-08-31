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

import cv2
import numpy as np
from PySide6.QtCore import QObject
from videof2b.core.flight import Flight

log = logging.getLogger(__name__)


class CalCamera(QObject):
    '''Represents a real-world camera whose intrinsic and extrinsic optical properties are known.'''

    # TODO: implement a `Calibrate()`` method here with all the functionality of CamCalibration.py to encapsulate all CalCamera-related functionality in one place.

    def __init__(self, frame_size, flight: Flight):
        '''Create a CalCamera for a given image frame size and a given Flight.'''
        # We expect a properly populated Flight instance.
        self._flight = flight
        self.frame_size = frame_size
        # Calibration defaults
        self.newcameramtx = None
        self.mtx = None
        self.roi = None
        self.dist = None
        # Fisheye calibration defaults (experimental)
        self.is_fisheye = False
        self.scaled_cam_mat = None
        self.cam_mat = None
        self.balance = None
        # Runtime attributes
        self.dist_zero = None
        self.map1 = None
        self.map2 = None
        # 3D-related attributes
        self.rvec = None
        self.tvec = None
        self.rmat = None
        self.qmat = None
        self.rtvec = None
        self.cam_pos = None

        if self._flight.is_calibrated:
            try:
                self.load_calibration(self._flight.calibration_path)
            except Exception as load_err:
                log.error(f'Error loading calibration file: {load_err}')
                self._flight.is_calibrated = False
                return
            self._calc_undistortion_maps()
            # If the flight is already located, locate this camera as well.
            if self._flight.is_located:
                try:
                    self.locate(self._flight)
                except Exception as loc_err:
                    log.error(
                        'Failed to auto-locate camera even though its flight is located. This should not have happened.\n'
                        f'The problem was: {loc_err}'
                    )
                    raise loc_err

    def load_calibration(self, path):
        '''Load a camera calibration from the specified path.'''
        npzfile = np.load(path)
        # is_fisheye: new in v0.6, default=False for compatibility with pre-v0.6 camera files.
        self.is_fisheye = npzfile.get('is_fisheye', False)
        if self.is_fisheye:
            # TODO: fisheye cal is an experimental feature. Needs more testing.
            self.scaled_cam_mat = npzfile['scaled_cam_mat']
            self.cam_mat = npzfile['cam_mat']
            # 'balance' is a scalar, but numpy insists on serializing it as ndarray
            self.balance = float(npzfile['balance'])
        self.mtx = npzfile['mtx']
        self.dist = npzfile['dist']
        self.roi = npzfile['roi']
        self.newcameramtx = npzfile['newcameramtx']
        self.dist_zero = np.zeros_like(self.dist)

    def _calc_undistortion_maps(self):
        '''One-time calculation of the optimal new camera matrix.
        This is a valid calculation when the camera's POV
        does not move from frame to frame.'''
        # Recalculate matrix and roi in case video from this camera was scaled after recording.
        # FIXME: I suspect this approach doesn't actually work for rescaled videos. Currently, it results in incorrectly scaled AR geometry.
        # If `self.frame_size` is the same size as the original video recording, this call has no side effects.
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

    def undistort(self, img):
        '''Undistort a given image according to the camera's calibration.'''
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

            # TODO: create regression tests for this if possible.
            # Verify the current `undistort` behavior against the original slow one,
            # in case we ever change anything in the current approach.
            # ==================================================================================
            # Diagnostics for testing the equality of the faster approach against the original.
            # Uncomment above and below sections to verify.
            # np.save('img_undistort_slow', img_slow)
            # np.save('img_undistort_fast', img)
            # if not np.allclose(img_slow, img):
            #     raise ArithmeticError('img_slow and img are not exactly equal!')

        return img

    def locate(self, flight: Flight) -> bool:
        '''Locate a given Flight instance using this camera.'''
        obj_pts = np.float32(flight.obj_pts)
        loc_pts = np.float32(flight.loc_pts)
        ret, rvec, tvec = cv2.solvePnP(obj_pts, loc_pts,
                                       self.newcameramtx,
                                       self.dist_zero,
                                       cv2.SOLVEPNP_ITERATIVE)
        if not ret:
            log.error('Failed to locate the camera. Values returned from `cv2.solvePnP`:')
            log.error(f'rvec =\n{rvec}')
            log.error(f'tvec =\n{tvec}')
            self._flight.is_located = False
            return False
        # Precalculate all pieces necessary for line/sphere intersections.
        self.rvec = rvec
        self.tvec = tvec
        self.rmat = None
        self.rmat, _ = cv2.Rodrigues(self.rvec, self.rmat)
        rmat_inv = np.linalg.inv(self.rmat)
        m_inv = np.linalg.inv(self.newcameramtx)
        self.qmat = rmat_inv.dot(m_inv)
        self.rtvec = rmat_inv.dot(self.tvec)
        # Direct result: camera location in world coordinates
        # is where the free scaling factor = 0.
        self.cam_pos = -self.rtvec
        cam_d = np.linalg.norm(self.cam_pos)

        log.info('Located the camera successfully.')
        log.info(f'loc_pts =\n{loc_pts}')
        log.debug('Matrices and vectors for 3D tracking: =====================')
        log.debug(f'newcameramtx =\n{type(self.newcameramtx)}\n{self.newcameramtx}')
        log.debug(f'rvec =\n{type(self.rvec)}\n{self.rvec}')
        log.debug(f'tvec =\n{type(self.tvec)}\n{self.tvec}')
        log.debug(f'rmat =\n{type(self.rmat)}\n{self.rmat}')
        log.debug(f'rmat_inv =\n{type(rmat_inv)}\n{rmat_inv}')
        log.debug(f'm_inv =\n{type(m_inv)}\n{m_inv}')
        log.debug(f'qmat =\n{type(self.qmat)}\n{self.qmat}')
        log.debug(f'rtvec =\n{type(self.rtvec)}\n{self.rtvec}')
        log.debug('End of matrices and vectors for 3D tracking ===============')
        log.info(f'World cam location =\n{self.cam_pos}')
        log.info(f'World cam straight distance from sphere center = {cam_d:.3f}')
        self._flight.is_located = True
        return True
