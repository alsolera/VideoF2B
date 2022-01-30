# -*- coding: utf-8 -*-
# CamCalibration - Calibrate a camera for use with VideoF2B.
# Copyright (C) 2018  Alberto Solera Rico - videof2b.dev@gmail.com
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
#
# https://markhedleyjones.com/projects/calibration-checkerboard-collection
# https://raw.githubusercontent.com/MarkHedleyJones/markhedleyjones.github.io/master/media/calibration-checkerboard-collection/Checkerboard-A4-25mm-10x7.pdf

'''
Module for calibrating cameras.
'''

import enum
import logging
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QCoreApplication, QObject, Signal
from PySide6.QtGui import QImage
from videof2b.core.common.path import path_to_str
from videof2b.core.imaging import cv_img_to_qimg

log = logging.getLogger(__name__)


@enum.unique
class CalibratorReturnCodes(enum.IntEnum):
    '''Definition of the return codes from CameraCalibrator's processing loop.'''
    # An exception occurred. Caller should check .exc for details.
    EXCEPTION_OCCURRED = -2
    # This is the code at init before the loop starts.
    UNDEFINED = -1
    # The loop exited normally.
    NORMAL = 0
    # User canceled the loop early.
    USER_CANCELED = 1
    # No valid frames found.
    NO_VALID_FRAMES = 2
    # Insufficient valid frames found.
    INSUFFICIENT_VALID_FRAMES = 3


class CameraCalibrator(QObject):
    '''
    Calibrates a camera.
    '''

    # Emits when a new frame of video is available for display.
    new_frame_available = Signal(QImage)
    # Emits when we send a progress update.
    progress_updated = Signal(tuple)
    # Emits when the processing loop is about to return.
    finished = Signal(int)
    # Emits when an exception occurs during processing. The calling thread should know.
    error_occurred = Signal(str, str)

    def __init__(self, path: Path, is_fisheye: bool = False) -> None:
        '''
        Create a new camera calibrator.

        :param Path path: The path to a video file where a chessboard pattern was recorded.
        :param bool is_fisheye: Flag indicating whether this is a fisheye camera calibration.
                                If not, the camera will be calibrated according to the
                                standard pinhole model.
        '''
        super().__init__()
        self.path = path
        self.is_fisheye = is_fisheye
        # Exception object, if any occurred during the main loop.
        self.exc: Exception = None
        # Return code that indicates our status when the proc loop exits.
        self.ret_code: CalibratorReturnCodes = CalibratorReturnCodes.UNDEFINED
        # Flags
        self._keep_running: bool = False
        # Video frame rate in frames per second.
        self._video_fps: float = None
        # Time per video frame (seconds per frame). Inverse of FPS. Used to avoid repetitive division.
        self._video_spf: float = None
        self.frame_idx: int = None
        self.frame_time: float = None
        self.num_input_frames: int = -1
        self.progress: int = None
        self._cap: cv2.VideoCapture = None

    def run(self):
        '''Calibrate the camera using the chessboard pattern.'''
        log.debug('Entering `CameraCalibrator.run()`')
        self._keep_running = True
        # Prep
        self.frame_idx = 0
        self.frame_time = 0.
        self.progress = 0
        self.progress_updated.emit(
            (self.frame_time,
             self.progress,
             'Starting camera calibration. This may take a while, please be patient.')
        )
        self._cap = cv2.VideoCapture(path_to_str(self.path))
        self._video_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._video_spf = 1. / self._video_fps
        self.num_input_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # Start calibration
        try:
            if self.is_fisheye:
                self._calibrate_fisheye()
            else:
                self._calibrate_standard()
        except Exception as exc:
            log.critical('An unhandled exception occurred while running CamerCalibrator.run()!')
            log.critical('Exception details follow:')
            log.critical(exc)
            self.exc = exc
            self.ret_code = CalibratorReturnCodes.EXCEPTION_OCCURRED
            self.finished.emit(self.ret_code)

    def stop(self):
        '''Cancel the calibration procedure.'''
        log.debug('Entering `CameraCalibrator.stop()`')
        self.ret_code = CalibratorReturnCodes.USER_CANCELED
        self._keep_running = False

    @staticmethod
    def _get_interval_frame_count(cap: cv2.VideoCapture, time_delta: float) -> int:
        '''
        Returns the number of frames in a video capture
        that corresponds to the given time interval.

        cap: VideoCapture instance
        time_delta: time interval, in seconds.
        '''
        return int(time_delta * cap.get(cv2.CAP_PROP_FPS))

    def _validate(self):
        '''Validate available data before performing calibration
        by examining the loop's retcode.'''
        success = True
        progress_msg = None
        # Check if loop exited early
        if self.ret_code == CalibratorReturnCodes.USER_CANCELED:
            log.info('Quitting calibration early.')
            progress_msg = 'Canceling camera calibration.'
            success = False
        if self.ret_code == CalibratorReturnCodes.NO_VALID_FRAMES:
            msg = 'No frames found for calibration. Quitting.'
            log.error(msg)
            progress_msg = f'ERROR: {msg}'
            success = False
        if self.ret_code == CalibratorReturnCodes.INSUFFICIENT_VALID_FRAMES:
            msg = 'Insufficient information for calibration. Quitting.'
            log.error(msg)
            progress_msg = f'ERROR: {msg}'
            success = False
        if not success:
            # Our thread is about to quit, so clean up appropriately.
            self._cap.release()
            self.progress_updated.emit((self.frame_time, self.progress, progress_msg))
            self.finished.emit(self.ret_code)
        return success

    def _calibrate_standard(self) -> None:
        '''Calibrate the camera as a standard (pinhole model) camera.'''
        log.info(f'Calibrating standard camera based on {self.path.name}')
        # Board dimensions
        chess_cols = 7
        chess_rows = 10
        # Minimum number of required valid frames
        min_num_used = 10
        # Termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chess_rows*chess_cols, 3), np.float32)
        objp[:, :2] = np.mgrid[0:chess_cols, 0:chess_rows].T.reshape(-1, 2)
        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.
        # number of seconds between candidate frames
        read_interval = 1.
        # number of frames to skip so that we read candidate frames at read_interval.
        frame_skip_count = CameraCalibrator._get_interval_frame_count(self._cap, read_interval)
        frame_skip_count = 9  # TODO: temporary value to match legacy behavior for comparison purposes.
        num_used = 0
        num_images = 0
        self.progress_updated.emit((self.frame_time, self.progress, 'Collecting data from input video...'))
        while self._keep_running:
            for i in range(frame_skip_count):
                _, img = self._cap.read()
                self.frame_idx += 1
                if num_images == 0:
                    sample_img = img
            if img is None:
                break
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (chess_cols, chess_rows), None)
            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                sub_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(sub_corners)
                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, (chess_cols, chess_rows), sub_corners, ret)
                self.new_frame_available.emit(cv_img_to_qimg(img))
                self.frame_time = self.frame_idx * self._video_spf
                # A full read of the input video is 60% of the whole procedure.
                self.progress = int(self.frame_idx / (self.num_input_frames) * 60)
                self.progress_updated.emit((self.frame_time, self.progress, ''))
                num_used += 1
            num_images += 1
            # Breathe, dawg
            QCoreApplication.processEvents()
        # Check for failures. UserCanceled has priority.
        if self.ret_code != CalibratorReturnCodes.USER_CANCELED:
            if 0 < num_used < min_num_used:
                self.ret_code = CalibratorReturnCodes.INSUFFICIENT_VALID_FRAMES
            if num_images == 0 or num_used == 0:
                self.ret_code = CalibratorReturnCodes.NO_VALID_FRAMES
        # Validate us before diving into calibration
        if not self._validate():
            return
        # Proceed with calibration
        log.info(f'Used: {num_used}/{num_images} images')
        log.info('Calibrating camera...')
        self.progress_updated.emit((self.frame_time, self.progress, 'Calculating camera parameters...'))
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        self._bump_progress(msg='Initial calibration complete.')
        # Optimize the cam matrix
        log.info('Getting optimal new camera matrix...')
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1)
        self._bump_progress(msg='Calibration optimized.')
        log.info('Creating calibration images...')
        out_dir_path = self.path.parent
        # Undistort
        dst = cv2.undistort(sample_img, mtx, dist, None, newcameramtx)
        out_img_nocrop = out_dir_path / 'calibresult_nocrop.png'
        cv2.imwrite(path_to_str(out_img_nocrop), dst)
        self._display_freeze_frame(dst, f'Displaying a sample frame: corrected, uncropped ({w}x{h} px)')
        # Crop the image
        x, y, w, h = roi
        dst = dst[y:y+h, x:x+w]
        out_img_crop = out_dir_path / 'calibresult.png'
        cv2.imwrite(path_to_str(out_img_crop), dst)
        self._display_freeze_frame(
            dst,
            f'Displaying a sample frame: corrected, cropped ({w}x{h} px), offset by ({x},{y}) px.'
        )
        # Calculate and report reprojection error
        tot_error = 0.
        num_objpoints = len(objpoints)
        log.debug(f'num_objpoints = {num_objpoints}')
        for i in range(num_objpoints):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(src1=imgpoints[i], src2=imgpoints2, normType=cv2.NORM_L2) / len(imgpoints2)
            tot_error += error
        mean_reproj_err = tot_error / num_objpoints
        mre_msg = f'Mean reprojection error = {mean_reproj_err:6.4f} px'
        log.debug(mre_msg)
        self.progress_updated.emit((self.frame_time, self.progress, mre_msg))
        # Write calibration to disk
        npz_name = out_dir_path / 'CamCalibration.npz'
        self._bump_progress(msg='Calibration images written to disk.')
        log.info(f'Saving calibration data to {npz_name.name}')
        np.savez(path_to_str(npz_name), mtx=mtx, dist=dist, newcameramtx=newcameramtx, roi=roi)
        log.info('Camera calibration done.')
        # Release the video stream
        self._cap.release()
        self.progress = 100
        self.progress_updated.emit((self.frame_time, self.progress, 'Camera calibration complete.'))
        self.ret_code = CalibratorReturnCodes.NORMAL
        self.finished.emit(self.ret_code)

    def _bump_progress(self, bump_interval=10, msg=''):
        '''Bump progress by the specified interval and push an optional message.'''
        self.progress += bump_interval
        self.progress_updated.emit((self.frame_time, self.progress, msg))

    def _display_freeze_frame(self, img, msg, interval=3.0):
        '''
        Display the given image for the specified time interval and push the specified message.

        :param np.ndarray img: Image frame to display.
        :param str msg: The message to emit in the progress update signal.
        :param float interval: How long to display the image, in seconds.
        '''
        self.new_frame_available.emit(cv_img_to_qimg(img))
        self.progress_updated.emit((self.frame_time, self.progress, msg))
        time.sleep(interval)

    def _calibrate_fisheye(self):
        '''
        Calibrate the camera as a fisheye camera.
        EXPERIMENTAL.

        Based on lots of sources...
            Translate from C:
                https://vovkos.github.io/doxyrest-showcase/opencv/sphinx_rtd_theme/page_tutorial_camera_calibration.html
            Decent posts (two parts):
                https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
                https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f
        '''
        # pylint: disable=no-member
        cal_dir = self.path.parent
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
        img_shape = None
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        cap = cv2.VideoCapture(path_to_str(self.path))
        # number of seconds between candidate frames
        read_interval = 1.
        # number of frames to skip so that we read candidate frames at read_interval.
        frame_skip_count = CameraCalibrator._get_interval_frame_count(cap, read_interval)
        image_1 = None
        num_used = 0
        num_images = 0

        while self._keep_running:
            for _ in range(frame_skip_count):
                _, img = cap.read()
                if num_images == 0:
                    image_1 = img
            if img is None:
                break
            if img_shape is None:
                img_shape = img.shape[:2]
            else:
                assert img_shape == img.shape[:2], "All images must share the same size."

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
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

        log.info(f'Found {num_used} of {num_images} valid images for calibration')
        log.info('Calibrating fisheye camera...')
        assert num_used == len(objpoints)
        cam_mat = np.zeros((3, 3))
        dist_coeffs = np.zeros((4, 1))
        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num_used)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(num_used)]
        _, _, _, _, _ = cv2.fisheye.calibrate(
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
        img_dims = img_shape[::-1]
        log.debug(f"img_dims = {img_dims}")
        log.debug(f"cam_mat =\n{cam_mat}")
        log.debug(f"dist_coeffs =\n{dist_coeffs}")

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

        # This is how scaled_cam_mat, dim2 and balance are used
        # to determine final_cam_mat used to un-distort image.
        # OpenCV document failed to make this clear!
        final_cam_mat = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            scaled_cam_mat, dist_coeffs, dim2, np.eye(3), balance=balance)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            scaled_cam_mat, dist_coeffs, np.eye(3), final_cam_mat, dim3, cv2.CV_16SC2)
        undistorted_img = cv2.remap(
            img, map1, map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT)

        log.info('Getting optimal new camera matrix...')
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

        log.info("Writing calibration data:")
        for k, v in data.items():
            log.info(k, v)
        np.savez(path_to_str(cal_dir / "FisheyeCal.npz"), **data)

        log.info('Creating images...')
        cv2.imwrite(path_to_str(cal_dir / 'calibresult.png'), undistorted_img)

        cv2.destroyAllWindows()
