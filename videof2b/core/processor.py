# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021  Andrey Vasilik - basil96@users.noreply.github.com
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

'''
The main flight processor in VideoF2B.
'''

import enum
import logging
import platform
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import videof2b.core.figure_tracker as figtrack
import videof2b.core.projection as projection
from imutils import resize
from imutils.video import FPS
from PySide6.QtCore import QCoreApplication, QObject, Signal
from PySide6.QtGui import QImage
from videof2b.core.camera import CalCamera
from videof2b.core.common import FigureTypes, SphereManipulations
from videof2b.core.common.store import StoreProperties
from videof2b.core.detection import Detector
from videof2b.core.drawing import Drawing
from videof2b.core.flight import Flight

log = logging.getLogger(__name__)


@enum.unique
class ProcessorReturnCodes(enum.IntEnum):
    '''Definition of the return codes from VideoProcessor's processing loop.'''
    # NOTE: Every `break` in `VideoProcessor._process()`
    # must be preceeded by an update to `self.ret_code`.
    # Any new breaks that are added due to new reasons for loop exit
    # should get a new code definition here, and `MainWindow._retcodes_msgs`
    # should be updated to include a user-friendly message for the new code.
    # =============================================================================================
    # An exception occurred. Caller should check .exc for details.
    ExceptionOccurred = -2
    # This is the code at init before the loop starts.
    Undefined = -1
    # The loop exited normally.
    Normal = 0
    # User cancelled the loop early.
    UserCanceled = 1
    # Pose estimation failed
    PoseEstimationFailed = 2
    # Too many consecutive empty frames encountered
    TooManyEmptyFrames = 3


class ProcessorSettings:
    '''Stores persistable user settings.'''
    # TODO: implement all of these in the shared Settings object and get rid of this class.
    # TODO: expose some of these as appropriate in the Tools > Options menu.
    #
    # Flag that controls the state of 3D tracking.
    perform_3d_tracking = False
    # Maximum length of the track behind the aircraft, in seconds.
    max_track_time = 15
    # Width of detector frame.
    im_width = 960
    # Sphere's XY offset delta in meters.
    sphere_xy_delta = 0.1
    # Rotation offset delta in degrees.
    sphere_rot_delta = 0.5
    # Folder where we write processed output files of live videos.
    live_videos = Path('../VideoF2B_videos')


class VideoProcessor(QObject, StoreProperties):
    '''Main video processor. Handles processing
    of a video input from start to finish.'''

    # --- Signals
    # Emits when cam locating begins.
    locating_started = Signal()
    # Emits when locator points changed during camera locating.
    locator_points_changed = Signal(tuple, str)
    # Emits when all required points have been defined during the locating procedure so that the user can confirm and continue.
    locator_points_defined = Signal()
    # Emits the availability of AR geometry. Typically corresponds to value of `flight.is_located`.
    ar_geometry_available = Signal(bool)
    # Emits when a new frame of video has been processed and is available for display.
    new_frame_available = Signal(QImage)
    # Emits when we send a progress update.
    progress_updated = Signal(tuple)
    # Emits when the processing loop is about to return.
    finished = Signal(int)
    # Emits when we finish clearing the flight track.
    track_cleared = Signal()
    # Emits when we pause/resume processing.
    paused = Signal(bool)
    # Emits when an exception occurs during processing. The calling thread should know.
    error_occurred = Signal(str, str)

    def __init__(self) -> None:
        '''Create a new video processor.'''
        super().__init__()
        # Exception object, if any occurred during the main loop.
        self.exc: Exception = None
        # Return code that indicates our status when the proc loop exits.
        self.ret_code: ProcessorReturnCodes = ProcessorReturnCodes.Undefined
        # Flags
        self._keep_processing: bool = False
        self._keep_locating: bool = False
        self._clear_track_flag: bool = False
        # We emit progress updates at this interval, in percent. Must be an integer.
        self._progress_interval: int = 5
        self.flight: Flight = None
        self.cam: CalCamera = None
        self._full_frame_size: Tuple[int, int] = None
        self._fourcc: cv2.VideoWriter_fourcc = None
        self._fps: FPS = None
        self._det_scale: float = None
        self._inp_width: int = None
        self._inp_height: int = None
        self._artist: Drawing = None
        self._frame: np.ndarray = None  # the frame currently being processed for output.
        self._frame_loc: np.ndarray = None  # the current frame during pose estimation.
        self.frame_idx: int = None
        self.frame_time: float = None
        self._video_fps: float = None
        self._is_size_restorable: bool = False
        self._detector: Detector = None
        self._azimuth: float = None
        self._sphere_offset = None
        self._data_writer = None
        self._fig_tracker = None
        self.video_name = None

    def load_flight(self, flight: Flight) -> None:
        '''Load a Flight and prepare for processing.

        :param Flight flight: a properly populated Flight instance.
        '''
        self.flight = flight
        self.flight.locator_points_changed.connect(self.on_locator_points_changed)
        self.flight.locator_points_defined.connect(self.on_locator_points_defined)
        self._full_frame_size = (
            int(self.flight.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.flight.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        self._video_fps = self.flight.cap.get(cv2.CAP_PROP_FPS)
        self.num_input_frames = int(self.flight.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_name = self.flight.video_path.stem
        self.frame_idx = 0
        # Load camera
        self.cam = CalCamera(self._full_frame_size, self.flight)
        self.ar_geometry_available.emit(self.flight.is_calibrated)
        # Determine input video size and the scale wrt detector's frame size
        self._calc_size()
        log.debug(f'detector im_width = {ProcessorSettings.im_width} px')
        log.debug(f'detector det_scale = {self._det_scale:.4f}')
        # Platform-dependent stuff
        self._fourcc = cv2.VideoWriter_fourcc(*'H264')
        if platform.system() == 'Windows':
            self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Detector
        max_track_len = int(ProcessorSettings.max_track_time * self._video_fps)
        self._detector = Detector(max_track_len, self._det_scale)
        # Drawing artist
        self._artist = Drawing(self._detector, cam=self.cam, flight=self.flight, axis=False)
        # Angle offset of current AR hemisphere wrt world coordinate system
        self._azimuth = 0.0
        # TODO: Like `self._azimuth`, `self._sphere_offset` really should be the driver for Drawing, but Drawing API uses incremental positioning right now.
        # Right now the two offsets go out of sync. The effect may be visible during (re)locating if offset is nonzero.
        self._sphere_offset = self.flight.sphere_offset
        # Misc
        self.is_paused = False

    def _prep_video_output(self, path_out):
        '''Prepare the output video file.'''
        w_ratio = self._inp_width / self._full_frame_size[0]
        h_ratio = self._inp_height / self._full_frame_size[1]
        self._is_size_restorable = w_ratio > 0.95 and h_ratio > 0.95
        log.debug(f'full frame size  = {self._full_frame_size}')
        log.debug(f'input ratios w,h = {w_ratio:.4f}, {h_ratio:.4f}')
        result = None
        if not self.flight.is_live:
            if self._is_size_restorable:
                log.info(f'Output size: {self._full_frame_size}')
                result = cv2.VideoWriter(
                    str(path_out),
                    self._fourcc, self._video_fps,
                    self._full_frame_size
                )
                # The resized width if we resize height to full size
                w_final = int(self._full_frame_size[1] / self._inp_height * self._inp_width)
                resize_kwarg = {'height': self._full_frame_size[1]}
                crop_offset = (int(0.5*(w_final - self._full_frame_size[0])), 0)
                if w_final < self._full_frame_size[0]:
                    # The resized height if we resize width to full size
                    h_final = int(self._full_frame_size[0] / self._inp_width * self._inp_height)
                    resize_kwarg = {'width': self._full_frame_size[0]}
                    crop_offset = (0, int(0.5*(h_final - self._full_frame_size[1])))
                crop_idx = (crop_offset[0] + self._full_frame_size[0],
                            crop_offset[1] + self._full_frame_size[1])
            else:
                result = cv2.VideoWriter(
                    str(path_out),
                    self._fourcc, self._video_fps,
                    (int(self._inp_width), int(self._inp_height))
                )
        else:
            # TODO: clean up this path repetition
            if not ProcessorSettings.live_videos.exists():
                ProcessorSettings.live_videos.mkdir(parents=True)
            timestr = time.strftime("%Y%m%d-%H%M")
            result = cv2.VideoWriter(
                ProcessorSettings.live_videos / f'out_{timestr}.mp4',
                self._fourcc, self._video_fps,
                (int(self._inp_width), int(self._inp_height))
            )
        # TODO: it might be cleaner to return a custom object here..
        return result, resize_kwarg, crop_offset, crop_idx

    def _prep_fig_tracking(self):
        '''Prepare processor for figure tracking, if enabled.'''
        log.info(f'3D tracking is {"ON" if ProcessorSettings.perform_3d_tracking else "OFF"}')
        if self.flight.is_calibrated and ProcessorSettings.perform_3d_tracking:
            data_path = self.flight.video_path.with_name(f'{self.video_name}_out_data.csv')
            self._data_writer = data_path.open('w', encoding='utf8')
            # self._data_writer.write('self.frame_idx,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,root1,root2\n')
            self._fig_tracker = figtrack.FigureTracker(
                callback=sys.stdout.write, enable_diags=True)  # FIXME: callback to a log file. callback=log.debug ?
        # TODO: maybe this section can be included in the above condition?
        # Class-level aids for drawing figure start/end points over track
        self._is_fig_in_progress = False
        self._fig_img_pts = []
        # Class-level aids for drawing nominal paths
        self._fit_img_pts = []
        self._draw_fit_flag = False

    def _map_img_to_sphere(self, writer):
        ''' **EXPERIMENTAL FUNCTION**
        Map the entire image space to world space, effectively
        meshing the sphere to enable the approximation of real-world
        error at a given pixel.

        :param writer: a file-like object for the projection algorithm.

        The flight must be calibrated to produce the correct results here.

        NOTE: This call is extremely time-consuming. It should only
        be performed for science.
        '''
        # Format of 4-dimensional `world_map` array:
        # Dimension 1: near/far point index
        # Dimension 2: image width
        # Dimension 3: image height
        # Dimension 4: XYZ point on sphere
        log.debug('Processing `world_map` array...')
        world_map = np.zeros((2, self._inp_width, self._inp_height, 3), dtype=np.float32)
        world_map[:, :, :, :] = np.nan
        num_pts_collected = 0
        for v in range(self._inp_height):
            t1 = time.process_time()
            for u in range(self._inp_width):
                world_pts = projection.projectImagePointToSphere(
                    self._frame,
                    self.cam, self.flight.flight_radius,
                    (u, v), writer
                )
                if world_pts is not None:
                    # log.debug(u)
                    # log.debug(world_pts)
                    world_map[:, u, v] = world_pts
                    num_pts_collected += 1
            t2 = time.process_time()
            t_diff = t2 - t1
            log.debug(
                f'Number of world points in row index {v}: {num_pts_collected}, collected in {t_diff} s')
            num_pts_collected = 0
        log.debug('Saving world_map array to file...')
        np.save('world_map.npy', world_map)
        log.debug('...done.')

    def _track_in_3d(self):
        '''Try to track the aircraft in world coordinates.'''
        if self._detector.pts_scaled[0] is not None:
            act_pts = projection.projectImagePointToSphere(
                self._frame, self.cam,
                self.flight.flight_radius, self._artist.center,
                self._detector.pts_scaled[0], self._data_writer
            )
            if act_pts is not None:  # and act_pts.shape[0] == 2:
                # self._fig_tracker.add_actual_point(act_pts)
                # Typically the first point is the "far" point on the sphere...Good enough for most figures that are on the far side of the camera.
                # TODO: Make this smarter so that we track the correct path point at all times.
                self._fig_tracker.add_actual_point(self.frame_idx, act_pts[0])
                # self._fig_tracker.add_actual_point(act_pts[1])
                if self._is_fig_in_progress:
                    self._fig_img_pts.append(self._detector.pts_scaled[0])

        # ========== Draw the fitted figures ==========
        if self._draw_fit_flag and self._num_draw_fit_frames < self._max_draw_fit_frames:
            # Draw the nominal paths: initial and best-fit
            for i, f_pts in enumerate(self._fit_img_pts):
                # Draw the initial guess (the defined nominal path) in GREEN
                # Draw the optimized fit path in CYAN
                f_color = (0, 255, 0) if i == 0 else (255, 255, 0)
                for j in range(1, len(f_pts)):
                    cv2.line(self._frame, f_pts[j], f_pts[j-1], f_color, 2)
                    # Draw the error whiskers.
                    # TODO: Verify the arrays are correct after figure.u is trimmed.
                    if i == 1:
                        cv2.line(self._frame, self._fig_img_pts[j], f_pts[j], (255, 255, 255), 1)
            self._num_draw_fit_frames += 1
        if self._num_draw_fit_frames == self._max_draw_fit_frames:
            self._draw_fit_flag = False
            self._fit_img_pts = []
            self._fig_img_pts = []
            self._num_draw_fit_frames = 0

        # Draw the start/end points of the figure's actual tracked path (as marked by user)
        if self._fig_img_pts:
            cv2.circle(self._frame, self._fig_img_pts[0], 6, (0, 255, 255), -1)
            if not self._is_fig_in_progress:
                cv2.circle(self._frame, self._fig_img_pts[-1], 6, (255, 0, 255), -1)

    def _locate(self):
        '''Interactively locate this Flight.'''
        log.debug('Entering VideoProcessor._locate()')
        log.debug(f'Asking to display the locating frame: {self._frame_loc.shape}')
        self.locating_started.emit()
        # Show the requested frame.
        self.new_frame_available.emit(self._cv_img_to_qimg(self._frame_loc))
        # Kind of a hack: force the first instruction signal.
        self.flight.on_locator_points_changed()
        self._keep_locating = True
        while self._keep_locating:
            time.sleep(0.010)
            # Breathe, dawg
            QCoreApplication.processEvents()
        log.debug(f'loc_pts after locating: {self.flight.loc_pts}')

        log.debug('Done locating the flight.')
        if not self._keep_processing:
            # The processor was requested to stop during locating.
            # Do not proceed with the rest of the processing loop.
            log.info('A request to cancel processing was sent during locating.')
            self.stop()
            return False
        # Calculate pose estimation (aka locate the camera).
        is_pose_good = self.cam.locate(self.flight)
        if not is_pose_good:
            log.error('Pose estimation failed. Cannot process video.')
            self.ret_code = ProcessorReturnCodes.PoseEstimationFailed
            return False
        # Finally, locate the artist.
        self._artist.locate(self.cam, self.flight, center=self._sphere_offset)
        # Signal the updated state of AR geometry availability.
        self.ar_geometry_available.emit(self.flight.is_located)
        log.debug('Exiting VideoProcessor._locate()')
        return True

    def stop_locating(self):
        '''Cancel the flight locating procedure.'''
        self._keep_locating = False

    def add_locator_point(self, point):
        '''Add a potential locator point.'''
        # Thin wrapper for the same method in Flight.
        if self.flight is not None:
            self.flight.add_locator_point(point)

    def pop_locator_point(self, _p):
        '''Remove the last locator point, if one exists.'''
        # Thin wrapper for the same method in Flight.
        if self.flight is not None:
            self.flight.pop_locator_point()

    def on_locator_points_changed(self, points, msg):
        '''Handles changes in locator points during the camera locating procedure.'''
        self.locator_points_changed.emit(points, msg)
        # Draw the current locator points in the locating frame.
        # Do not modify the locating frame, just a copy of it.
        img = self._frame_loc.copy()
        for p in points:
            img = cv2.circle(img, tuple(p), 6, (0, 255, 0))
        self.new_frame_available.emit(self._cv_img_to_qimg(img))

    def on_locator_points_defined(self):
        '''Locator points are completely defined. Let the world know.'''
        self.locator_points_defined.emit()

    def update_figure_state(self, figure_type: FigureTypes, val: bool) -> None:
        '''Update figure state in the drawing.'''
        if self.flight.is_located:  # protect drawing
            self._artist.figure_state[figure_type] = val

    def update_figure_diags(self, val: bool) -> None:
        '''Update figure diags state in the drawing.'''
        if self.flight.is_located:  # protect drawing
            self._artist.DrawDiags = val

    def mark_figure(self, is_start: bool) -> None:
        '''Mark the start/end of a tracked figure.

        :param is_start: True to mark the start of the figure,
                         False to mark the end.
        '''
        if self._fig_tracker is None:
            # No effect if tracker is not active
            return
        if is_start:
            # Mark the beginning of a new figure
            self._detector.clear()
            self._fig_tracker.start_figure()
            self._is_fig_in_progress = True
        else:
            # Mark the end of the current figure
            log.debug(f'before finishing figure: _fig_img_pts size ={len(self._fig_img_pts)}')
            self._fig_tracker.finish_figure()
            self._is_fig_in_progress = False
            # Uniform t along the figure path
            # t = np.linspace(0., 1., self._fig_tracker._curr_figure_fitter.num_nominal_pts)
            # The figure's chosen t distribution (initial, final)
            t = (self._fig_tracker._curr_figure_fitter.diag.u0,
                 self._fig_tracker._curr_figure_fitter.u)
            log.debug(f'figure finish: shape of u0 vs. u final: {t[0].shape, t[1].shape}')
            # Trim our detected points according to the fit
            trims = self._fig_tracker._curr_figure_fitter.diag.trim_indexes
            log.debug(f'_fig_img_pts before: size = {len(self._fig_img_pts)}')
            log.debug(f'trims: shape = {trims.shape}')
            self._fig_img_pts = list(tuple(pair)
                                     for pair in np.array(self._fig_img_pts)[trims].tolist())
            log.debug(f'_fig_img_pts after: size = {len(self._fig_img_pts)}')
            # The last item is the tuple of (initial, final) fit params
            for i, fp in enumerate(self._fig_tracker.figure_params[-1]):
                nom_pts = projection.projectSpherePointsToImage(
                    self.cam, self._fig_tracker._curr_figure_fitter.get_nom_point(*fp, *t[i]))
                self._fit_img_pts.append(tuple(map(tuple, nom_pts.tolist())))
            # log.debug(f't ({len(t)} pts)')
            log.debug(f'_fit_img_pts[initial] ({len(self._fit_img_pts[0])} pts)')
            # # log.debug(self._fit_img_pts[0])
            log.debug(f'_fit_img_pts[final] ({len(self._fit_img_pts[1])} pts)')
            # # log.debug(self._fit_img_pts[1])
            log.debug(f'_fig_img_pts ({len(self._fig_img_pts)} pts)')
            # # log.debug()
            # Set the flag for drawing the fit figures, diags, etc.
            self._draw_fit_flag = True

    def clear_track(self):
        '''Clear the aircraft's existing flight track.'''
        log.info('Clearing the flight track.')
        self._clear_track_flag = True

    def relocate(self):
        '''Relocate the flight.'''
        if not self.flight.is_located:
            return
        self.flight.is_located = False
        self.flight.is_ar_enabled = True
        self._frame_loc = self._frame
        self.flight.on_locator_points_changed()

    def pause_resume(self):
        '''Pause/Resume processing at the current frame.'''
        # Pause/Resume with ability to quit while paused
        # TODO: allow sphere rotation and movement while paused!
        self.is_paused = not self.is_paused

    def manipulate_sphere(self, command: SphereManipulations) -> None:
        '''Manipulate the AR sphere via the specified command.'''
        if not self.flight.is_located:
            return
        if command == SphereManipulations.RotateCCW:
            # Rotate sphere CCW on Z axis
            self._azimuth += ProcessorSettings.sphere_rot_delta
            self._artist.set_azimuth(self._azimuth)
        elif command == SphereManipulations.RotateCW:
            # Rotate sphere CW on Z axis
            self._azimuth -= ProcessorSettings.sphere_rot_delta
            self._artist.set_azimuth(self._azimuth)
        elif command == SphereManipulations.MoveEast:
            # Move sphere right (+X)
            log.info(
                f'User moves sphere center X by {ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.MoveCenterX(ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.MoveWest:
            # Move sphere left (-X)
            log.info(
                f'User moves sphere center X by {-ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.MoveCenterX(-ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.MoveNorth:
            # Move sphere away from camera (+Y)
            log.info(
                f'User moves sphere center Y by {ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.MoveCenterY(ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.MoveSouth:
            # Move sphere toward camera (-Y)
            log.info(
                f'User moves sphere center Y by {-ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.MoveCenterY(-ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.ResetCenter:
            # Reset sphere center to world origin
            log.info(
                f'User resets sphere center '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.ResetCenter()

    @staticmethod
    def _cv_img_to_qimg(cv_img: np.ndarray) -> QImage:
        '''Convert a cv2 image to a QImage for display in QPixmap objects.'''
        # This is an adaptation of this simple idea:
        # https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713
        #
        # One way to do it, maybe there are others:
        # https://stackoverflow.com/questions/57204782/show-an-opencv-image-with-pyqt5
        #
        # When cropping cv2 images, we end up with non-contiguous arrays.
        # First, `strides` is required:
        # https://stackoverflow.com/questions/52869400/how-to-show-image-to-pyqt-with-opencv/52869969#52869969
        # Second, it must be contiguous:
        # https://github.com/almarklein/pyelastix/issues/14
        #
        # These extra requirements manifest themselves when we process calibrated flights.
        # This can be verified by uncommenting the log messages around `np.ascontiguousarray()` call below,
        # but leave them commented for production!
        # Note that this extra step is not necessary for cv2 processing, only for converting to QImage for display.

        # TODO: profile this whole method for an idea of the performance hit involved here.
        # log.debug(f'Is `cv_img` C-contiguous before? {cv_img.flags["C_CONTIGUOUS"]}')
        cv_img = np.ascontiguousarray(cv_img)
        # log.debug(f'Is `cv_img` C-contiguous  after? {cv_img.flags["C_CONTIGUOUS"]}')
        image = QImage(
            cv_img.data,
            cv_img.shape[1],
            cv_img.shape[0],
            cv_img.strides[0],
            QImage.Format_RGB888
        ).rgbSwapped()
        return image

    def _calc_size(self):
        '''Calculate sizing information.'''
        # Calculates self._det_scale, self._inp_width, self._inp_height
        if self.flight.is_calibrated:
            self._inp_width = self.cam.roi[2]
            self._inp_height = self.cam.roi[3]
        else:
            self._inp_width = self.flight.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self._inp_height = self.flight.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self._det_scale = float(self._inp_width) / float(ProcessorSettings.im_width)
        log.info(f'Input FPS : {self.flight.cap.get(cv2.CAP_PROP_FPS)}')
        log.info(f'Input size: {self._inp_width} x {self._inp_height} px')
        log.info(f' Full size: {self._full_frame_size}')

    def stop(self):
        '''Respond to a "nice" request to stop our processing loop.'''
        log.debug('Entering `VideoProcessor.stop()`')
        self.ret_code = ProcessorReturnCodes.UserCanceled
        # Stop all of our event loops.
        # TODO: I think there is a cleaner way to do this with QEventLoop..
        self._keep_processing = False
        self._keep_locating = False
        self.is_paused = False

    def run(self):
        '''Run the processor.'''
        log.debug('Entering `VideoProcessor.run()...`')
        # This exception handler is necessary because an exception
        # does not propagate from a thread to its calling thread.
        try:
            self._process()
        except Exception as exc:
            log.critical('An unhandled exception occurred while running VideoProcessor._process()!')
            log.critical('Exception details follow:')
            log.critical(exc)
            self.exc = exc
            self.ret_code = ProcessorReturnCodes.ExceptionOccurred
            self.finished.emit(self.ret_code)

    def _process(self):
        '''The main processing loop.'''
        log.debug('Entering `VideoProcessor._process()`')
        self._keep_processing = True
        self.ret_code = ProcessorReturnCodes.Normal
        # --- Prepare for processing
        out_video_path = self.flight.video_path.with_name(f'{self.video_name}_out.mp4')
        # Prepare the output video file
        stream_out, resize_kwarg, crop_offset, crop_idx = self._prep_video_output(out_video_path)
        # Number of total empty frames in the input.
        num_empty_frames = 0
        # Number of consecutive empty frames at beginning of capture
        num_consecutive_empty_frames = 0
        # Maximum allowed number of consecutive empty frames. If we find more, we quit.
        max_consecutive_empty_frames = 256
        # Prepare for 3D tracking
        self._prep_fig_tracking()
        # Local aids for drawing nominal paths
        self._max_draw_fit_frames = int(self._video_fps * 2.)  # multiplier is the number of seconds
        self._num_draw_fit_frames = 0
        # Misc
        frame_delta = 0
        max_frame_delta = int(self.num_input_frames * self._progress_interval / 100)
        progress = 0
        # Speed meter
        self._fps = FPS().start()
        watermark_text = f'{self.application.applicationName()} - v{self.application.applicationVersion()}'

        # ============================ PROCESSING LOOP ======================================================
        log.debug('Processing loop begins.')
        cap = self.flight.cap
        while cap.more() and self._keep_processing:
            self._frame = cap.read()

            if self._frame is None:
                num_empty_frames += 1
                num_consecutive_empty_frames += 1
                if num_consecutive_empty_frames > max_consecutive_empty_frames:
                    # GoPro videos show empty frames, quick fix
                    self.ret_code = ProcessorReturnCodes.TooManyEmptyFrames
                    log.warning(
                        f'Failed to read frame from input! '
                        f'frame_idx={self.frame_idx}/{self.num_input_frames}, '
                        f'num_empty_frames={num_empty_frames}, '
                        f'num_consecutive_empty_frames={num_consecutive_empty_frames}'
                    )
                    break
                QCoreApplication.processEvents()
                continue
            num_consecutive_empty_frames = 0

            self.frame_idx += 1

            if self.flight.is_calibrated:
                # log.debug(f'frame.shape before undistort: {self._frame.shape}')
                # log.debug(f'frame.data = {self._frame.data}')
                self._frame = self.cam.undistort(self._frame)
                # log.debug(f'frame.shape after undistort: {self._frame.shape}')
                # log.debug(f'frame.data = {self._frame.data}')
                if not self.flight.is_located and self.flight.is_ar_enabled:
                    log.debug('Locating the flight...')
                    self._frame_loc = self._frame
                    ret = self._locate()
                    self._frame_loc = None
                    # The call to self._locate() may take a long time.
                    # Restart FPS meter to be fair.
                    self._fps.start()
                    if not ret:
                        # Note that `self._locate()`` handles `self.ret_code` appropriately.
                        break

                ''' CAUTION: not for production use.
                # EXPERIMENTAL: Map the whole image frame to world sphere.
                if self.frame_idx == 1:
                    self._map_img_to_sphere(self._data_writer)
                # '''

            # TODO: should we make `im_width` variable so that it's always less than input video width?
            det_frame = resize(self._frame, width=ProcessorSettings.im_width)
            self._detector.process(det_frame)

            self._artist.draw(self._frame)

            if ProcessorSettings.perform_3d_tracking and self.flight.is_located:
                self._track_in_3d()

            # TODO: maybe `is_live` is not a necessary condition here..? Live video needs testing.
            if not self.flight.is_live and self._is_size_restorable:
                # Size us back up to original. preserve aspect, crop to middle
                self._frame = resize(self._frame, **resize_kwarg)[
                    crop_offset[1]:crop_idx[1],
                    crop_offset[0]:crop_idx[0]
                ]

            # Write text
            cv2.putText(self._frame, watermark_text, (10, 15),
                        cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 255), 1)

            # Show output
            self.new_frame_available.emit(self._cv_img_to_qimg(self._frame))

            # Save frame
            stream_out.write(self._frame)
            self._fps.update()

            # Display processing progress to user at a fixed percentage interval
            self.frame_time = self.frame_idx / self._video_fps
            progress = int(self.frame_idx / (self.num_input_frames) * 100)
            if ((self.frame_idx == 1) or
                    (progress % self._progress_interval == 0
                        and frame_delta >= max_frame_delta)):
                self.progress_updated.emit((self.frame_time, progress))
                frame_delta = 0
            frame_delta += 1

            # Optional animation effect: spin the AR sphere
            # self._azimuth += 0.4
            # self._artist.set_azimuth(self._azimuth)

            if self._clear_track_flag:
                # Clear the track, reset the flag, let the world know.
                self._detector.clear()
                self._clear_track_flag = False
                self.track_cleared.emit()

            # IMPORTANT: Let this thread process events from the calling thread,
            # otherwise they are queued up until our processing loop exits.
            QCoreApplication.processEvents()

            if self.is_paused:
                self._fps.pause()
                log.info(f'pausing at frame {self.frame_idx}/{self.num_input_frames} '
                         f'(time={timedelta(seconds=self.frame_time)})')
                self.paused.emit(True)
                # Trap us in an event loop here until resume is requested.
                while self.is_paused:
                    # TODO: Notes on future ability to have AR interaction while paused:
                    # For this to work with drawing feedback while paused, we need methods
                    # that will redraw all the elements in the frame AS IN THE NORMAL LOOP
                    # within this event loop and emit the new image on each request.
                    # I suspect this means that we need to keep a copy of the original frame
                    # before any drawing was done on it, then draw all the things on it within this loop.
                    # Right now, all events are processed during this loop
                    # but we have no visual feedback while paused.
                    # When we resume, the next frame shows the end result of all the requests.
                    # This is what we want, we just need to rethink the processing loop.
                    QCoreApplication.processEvents()
                if not self._keep_processing:
                    # A stop request was sent while paused.
                    log.info(f'Quitting from pause at frame {self.frame_idx}')
                    self.stop()
                    break
                self._fps.resume()
                self.paused.emit(False)
                log.info(f'resuming from frame {self.frame_idx}')
        # ============================ END OF PROCESSING LOOP ===============================================

        log.debug('Processing loop ended, cleaning up...')
        self._fps.stop()
        final_progress_str = f'frame_idx={self.frame_idx}, '
        f'num_input_frames={self.num_input_frames}, '
        f'num_empty_frames={num_empty_frames}, '
        f'progress={progress}%'
        elapsed_time_str = f'Elapsed time: {self._fps.elapsed():.1f}'
        mean_fps_str = f'Approx. FPS: {self._fps.fps():.1f}'
        proc_extent = 'partial'
        if self.ret_code == ProcessorReturnCodes.Normal:
            proc_extent = 'full'
        log.info(f'Finished {proc_extent} processing of {self.flight.video_path.name}')
        log.info(f'Result video written to {out_video_path.name}')
        log.info(final_progress_str)
        log.info(elapsed_time_str)
        log.info(mean_fps_str)

        if self._fig_tracker is not None:
            self._fig_tracker.finish_all()
            self._fig_tracker.export(self.flight.video_path.with_name(f'{self.video_name}_out_figures.npz'))
            self._fig_tracker = None

        # Clean up
        stream_out.release()
        if self.flight.is_located and ProcessorSettings.perform_3d_tracking:
            self._data_writer.close()

        # Exit with a code
        log.debug(f'Exiting `VideoProcessor._process()` with retcode = {self.ret_code}')
        self.finished.emit(self.ret_code)
