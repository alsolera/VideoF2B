# -*- coding: utf-8 -*-
# VideoF2B - Draw F2B figures from video
# Copyright (C) 2021 - 2022  Andrey Vasilik - basil96
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
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from imutils import resize
from imutils.video import FPS
from PySide6.QtCore import QCoreApplication, QObject, Signal
from PySide6.QtGui import QImage
from videof2b.core import figure_tracker as figtrack
from videof2b.core import projection
from videof2b.core.camera import CalCamera
from videof2b.core.common import FigureTypes, SphereManipulations, is_win
from videof2b.core.common.store import StoreProperties
from videof2b.core.detection import Detector
from videof2b.core.drawing import Drawing
from videof2b.core.flight import Flight
from videof2b.core.imaging import cv_img_to_qimg

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
    EXCEPTION_OCCURRED = -2
    # This is the code at init before the loop starts.
    UNDEFINED = -1
    # The loop exited normally.
    NORMAL = 0
    # User canceled the loop early.
    USER_CANCELED = 1
    # Pose estimation failed
    POSE_ESTIMATION_FAILED = 2
    # Too many consecutive empty frames encountered
    TOO_MANY_EMPTY_FRAMES = 3


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
    # TODO: should we make `im_width` variable so that it's always less than input video width?
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

    # pylint: disable=too-many-instance-attributes

    # --- Signals
    # Emits when cam locating begins.
    locating_started = Signal()
    # Emits when locator points changed during camera locating.
    locator_points_changed = Signal(tuple, str)
    # Emits when all required points have been defined during the locating procedure
    # so that the user can confirm and continue.
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
    # Emits when we pause/resume processing. True means we just paused, False means we just resumed.
    paused = Signal(bool)
    # Emits when an exception occurs during processing. The calling thread should know.
    error_occurred = Signal(str, str)

    def __init__(self) -> None:
        '''Create a new video processor.'''
        super().__init__()
        # Exception object, if any occurred during the main loop.
        self.exc: Exception = None
        # Return code that indicates our status when the proc loop exits.
        self.ret_code: ProcessorReturnCodes = ProcessorReturnCodes.UNDEFINED
        # Flags
        self._keep_processing: bool = False
        self._keep_locating: bool = False
        self._clear_track_flag: bool = False
        self.flight: Flight = None
        self.cam: CalCamera = None
        self._full_frame_size: Tuple[int, int] = None
        self._fourcc: cv2.VideoWriter_fourcc = None
        self._fps: FPS = None
        self._det_scale: float = None
        self._inp_width: int = None
        self._inp_height: int = None
        self._watermark_text = None
        self._resize_kwarg = None
        self._crop_offset = None
        self._crop_idx = None
        self._artist: Drawing = None
        self._frame: np.ndarray = None  # the frame currently being processed for output.
        self._frame_loc: np.ndarray = None  # the current frame during pose estimation.
        self.frame_idx: int = None
        self.frame_time: float = None
        self.progress: int = None
        # Video frame rate in frames per second.
        self._video_fps: float = None
        # Time per video frame (seconds per frame). Inverse of FPS. Used to avoid repetitive division.
        self._video_spf: float = None
        self.num_input_frames: int = -1
        self._is_size_restorable: bool = False
        self._detector: Detector = None
        self._azimuth: float = None
        self._sphere_offset = None
        self._data_writer = None
        self._fig_tracker: figtrack.FigureTracker = None
        self._is_fig_in_progress = False
        self._fig_img_pts = []
        self._fit_img_pts = []
        self._draw_fit_flag: bool = False
        self._max_draw_fit_frames: int = 1
        self._num_draw_fit_frames: int = 0
        self._video_name = None
        self._frame_delta = None
        self.is_paused = False
        # Set this flag during an event handler that requires frame update while paused.
        # This flag is just an optimization to keep the CPU
        # from being unnecessarily busy while we are paused.
        self._update_during_pause_flag = False

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
        self._video_spf = 1. / self._video_fps
        self.num_input_frames = int(self.flight.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._video_name = self.flight.video_path.stem
        self.frame_idx = 0
        self.frame_time = 0.
        self.progress = 0
        # Load camera
        self.cam = CalCamera(self._full_frame_size, self.flight)
        self.ar_geometry_available.emit(self.flight.is_located)
        # Determine input video size and the scale wrt detector's frame size
        self._calc_size()
        log.debug(f'detector im_width = {ProcessorSettings.im_width} px')
        log.debug(f'detector det_scale = {self._det_scale:.4f}')
        # Platform-dependent stuff
        self._fourcc = cv2.VideoWriter_fourcc(*'H264')
        if is_win():
            self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Detector
        max_track_len = int(ProcessorSettings.max_track_time * self._video_fps)
        self._detector = Detector(max_track_len, self._det_scale)
        # Drawing artist
        self._artist = Drawing(self._detector, cam=self.cam, flight=self.flight, axis=False)
        # Angle offset of current AR hemisphere wrt world coordinate system
        self._azimuth = 0.0
        # TODO: Like `self._azimuth`, `self._sphere_offset` really should be the driver
        #       for Drawing, but Drawing API uses incremental positioning right now.
        #       Right now the two offsets go out of sync. The effect may be visible
        #       during (re)locating if offset is nonzero.
        self._sphere_offset = self.flight.sphere_offset
        # Prepare for 3D tracking
        self._prep_fig_tracking()
        # Misc
        app_name = self.application.applicationName()
        app_version = self.application.applicationVersion()
        self._watermark_text = f'{app_name} - v{app_version}'
        self._frame_delta = 0
        self.is_paused = False
        # Emit initial progress
        self.progress_updated.emit((self.frame_time, self.progress, ''))

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
                self._resize_kwarg = {'height': self._full_frame_size[1]}
                self._crop_offset = (int(0.5*(w_final - self._full_frame_size[0])), 0)
                if w_final < self._full_frame_size[0]:
                    # The resized height if we resize width to full size
                    h_final = int(self._full_frame_size[0] / self._inp_width * self._inp_height)
                    self._resize_kwarg = {'width': self._full_frame_size[0]}
                    self._crop_offset = (0, int(0.5*(h_final - self._full_frame_size[1])))
                self._crop_idx = (self._crop_offset[0] + self._full_frame_size[0],
                                  self._crop_offset[1] + self._full_frame_size[1])
            else:
                result = cv2.VideoWriter(
                    str(path_out),
                    self._fourcc, self._video_fps,
                    (int(self._inp_width), int(self._inp_height))
                )
        else:
            live_dir_path = ProcessorSettings.live_videos
            if not live_dir_path.exists():
                live_dir_path.mkdir(parents=True)
            timestr = time.strftime("%Y%m%d-%H%M")
            result = cv2.VideoWriter(
                live_dir_path / f'out_{timestr}.mp4',
                self._fourcc, self._video_fps,
                (int(self._inp_width), int(self._inp_height))
            )
        return result

    def _prep_fig_tracking(self):
        '''Prepare processor for figure tracking, if enabled.'''
        log.info(f'3D tracking is {"ON" if ProcessorSettings.perform_3d_tracking else "OFF"}')
        if self.flight.is_calibrated and ProcessorSettings.perform_3d_tracking:
            data_path = self.flight.video_path.with_name(f'{self._video_name}_out_data.csv')
            self._data_writer = data_path.open('w', encoding='utf8')
            # self._data_writer.write('self.frame_idx,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,root1,root2\n')
            # FIXME: callback to a log file. `callback=log.debug` maybe?
            self._fig_tracker = figtrack.FigureTracker(
                callback=sys.stdout.write, enable_diags=True)
            # Aids for drawing figure start/end points over track
            self._is_fig_in_progress = False
            self._fig_img_pts = []
            # Aids for drawing nominal paths
            self._fit_img_pts = []
            self._draw_fit_flag = False
            self._max_draw_fit_frames = int(self._video_fps * 2.)  # multiplier is the number of seconds
            self._num_draw_fit_frames = 0

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
                world_pts = projection.project_image_point_to_sphere(
                    self._frame,
                    self.cam, self.flight.flight_radius, self._sphere_offset,
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
            act_pts = projection.project_image_point_to_sphere(
                self._frame, self.cam,
                self.flight.flight_radius, self._artist.center,
                self._detector.pts_scaled[0], self._data_writer
            )
            if act_pts is not None:  # and act_pts.shape[0] == 2:
                # self._fig_tracker.add_actual_point(act_pts)
                # Typically the first point is the "far" point on the sphere.
                # ...Good enough for most figures that are on the far side of the camera.
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

    def _update_frame(self, frame):
        '''Update the specified frame with all of our processing and drawing of artifacts.
            Emit the resulting QImage.
            Return the result to caller.
        '''
        # Run motion detection
        if self._clear_track_flag:
            # Clear the track, reset the flag, let the world know.
            self._detector.clear()
            self._clear_track_flag = False
            self.track_cleared.emit()
        if not self.is_paused:
            # Resize the frame for the detector and have the detector find the moving aircraft.
            self._detector.process(resize(self._frame, width=ProcessorSettings.im_width))
        # Draw most of the artifacts in the original frame.
        # This includes the detected track as well as any AR geometry, if applicable.
        self._artist.draw(frame)
        # Process 3D information if appropriate.
        if ProcessorSettings.perform_3d_tracking and self.flight.is_located:
            self._track_in_3d()
        # Restore the frame to original full size, if appropriate.
        # TODO: maybe `is_live` is not a necessary condition here..? Live video needs testing.
        if not self.flight.is_live and self._is_size_restorable:
            # Size us back up to original. preserve aspect, crop to middle
            frame = resize(frame, **self._resize_kwarg)[
                self._crop_offset[1]:self._crop_idx[1],
                self._crop_offset[0]:self._crop_idx[0]
            ]
        # Label the frame with our text.
        cv2.putText(frame, self._watermark_text, (10, 15),
                    cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 255), 1)
        # Emit the resulting frame to client.
        self.new_frame_available.emit(cv_img_to_qimg(frame))
        return frame

    def _update_progress(self, forced=False):
        '''Update processing progress and report it.'''
        # Report progress at every whole second of video.
        self.frame_time = self.frame_idx * self._video_spf
        if self.frame_time - int(self.frame_time) <= self._video_spf or forced:
            self.progress = int(self.frame_idx / (self.num_input_frames) * 100)
            self.progress_updated.emit((self.frame_time, self.progress, ''))
            self._frame_delta = 0
        self._frame_delta += 1

    def _locate(self):
        '''Interactively locate this Flight.'''
        log.debug('Entering VideoProcessor._locate()')
        log.debug(f'Asking to display the locating frame: {self._frame_loc.shape}')
        self.locating_started.emit()
        # Kind of a hack, but it works: force the first instruction signal and frame display.
        self.flight.on_locator_points_changed()
        self._keep_locating = True
        while self._keep_locating:
            # Breathe, dawg
            QCoreApplication.processEvents()
        log.debug(f'loc_pts after locating: {self.flight.loc_pts}')
        log.info('Done locating the flight.')
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
            self.ret_code = ProcessorReturnCodes.POSE_ESTIMATION_FAILED
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

    def pop_locator_point(self, _):
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
        self.new_frame_available.emit(cv_img_to_qimg(img))

    def on_locator_points_defined(self):
        '''Locator points are completely defined. Let the world know.'''
        self.locator_points_defined.emit()

    def update_figure_state(self, figure_type: FigureTypes, val: bool) -> None:
        '''Update figure state in the drawing.'''
        if self.flight.is_located:  # protect artist
            self._artist.figure_state[figure_type] = val
            self._update_during_pause_flag = True

    def update_figure_diags(self, val: bool) -> None:
        '''Update figure diags state in the drawing.'''
        if self.flight.is_located:  # protect artist
            self._artist.draw_diags = val
            self._update_during_pause_flag = True

    def mark_figure(self, is_start: bool) -> None:
        '''Mark the start/end of a tracked figure.

        :param is_start: True to mark the start of the figure,
                         False to mark the end.
        '''
        if self._fig_tracker is None or self.is_paused:
            # No effect if tracker is not active.
            # Also, do not allow this interaction while paused.
            # It causes undesirable side effects.
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
            # t = np.linspace(0., 1., self._fig_tracker.curr_figure_fitter.num_nominal_pts)
            # The figure's chosen t distribution (initial, final)
            t = (self._fig_tracker.curr_figure_fitter.diag.u0,
                 self._fig_tracker.curr_figure_fitter.u)
            log.debug(f'figure finish: shape of u0 vs. u final: {t[0].shape, t[1].shape}')
            # Trim our detected points according to the fit
            trims = self._fig_tracker.curr_figure_fitter.diag.trim_indexes
            log.debug(f'_fig_img_pts before: size = {len(self._fig_img_pts)}')
            log.debug(f'trims: shape = {trims.shape}')
            self._fig_img_pts = list(tuple(pair)
                                     for pair in np.array(self._fig_img_pts)[trims].tolist())
            log.debug(f'_fig_img_pts after: size = {len(self._fig_img_pts)}')
            # The last item is the tuple of (initial, final) fit params
            for i, fp in enumerate(self._fig_tracker.figure_params[-1]):
                nom_pts = projection.project_sphere_points_to_image(
                    self.cam, self._fig_tracker.curr_figure_fitter.get_nom_point(*fp, *t[i]))
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
        log.debug('Clearing the flight track.')
        self._update_during_pause_flag = True
        self._clear_track_flag = True

    def relocate(self):
        '''Relocate the flight.'''
        if not self.flight.is_located:
            return
        self.flight.is_located = False
        self.flight.is_ar_enabled = True
        self._frame_loc = self._frame
        self.flight.on_locator_points_changed()
        self.ar_geometry_available.emit(self.flight.is_located)

    def pause_resume(self):
        '''Pause/Resume processing at the current frame. Allows the following
        functionality with immediate feedback while paused:
        * To quit processing.
        * To clear the track.
        * To manipulate sphere rotation and movement.
        '''
        self.is_paused = not self.is_paused

    def manipulate_sphere(self, command: SphereManipulations) -> None:
        '''Manipulate the AR sphere via the specified command.'''
        if not self.flight.is_located:
            return
        self._update_during_pause_flag = True
        if command == SphereManipulations.ROTATE_CCW:
            # Rotate sphere CCW on Z axis
            self._azimuth += ProcessorSettings.sphere_rot_delta
            self._artist.set_azimuth(self._azimuth)
        elif command == SphereManipulations.ROTATE_CW:
            # Rotate sphere CW on Z axis
            self._azimuth -= ProcessorSettings.sphere_rot_delta
            self._artist.set_azimuth(self._azimuth)
        elif command == SphereManipulations.MOVE_EAST:
            # Move sphere right (+X)
            log.debug(
                f'User moves sphere center X by {ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.move_center_x(ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.MOVE_WEST:
            # Move sphere left (-X)
            log.debug(
                f'User moves sphere center X by {-ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.move_center_x(-ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.MOVE_NORTH:
            # Move sphere away from camera (+Y)
            log.debug(
                f'User moves sphere center Y by {ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.move_center_y(ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.MOVE_SOUTH:
            # Move sphere toward camera (-Y)
            log.debug(
                f'User moves sphere center Y by {-ProcessorSettings.sphere_xy_delta} '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.move_center_y(-ProcessorSettings.sphere_xy_delta)
        elif command == SphereManipulations.RESET_CENTER:
            # Reset sphere center to world origin
            log.debug(
                f'User resets sphere center '
                f'after frame {self.frame_idx} ({timedelta(seconds=self.frame_time)})')
            self._artist.reset_center()

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
        self.ret_code = ProcessorReturnCodes.USER_CANCELED
        # Stop all of our event loops.
        # TODO: I think there is a cleaner way to do this with QEventLoop..
        self._keep_processing = False
        self._keep_locating = False
        self.is_paused = False

    def run(self):
        '''Run the processor.'''
        # ===== FOR DEBUGGING ONLY. DO NOT ENABLE IN PRODUCTION. =============================
        # Allows debugging of QThreads. See https://stackoverflow.com/a/56095987/472566
        # import pydevd; pydevd.settrace(suspend=False)  # `pip install pydevd` as needed.
        # ====================================================================================

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
            self.ret_code = ProcessorReturnCodes.EXCEPTION_OCCURRED
            self.finished.emit(self.ret_code)

    def _process(self):
        '''The main processing loop.'''
        log.debug('Entering `VideoProcessor._process()`')
        self._keep_processing = True
        self.ret_code = ProcessorReturnCodes.NORMAL
        # --- Prepare for processing
        out_video_path = self.flight.video_path.with_name(f'{self._video_name}_out.mp4')
        # Prepare the output video file
        stream_out = self._prep_video_output(out_video_path)
        # Number of total empty frames in the input.
        num_empty_frames = 0
        # Number of consecutive empty frames at beginning of capture
        num_consecutive_empty_frames = 0
        # Maximum allowed number of consecutive empty frames. If we find more, we quit.
        max_consecutive_empty_frames = 256
        # True when frame was updated during pause.
        was_updated_flag = False
        # Speed meter
        self._fps = FPS().start()

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
                    self.ret_code = ProcessorReturnCodes.TOO_MANY_EMPTY_FRAMES
                    log.warning(
                        f'Failed to read frame from input! '
                        f'frame_idx={self.frame_idx}/{self.num_input_frames}, '
                        f'num_empty_frames={num_empty_frames}, '
                        f'num_consecutive_empty_frames={num_consecutive_empty_frames}'
                    )
                    break
                # Breathe, dawg
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
                if not self.flight.is_located and not self.flight.skip_locate and self.flight.is_ar_enabled:
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

                # ===========================================================
                # CAUTION: not for production use.
                # EXPERIMENTAL: Map the whole image frame to world sphere.
                # ===========================================================
                # if self.frame_idx == 1:
                #     self._map_img_to_sphere(self._data_writer)

            if self.is_paused:
                self._fps.pause()
                log.debug(f'Pausing at frame {self.frame_idx}/{self.num_input_frames} '
                          f'(time={timedelta(seconds=self.frame_time)})')
                self.paused.emit(True)
                paused_frame = self._frame
                # Update and emit a copy of the current frame because we haven't displayed it yet.
                _ = self._update_frame(paused_frame.copy())
                # Reset this flag before entering this event loop to clear all pre-pause requests.
                self._update_during_pause_flag = False
                # Trap us in an event loop here until resume is requested.
                while self.is_paused:
                    # Update only when something actually changed, then clear the request.
                    if self._update_during_pause_flag:
                        # Always send a copy of the pre-pause frame.
                        paused_frame = self._update_frame(self._frame.copy())
                        self._update_during_pause_flag = False
                        was_updated_flag = True
                    # Breathe, dawg
                    QCoreApplication.processEvents()
                if not self._keep_processing:
                    # A stop request was sent while paused.
                    log.debug(f'Quitting from pause at frame {self.frame_idx}')
                    self.stop()
                    break
                self._frame = paused_frame.copy()
                paused_frame = None
                self._fps.resume()
                self.paused.emit(False)
                log.debug(f'Resuming from frame {self.frame_idx}')

            if not was_updated_flag:
                # Update the frame with all our processing, drawing, etc., and emit it.
                # Update either here or during pause, but never both.
                self._frame = self._update_frame(self._frame)
            was_updated_flag = False

            # Save the processed frame.
            stream_out.write(self._frame)
            self._fps.update()

            # Report the processing progress to user
            self._update_progress()

            # Optional animation effect: spin the AR sphere.
            # TODO: could be an option in UI.
            # self._azimuth += 0.4
            # self._artist.set_azimuth(self._azimuth)

            # IMPORTANT: Let this thread process events from the calling thread,
            # otherwise they are queued up until our processing loop exits.
            QCoreApplication.processEvents()

        # ============================ END OF PROCESSING LOOP ===============================================

        log.debug('Processing loop ended, cleaning up...')
        cap.stop()
        self._fps.stop()
        self._update_progress(forced=True)
        elapsed_time = self._fps.elapsed()
        final_progress_str = (f'frame_idx={self.frame_idx}, '
                              f'num_input_frames={self.num_input_frames}, '
                              f'num_empty_frames={num_empty_frames}, '
                              f'progress={self.progress}%')
        elapsed_time_str = f'Elapsed time: {elapsed_time:.1f} s  [{timedelta(seconds=elapsed_time)}]'
        mean_fps_str = f'Approx. FPS: {self._fps.fps():.1f}'
        proc_extent = 'partial'
        if self.ret_code == ProcessorReturnCodes.NORMAL:
            proc_extent = 'full'
        log.info(f'Finished {proc_extent} processing of {self.flight.video_path.name}')
        log.info(f'Result video written to {out_video_path.name}')
        log.info(final_progress_str)
        log.info(elapsed_time_str)
        log.info(mean_fps_str)

        if self._fig_tracker is not None:
            self._fig_tracker.finish_all()
            self._fig_tracker.export(self.flight.video_path.with_name(f'{self._video_name}_out_figures.npz'))
            self._fig_tracker = None

        # Clean up
        stream_out.release()
        if self.flight.is_located and ProcessorSettings.perform_3d_tracking:
            self._data_writer.close()

        # Exit with a code
        log.debug(f'Exiting `VideoProcessor._process()` with retcode = {self.ret_code}')
        self.finished.emit(self.ret_code)
