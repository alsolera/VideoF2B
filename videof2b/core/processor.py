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
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import videof2b.core.figure_tracker as figtrack
import videof2b.core.projection as projection
from imutils import resize
from imutils.video import FPS
from PySide6.QtCore import QCoreApplication, QObject, QThread, Signal
from PySide6.QtGui import QImage
from videof2b.core import common
from videof2b.core.camera import CalCamera
# from videof2b.core.common import FigureTypes  # TODO: for connecting UI checkboxes to drawing of figure templates here.
from videof2b.core.common.store import StoreProperties
from videof2b.core.detection import Detector
from videof2b.core.drawing import Drawing
from videof2b.core.flight import Flight

log = logging.getLogger(__name__)


@enum.unique
class ProcessorReturnCodes(enum.IntEnum):
    '''Definition of the return codes from VideoProcessor's processing loop.'''
    # An exception occurred. Caller should check .exc for details.
    ExceptionOccurred = -2,
    # This is the code at init before the loop starts.
    Undefined = -1,
    # The loop exited normally.
    Normal = 0,
    # User cancelled the loop early.
    UserCanceled = 1,


class ProcessorSettings:
    '''Stores persistable user settings.'''
    # TODO: implement all these in the shared Settings object and get rid of this class.
    perform_3d_tracking = False  # TODO: add this option to LoadFlightDialog
    max_track_time = 15  # seconds  # TODO: add this option to LoadFlightDialog
    # width of detector frame
    im_width = 960
    sphere_xy_delta = 0.1  # XY offset delta in m
    sphere_rot_delta = 0.5  # Rotation offset delta in degrees
    live_videos = Path('../VideoF2B_videos')  # TODO: add this option to LoadFlightDialog


class VideoProcessor(QObject, StoreProperties):
    '''Main video processor. Handles processing
    of a video input from start to finish.'''

    # --- Signals
    # TODO: locator point interaction is incomplete right now. Needs rework of CalCamera first.
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
        self._det_scale: float = None
        self._inp_width: int = None
        self._inp_height: int = None
        # TODO: create the default values of the rest of the processing attributes here.
        #

    def load_flight(self, flight: Flight) -> None:
        '''Load a Flight and prepare for processing.

        :param Flight flight: a fully instantiated Flight instance.
        '''
        self.flight = flight
        self.flight.locator_points_changed.connect(self.on_locator_points_changed)
        self.flight.locator_points_defined.connect(self.on_locator_points_defined)
        self._full_frame_size = (
            int(self.flight.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(self.flight.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        # TODO: move the setup code from _process() to here. The code in _process() should contain only the loop.

    def _locate(self, frame):
        '''Interactively locate this Flight.'''
        log.debug('Entering VideoProcessor._locate()')
        log.debug(f'Asking to display the locating frame: {frame.shape}')
        # Show the requested frame.
        self.new_frame_available.emit(self._cv_img_to_qimg(frame))
        # Kind of a hack: force the first instruction signal.
        self.flight.on_locator_points_changed()
        self._keep_locating = True
        while self._keep_locating:
            time.sleep(0.010)
            # Breathe, dawg
            QCoreApplication.processEvents()
        log.debug(f'loc_pts after locating: {self.flight.loc_pts}')
        log.debug('Exiting VideoProcessor._locate()')

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
        # TODO: draw the current locator points in the locating frame.
        #
        #

    def on_locator_points_defined(self):
        '''Locator points are completely defined. Let the world know.'''
        self.locator_points_defined.emit()

    def clear_track(self):
        '''Clear the aircraft's existing flight track.'''
        log.info('Clearing the flight track.')
        self._clear_track_flag = True

    @staticmethod
    def _cv_img_to_qimg(cv_img: np.ndarray) -> QImage:
        '''Convert a cv2 image to a QImage for display in QPixmap objects.'''
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

    def calc_size(self):
        '''Calculate sizing information.'''
        if self.flight.is_calibrated:
            self._inp_width = self.cam.roi[2]
            self._inp_height = self.cam.roi[3]
        else:
            self._inp_width = self.flight.cap.get(3)
            self._inp_height = self.flight.cap.get(4)
        self._det_scale = float(self._inp_width) / float(ProcessorSettings.im_width)
        log.info(f"Input FPS : {self.flight.cap.get(5)}")
        log.info(f"Input size: {self._inp_width} x {self._inp_height}")
        return self._det_scale, self._inp_width, self._inp_height

    def stop(self):
        '''Respond to a "nice" request to stop our processing loop.'''
        log.debug('Entering `VideoProcessor.stop()`')
        self.ret_code = ProcessorReturnCodes.UserCanceled
        # Stop all of our event loops
        self._keep_processing = False
        self._keep_locating = False

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
        # This is an adaptation of this simple idea:
        # https://stackoverflow.com/questions/44404349/pyqt-showing-video-stream-from-opencv/44404713

        log.debug('Entering `VideoProcessor._process()`')
        self._keep_processing = True
        self.ret_code = ProcessorReturnCodes.Normal

        # --- Prepare for processing
        cap = self.flight.cap

        # Load camera
        self.cam = CalCamera(self._full_frame_size, self.flight)
        self.ar_geometry_available.emit(self.flight.is_calibrated)

        # Determine input video size and the scale wrt detector's frame size
        self.calc_size()
        log.info(f'processing size: {self._inp_width} x {self._inp_height} px')
        log.debug(f'detector im_width = {ProcessorSettings.im_width} px')
        log.debug(f'detector det_scale = {self._det_scale:.4f}')

        # Platform-dependent stuff
        self._fourcc = cv2.VideoWriter_fourcc(*'H264')
        if platform.system() == 'Windows':
            self._fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        video_path = self.flight.video_path
        video_name = self.flight.video_path.stem

        # Output video file
        VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)
        OUT_VIDEO_PATH = video_path.with_name(f'{video_name}_out.mp4')
        w_ratio = self._inp_width / self._full_frame_size[0]
        h_ratio = self._inp_height / self._full_frame_size[1]
        RESTORE_SIZE = w_ratio > 0.95 and h_ratio > 0.95
        log.info(f'full frame size  = {self._full_frame_size}')
        log.debug(f'input ratios w,h = {w_ratio:.4f}, {h_ratio:.4f}')
        if not self.flight.is_live:
            if RESTORE_SIZE:
                log.info(f'Output size: {self._full_frame_size}')
                out = cv2.VideoWriter(
                    str(OUT_VIDEO_PATH),
                    self._fourcc, VIDEO_FPS,
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
                crop_idx_y = crop_offset[1] + self._full_frame_size[1]
                crop_idx_x = crop_offset[0] + self._full_frame_size[0]
            else:
                out = cv2.VideoWriter(
                    str(OUT_VIDEO_PATH),
                    self._fourcc, VIDEO_FPS,
                    (int(self._inp_width), int(self._inp_height))
                )
        else:
            # TODO: clean up this path repetition
            if not ProcessorSettings.live_videos.exists():
                ProcessorSettings.live_videos.mkdir(parents=True)
            timestr = time.strftime("%Y%m%d-%H%M")
            out = cv2.VideoWriter(
                ProcessorSettings.live_videos / f'out_{timestr}.mp4',
                self._fourcc, VIDEO_FPS,
                (int(self._inp_width), int(self._inp_height))
            )

        # Track length
        max_track_len = int(ProcessorSettings.max_track_time * VIDEO_FPS)
        # Detector
        detector = Detector(max_track_len, self._det_scale)
        # Drawing artist
        artist = Drawing(detector, cam=self.cam, flight=self.flight, axis=False)
        # Angle offset of current AR hemisphere wrt world coordinate system
        azimuth_delta = 0.0
        # Number of total empty frames in the input.
        num_empty_frames = 0
        # Number of consecutive empty frames at beginning of capture
        num_consecutive_empty_frames = 0
        # Maximum allowed number of consecutive empty frames. If we find more, we quit.
        MAX_CONSECUTIVE_EMPTY_FRAMES = 256
        log.info(f'3D tracking is {"ON" if ProcessorSettings.perform_3d_tracking else "OFF"}')

        fig_tracker = None
        if self.flight.is_calibrated and ProcessorSettings.perform_3d_tracking:
            data_path = video_path.with_name(f'{video_name}_out_data.csv')
            data_writer = data_path.open('w', encoding='utf8')
            # data_writer.write('frame_idx,p1_x,p1_y,p1_z,p2_x,p2_y,p2_z,root1,root2\n')
            fig_tracker = figtrack.FigureTracker(
                callback=sys.stdout.write, enable_diags=True)  # FIXME: callback to a log file. callback=log.debug ?

        # aids for drawing figure start/end points over track
        is_fig_in_progress = False
        fig_img_pts = []
        # aids for drawing nominal paths
        MAX_DRAW_FIT_FRAMES = int(VIDEO_FPS * 2.)  # multiplier is the number of seconds
        fit_img_pts = []
        draw_fit = False
        num_draw_fit_frames = 0
        # Misc
        frame_idx = 0
        frame_delta = 0
        num_input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        MAX_FRAME_DELTA = int(num_input_frames * self._progress_interval / 100)
        is_paused = False
        progress = 0
        # Speed meter
        fps = FPS().start()
        sphere_offset = self.flight.sphere_offset
        watermark_text = f'{self.application.applicationName()} - v{self.application.applicationVersion()}'

        log.debug('Processing loop begins.')
        while cap.more() and self._keep_processing:
            frame_or = cap.read()

            if frame_or is None:
                num_empty_frames += 1
                num_consecutive_empty_frames += 1
                if num_consecutive_empty_frames > MAX_CONSECUTIVE_EMPTY_FRAMES:
                    # GoPro videos show empty frames, quick fix
                    log.warning(
                        f'Failed to read frame from input! '
                        f'frame_idx={frame_idx}/{num_input_frames}, '
                        f'num_empty_frames={num_empty_frames}, '
                        f'num_consecutive_empty_frames={num_consecutive_empty_frames}'
                    )
                    break
                time.sleep(0.001)
                continue
            num_consecutive_empty_frames = 0

            frame_idx += 1

            if self.flight.is_calibrated:
                # log.debug(f'frame_or.shape before undistort: {frame_or.shape}')
                # log.debug(f'frame_or.data = {frame_or.data}')
                frame_or = self.cam.undistort(frame_or)
                # log.debug(f'frame_or.shape after undistort: {frame_or.shape}')
                # log.debug(f'frame_or.data = {frame_or.data}')
                if not self.flight.is_located and self.flight.is_ar_enabled:
                    log.debug('Locating the flight...')
                    self._locate(frame_or)
                    log.debug('Done locating the flight.')
                    if not self._keep_processing:
                        # The processor was requested to stop during locating.
                        # Do not proceed with the rest of the processing loop.
                        log.info('A request to cancel processing was sent during locating.')
                        break
                    self.cam.locate(self.flight)
                    artist.locate(self.cam, self.flight, center=sphere_offset)
                    # The above two calls, especially self.cam.locate(), take a long time.
                    # Restart FPS meter to be fair.
                    fps.start()
                    # Signal the updated state of AR geometry availability.
                    self.ar_geometry_available.emit(self.flight.is_located)

                # TODO: refactor this into a private method here.
                '''
                # This section maps the entire image space to world, effectively meshing the sphere
                # to enable the approximation of real-world error at a given pixel.
                if frame_idx == 1:
                    import time
                    # first dimension: near/far point index
                    # second dimension: image width
                    # third dimension: image height
                    # fourth dimension: XYZ point on sphere
                    world_map = np.zeros((2, self._inp_width, self._inp_height, 3), dtype=np.float32)
                    world_map[:, :, :, :] = np.nan
                    num_pts_collected = 0
                    for v in range(self._inp_height):  # NOTE: top of sphere starts at row index 48
                        t1 = time.process_time()
                        for u in range(self._inp_width):
                            world_pts = projection.projectImagePointToSphere(self.cam, (u, v), frame_or, data_writer)
                            if world_pts is not None:
                                # print(u)
                                # print(world_pts)
                                world_map[:, u, v] = world_pts
                                num_pts_collected += 1
                        t2 = time.process_time()
                        t_diff = t2 - t1
                        print(f'Number of world points in row index {v}: {num_pts_collected}, collected in {t_diff} s')
                        num_pts_collected = 0
                    print('saving world_map array to file...')
                    np.save('world_map.npy', world_map)
                    print('...done.')
                # '''

            # TODO: should we make `im_width` variable so that it's always less than input video width?
            frame = resize(frame_or, width=ProcessorSettings.im_width)

            detector.process(frame)

            if self.flight.is_located:
                # TODO: connect `artist.figure_state` to figure checkboxes in UI
                log.warning('TODO: connect `artist.figure_state` to figure checkboxes in UI')
                # OLD CODE HERE JUST FOR REFERENCE, DELETE IT WHEN FINISHED ===========================
                # artist.figure_state[FigureTypes.INSIDE_LOOPS] = loops_chk.get()
                # artist.figure_state[FigureTypes.INSIDE_SQUARE_LOOPS] = sq_loops_chk.get()
                # artist.figure_state[FigureTypes.INSIDE_TRIANGULAR_LOOPS] = tri_loops_chk.get()
                # artist.figure_state[FigureTypes.HORIZONTAL_EIGHTS] = hor_eight_chk.get()
                # artist.figure_state[FigureTypes.HORIZONTAL_SQUARE_EIGHTS] = sq_hor_eight_chk.get()
                # artist.figure_state[FigureTypes.VERTICAL_EIGHTS] = ver_eight_chk.get()
                # artist.figure_state[FigureTypes.HOURGLASS] = hourglass_chk.get()
                # artist.figure_state[FigureTypes.OVERHEAD_EIGHTS] = over_eight_chk.get()
                # artist.figure_state[FigureTypes.FOUR_LEAF_CLOVER] = clover_chk.get()
                # artist.DrawDiags = diag_chk.get()
                # END OF OLD CODE =====================================================================

            # TODO: add a `set_azimuth` method to Drawing class and connect UI events to it. The `Drawing.draw` method should just take an image frame as input.
            artist.draw(frame_or, azimuth_delta)

            if self.flight.is_located:
                if ProcessorSettings.perform_3d_tracking:
                    # try to track the aircraft in world coordinates
                    if detector.pts_scaled[0] is not None:
                        act_pts = projection.projectImagePointToSphere(
                            self.cam, artist.center, detector.pts_scaled[0], frame_or, data_writer)
                        if act_pts is not None:  # and act_pts.shape[0] == 2:
                            # fig_tracker.add_actual_point(act_pts)
                            # Typically the first point is the "far" point on the sphere...Good enough for most figures that are on the far side of the camera.
                            # TODO: Make this smarter so that we track the correct path point at all times.
                            fig_tracker.add_actual_point(frame_idx, act_pts[0])
                            # fig_tracker.add_actual_point(act_pts[1])
                            if is_fig_in_progress:
                                fig_img_pts.append(detector.pts_scaled[0])

                    # ========== Draw the fitted figures ==========
                    if draw_fit and num_draw_fit_frames < MAX_DRAW_FIT_FRAMES:
                        # Draw the nominal paths: initial and best-fit
                        for i, f_pts in enumerate(fit_img_pts):
                            # Draw the initial guess (the defined nominal path) in GREEN
                            # Draw the optimized fit path in CYAN
                            f_color = (0, 255, 0) if i == 0 else (255, 255, 0)
                            for j in range(1, len(f_pts)):
                                cv2.line(frame_or, f_pts[j], f_pts[j-1], f_color, 2)
                                # Draw the error whiskers. TODO: verify the arrays are correct after figure.u is trimmed.
                                if i == 1:
                                    cv2.line(frame_or, fig_img_pts[j], f_pts[j], (255, 255, 255), 1)
                        num_draw_fit_frames += 1
                    if num_draw_fit_frames == MAX_DRAW_FIT_FRAMES:
                        draw_fit = False
                        fit_img_pts = []
                        fig_img_pts = []
                        num_draw_fit_frames = 0

                    # Draw the start/end points of the figure's actual tracked path (as marked by user)
                    if fig_img_pts:
                        cv2.circle(frame_or, fig_img_pts[0], 6, (0, 255, 255), -1)
                        if not is_fig_in_progress:
                            cv2.circle(frame_or, fig_img_pts[-1], 6, (255, 0, 255), -1)

            if not self.flight.is_live and RESTORE_SIZE:
                # Size us back up to original. preserve aspect, crop to middle
                frame_or = resize(frame_or, **resize_kwarg)[
                    crop_offset[1]:crop_idx_y,
                    crop_offset[0]:crop_idx_x]

            # Write text
            cv2.putText(frame_or, watermark_text, (10, 15),
                        cv2.FONT_HERSHEY_TRIPLEX, .5, (0, 0, 255), 1)

            # Show output
            self.new_frame_available.emit(self._cv_img_to_qimg(frame_or))

            # Save frame
            out.write(frame_or)
            fps.update()

            # Display processing progress to user at a fixed percentage interval
            frame_time = frame_idx / VIDEO_FPS
            progress = int(frame_idx / (num_input_frames) * 100)
            if ((frame_idx == 1) or
                    (progress % self._progress_interval == 0
                        and frame_delta >= MAX_FRAME_DELTA)):
                self.progress_updated.emit((frame_time, progress))
                frame_delta = 0
            frame_delta += 1

            # azimuth_delta += 0.4  # For quick visualization of sphere outline

            if self._clear_track_flag:
                # Clear the track, reset the flag, let the world know.
                detector.clear()
                self._clear_track_flag = False
                self.track_cleared.emit()

            # IMPORTANT: Let this thread process events from the calling thread,
            # otherwise they are queued up until our processing loop exits.
            QCoreApplication.processEvents()

            # TODO: refactor this huge if block into handlers for signals from UI.
            # =============================================================================================================
            # key = cv2.waitKeyEx(1)  # & 0xFF
            # LSB = key & 0xff

            # if key != -1:
            #     print(f'\nKey: {key}  LSB: {key & 0xff}')

            # if LSB == 27 or key == 1048603 or cv2.getWindowProperty(WINDOW_NAME, 1) < 0:  # ESC
            #     # Exit
            #     break
            # elif LSB == 112:  # p
            #     # Pause/Resume with ability to quit while paused
            #     # TODO: allow sphere rotation and movement while paused!
            #     fps.pause()
            #     is_paused = True
            #     pause_str = f'pausing at frame {frame_idx}/{num_input_frames} (time={timedelta(seconds=frame_time)})'
            #     print(f'\n{pause_str}')
            #     log.info(pause_str)
            #     get_out = False
            #     while is_paused:
            #         key = cv2.waitKeyEx(100)
            #         LSB = key & 0xff
            #         if LSB == 27 or cv2.getWindowProperty(WINDOW_NAME, 1) < 0:  # ESC
            #             get_out = True
            #             break
            #         is_paused = not (LSB == 112)
            #     if get_out:
            #         log.info(f'quitting from pause at frame {frame_idx}')
            #         break
            #     fps.resume()
            #     resume_str = f'resuming from frame {frame_idx}'
            #     print(resume_str)
            #     log.info(resume_str)
            # elif LSB == 32 or key == 1048608:  # Space
            #     # Clear the current track
            #     detector.clear()
            # elif key == 1113939 or key == 2555904 or key == 65363:  # Right Arrow
            #     azimuth_delta -= 0.5
            # elif key == 1113937 or key == 2424832 or key == 65361:  # Left Arrow
            #     # Positive CCW around Z, per normal convention
            #     azimuth_delta += 0.5
            # elif LSB == 99 or key == 1048675:  # c
            #     # Relocate AR sphere
            #     self.flight.is_located = False
            #     self.flight.is_ar_enabled = True
            # elif ProcessorSettings.perform_3d_tracking and LSB == 91:  # [ key
            #     # Mark the beginning of a new figure
            #     if fig_tracker is not None:
            #         detector.clear()
            #         fig_tracker.start_figure()
            #         is_fig_in_progress = True
            # elif ProcessorSettings.perform_3d_tracking and LSB == 93:  # ] key
            #     # Mark the end of the current figure
            #     if fig_tracker is not None:
            #         print(f'before finishing figure: fig_img_pts size ={len(fig_img_pts)}')
            #         fig_tracker.finish_figure()
            #         is_fig_in_progress = False
            #         # Uniform t along the figure path
            #         # t = np.linspace(0., 1., fig_tracker._curr_figure_fitter.num_nominal_pts)
            #         # The figure's chosen t distribution (initial, final)
            #         t = (fig_tracker._curr_figure_fitter.diag.u0,
            #              fig_tracker._curr_figure_fitter.u)
            #         print(f'figure finish: shape of u0 vs. u final: {t[0].shape, t[1].shape}')
            #         # Trim our detected points according to the fit
            #         trims = fig_tracker._curr_figure_fitter.diag.trim_indexes
            #         print(f'fig_img_pts before: size = {len(fig_img_pts)}')
            #         print(f'trims: shape = {trims.shape}')
            #         fig_img_pts = list(tuple(pair)
            #                            for pair in np.array(fig_img_pts)[trims].tolist())
            #         print(f'fig_img_pts after: size = {len(fig_img_pts)}')
            #         # The last item is the tuple of (initial, final) fit params
            #         for i, fp in enumerate(fig_tracker.figure_params[-1]):
            #             nom_pts = projection.projectSpherePointsToImage(
            #                 self.cam, fig_tracker._curr_figure_fitter.get_nom_point(*fp, *t[i]))
            #             fit_img_pts.append(tuple(map(tuple, nom_pts.tolist())))
            #         # print(f't ({len(t)} pts)')
            #         print(f'fit_img_pts[initial] ({len(fit_img_pts[0])} pts)')
            #         # # print(fit_img_pts[0])
            #         print(f'fit_img_pts[final] ({len(fit_img_pts[1])} pts)')
            #         # # print(fit_img_pts[1])
            #         print(f'fig_img_pts ({len(fig_img_pts)} pts)')
            #         # # print()
            #         # Set the flag for drawing the fit figures, diags, etc.
            #         draw_fit = True
            # elif LSB == ord('d'):
            #     # offset sphere right (+X)
            #     log.info(
            #         f'User moves sphere center X by {ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterX(ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('a'):
            #     # offset sphere left (-X)
            #     log.info(
            #         f'User moves sphere center X by {-ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterX(-ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('w'):
            #     # offset sphere away from camera (+Y)
            #     log.info(
            #         f'User moves sphere center Y by {ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterY(ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('s'):
            #     # offset sphere toward camera (-Y)
            #     log.info(
            #         f'User moves sphere center Y by {-ProcessorSettings.sphere_xy_delta} after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.MoveCenterY(-ProcessorSettings.sphere_xy_delta)
            # elif LSB == ord('x'):
            #     # reset sphere offset to world origin
            #     log.info(
            #         f'User resets sphere center after frame {frame_idx} ({timedelta(seconds=frame_time)})')
            #     artist.ResetCenter()

        log.debug('Processing loop ended, cleaning up...')
        fps.stop()
        final_progress_str = f'frame_idx={frame_idx}, num_input_frames={num_input_frames}, num_empty_frames={num_empty_frames}, progress={progress}%'
        elapsed_time_str = f'Elapsed time: {fps.elapsed():.1f}'
        mean_fps_str = f'Approx. FPS: {fps.fps():.1f}'
        proc_relativity = 'partial'
        if self.ret_code == ProcessorReturnCodes.Normal:
            proc_relativity = 'full'
        log.info(f'Finished {proc_relativity} processing of {self.flight.video_path.name}')
        log.info(f'Result video written to {OUT_VIDEO_PATH.name}')
        log.info(final_progress_str)
        log.info(elapsed_time_str)
        log.info(mean_fps_str)

        if fig_tracker is not None:
            fig_tracker.finish_all()
            fig_tracker.export(video_path.with_name(f'{video_name}_out_figures.npz'))

        # Clean up
        out.release()
        if self.flight.is_located and ProcessorSettings.perform_3d_tracking:
            data_writer.close()

        # Exit with a code
        log.debug(f'Exiting `VideoProcessor._process()` with retcode = {self.ret_code}')
        self.finished.emit(self.ret_code)
